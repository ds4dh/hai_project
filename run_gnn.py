import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils.convert import from_networkx
from sklearn.metrics import classification_report
from data.data_utils import load_features_and_labels, load_edges


N_TRAIN_EPOCHS = 400
LR = 1e-2
N_HIDDEN = 32
LAYER_TYPE = 'sage'  # 'gcn', 'sage', 'gat'
SETTING_CONDS = ['transductive', 'inductive']
BALANCED_CONDS = ['over', 'under', 'non']
LINK_CONDS = ['all', 'wards', 'caregivers']
DATA_DIR = os.path.join('data', 'processed')
TEST_ONLY = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    """ Train a GNN in different settings, data balance and link conditions
    """
    for setting_cond in SETTING_CONDS:
        for balanced_cond in BALANCED_CONDS:
            for link_cond in LINK_CONDS:
                ckpt_path = os.path.join('models',
                                         'gnn',
                                         '%s_setting' % setting_cond,
                                         '%s_balanced' % balanced_cond,
                                         'links_%s' % link_cond)
                print('New simulation: %s setting, %s-balanced data, %s links'%
                      (setting_cond, balanced_cond, link_cond))
                train_model(setting_cond, balanced_cond, link_cond, ckpt_path)
                

def train_model(setting: str,
                balanced: str,
                link: str,
                ckpt_path: str):
    """ Train a GNN model to predict HAI from patient nodes and worker edges
    """
    # Initialize data and model
    dataset = IPCPredict(balanced, link, setting)
    model = Net(dataset)
    
    # Train model if required
    if not TEST_ONLY:
        train_loss_plot, dev_loss_plot = train_network(model, dataset, setting)
        os.makedirs(os.path.split(ckpt_path)[0], exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
    else:
        train_loss_plot, dev_loss_plot = [], []
    
    # Reload checkpoint and test model
    model.load_state_dict(torch.load(ckpt_path))
    result_text = evaluate_model(model, dataset, setting)

    # Plot training process (if any) and testing metrics
    result_path = ckpt_path.replace('.pt', '.png')
    plot_model_results(train_loss_plot, dev_loss_plot, result_text, result_path)


def train_network(model: nn.Module,
                  dataset: InMemoryDataset,
                  setting: str):
    """ Train a network to predict bacterial colonisation in hospitals
    """    
    # Load data (inductive or transductive setting)
    if setting == 'inductive':
        train_data = dataset.get_split('train').to(DEVICE)
        dev_data = dataset.get_split('dev').to(DEVICE)
        train_golds = train_data.y
        dev_golds = dev_data.y
    elif setting == 'transductive':
        whole_data = dataset.get_split('whole').to(DEVICE)
        train_mask = whole_data.masks['train']
        dev_mask = whole_data.masks['dev']
        train_golds = whole_data.y[train_mask]
        dev_golds = whole_data.y[dev_mask]

    # Set weights for class imbalance when computing loss
    wb = ((train_golds == 0).sum() / (train_golds == 1).sum()).item()
    train_weights = torch.tensor([1 if g == 0 else wb for g in train_golds])
    dev_weights = torch.tensor([1 if g == 0 else wb for g in dev_golds])

    # Initialize model and learning
    train_criterion = nn.BCEWithLogitsLoss(weight=train_weights.to(DEVICE))
    dev_criterion = nn.BCEWithLogitsLoss(weight=dev_weights.to(DEVICE))
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, LR, total_steps=N_TRAIN_EPOCHS)
    
    # Start training model
    print(' - Training GNN model (%s-conv layers)' % LAYER_TYPE)
    train_loss_plot, dev_loss_plot = [], []
    for epoch in range(N_TRAIN_EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        # Compute training loss (inductive or transductive setting)
        if setting == 'inductive':
            train_logits = model(train_data.x, train_data.edge_index)
            train_probs = torch.sigmoid(train_logits).view(-1)
        elif setting == 'transductive':
            whole_logits = model(whole_data.x, whole_data.edge_index)
            whole_probs = torch.sigmoid(whole_logits)
            train_probs = whole_probs[train_mask].view(-1)
        train_loss = train_criterion(train_probs, train_golds)
            
        # Perform backpropagation
        train_loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Compute validation loss (inductive or transductive setting)
        model.eval()
        with torch.no_grad():
            if setting == 'inductive':
                dev_logits = model(dev_data.x, dev_data.edge_index)
                dev_probs = torch.sigmoid(dev_logits).view(-1)
            elif setting == 'transductive':
                dev_probs = whole_probs[dev_mask].view(-1)
            dev_loss = dev_criterion(dev_probs, dev_golds)

        # Record and report training loss
        train_loss_plot.append(train_loss.item())
        dev_loss_plot.append(dev_loss.item())
        print('\r ---- Epoch %03d: loss = %.03f' % (epoch, train_loss), end='')
    
    # Return the output of training (and also: model has been updated)
    print('\n - Training finished')
    return train_loss_plot, dev_loss_plot


def evaluate_model(model: nn.Module,
                   dataset: InMemoryDataset,
                   setting: str,
                   thresh: float=0.9):
    """ Generates predictions with a trained network and report various metrics
    """
    # TODO: DO LIKE IN RUN_CONTROLS WITH COMPUTATION OF BEST THRESHOLD ETC!
    # Initialize model, data and labels
    print(' - Testing model')
    model.eval()
    if setting == 'inductive':
        test_data = dataset.get_split('dev').to(DEVICE)  # with dev for now
        test_golds = test_data.y
    elif setting == 'transductive':
        whole_data = dataset.get_split('whole').to(DEVICE)
        test_mask = whole_data.masks['dev']  # with dev for now
        test_golds = whole_data.y[test_mask]

    # Compute model predictions
    with torch.no_grad():
        if setting == 'inductive':
            test_logits = model(test_data.x, test_data.edge_index)
            test_probs = torch.sigmoid(test_logits).view(-1)
        elif setting == 'transductive':
            whole_logits = model(whole_data.x, whole_data.edge_index)
            whole_preds = torch.sigmoid(whole_logits)
            test_probs = whole_preds[test_mask].view(-1)

    # Compute all metrics using model predictions
    if DEVICE.type != 'cpu':
        test_probs = test_probs.cpu().numpy()
        test_golds = test_golds.cpu().numpy()
    test_preds = [0 if p < thresh else 1 for p in test_probs]
    return classification_report(test_golds, test_preds)


def plot_model_results(train_loss_plot: list[float],
                       dev_loss_plot: list[float],
                       result_text: str,
                       result_path) -> None:
    """ Plot a summary of the training process (training loss vs epoch)
    """
    _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(train_loss_plot)), train_loss_plot, label='Train loss')
    ax.plot(range(len(dev_loss_plot)), dev_loss_plot, label='Dev loss')
    ax.text(0.95, 0.95, result_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='square', facecolor='white'))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training loss')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(result_path, dpi=300)


class Net(nn.Module):
    """ Graph neural network that takes patient network into account
    """
    def __init__(self, dataset: InMemoryDataset):
        super(Net, self).__init__()
        if LAYER_TYPE == 'gcn':
            self.conv1 = GCNConv(dataset.num_features, N_HIDDEN)
            self.conv2 = GCNConv(N_HIDDEN, 1)
        elif LAYER_TYPE == 'sage':
            self.conv1 = SAGEConv(dataset.num_features, N_HIDDEN)
            self.conv2 = SAGEConv(N_HIDDEN, 1)
        elif LAYER_TYPE == 'gat':
            n_heads = 8  # 1
            assert N_HIDDEN % n_heads == 0
            n_out = N_HIDDEN // n_heads
            self.conv1 = GATConv(dataset.num_features, n_out, heads=n_heads)
            self.conv2 = GATConv(N_HIDDEN, 1)
        self.to(DEVICE)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class IPCPredict(InMemoryDataset):
    """ Dataset containing graph data, and node indices for train, dev and test
    """
    def __init__(self, balanced_cond, link_cond, setting_cond):
        super(IPCPredict, self).__init__()
        print(' - Creating data graph')
        # Load features, labels, node ids, and initialize data
        X, y, ids = load_features_and_labels(balanced_cond)
        graph_list = []

        # Create graph for a transductive setting
        if setting_cond == 'transductive':
            graph = self.create_transductive_graph(X, y, ids, link_cond)
            graph_list.append(graph)
        
        # Create graphs for an inductive setting
        elif setting_cond == 'inductive':
            for split in ['train', 'dev', 'test']:
                X_, y_, ids_ = X[split], y[split], ids[split]
                graph = self.create_graph(X_, y_, ids_, link_cond)
                graph_list.append(graph)

        # Define splits and collate data into a nice dataset
        self.split_indices = {'whole': 0, 'train': 0, 'dev': 1, 'test': 2}
        self.data, self.slices = self.collate(graph_list)
    
    def get_split(self, name):
        """ Workaround to get dataset splits by split name instead of indices
        """
        return self[self.split_indices[name]]
    
    def create_transductive_graph(self,
                                  X: dict[np.ndarray],
                                  y: dict[pd.Series],
                                  ids: dict[pd.Index],
                                  link_cond: str):
        """ Create transductive graph using nodes, node labels, node ids, and
            appends train, dev, and test masks to the graph
        """
        # Create graph using the totality of nodes, node labels, and node ids
        features = np.concatenate((X['train'], X['dev'], X['test']))
        labels = np.concatenate((y['train'], y['dev'], y['test']))
        ids = np.concatenate((ids['train'], ids['dev'], ids['test']))
        pyg_graph = self.create_graph(features, labels, ids, link_cond)
        
        # Create masks to retrieve train, dev, and test predictions
        masks = {k: torch.zeros(features.shape[0], dtype=torch.bool)
                 for k in ('train', 'dev', 'test')}
        masks['train'][:len(X['train'])] = True
        masks['dev'][len(X['train']):len(X['train']) + len(X['dev'])] = True
        masks['test'][-len(X['test']):] = True
        
        # Return final graph, after adding transductive masks
        pyg_graph.masks = masks
        return pyg_graph
    
    @staticmethod
    def create_graph(X: np.ndarray,
                     y: pd.Series,
                     ids: pd.Index,
                     link_cond: str):
        """ Create graph using nodes, node labels, and node ids
        """
        # Initialize graph and add nodes features and labels
        nx_graph = nx.Graph()
        for node_id, feat, lbl in zip(ids, X, y):
            node_info = {'x': feat.tolist(), 'y': float(lbl)}
            nx_graph.add_node(node_id, **node_info)
        
        # Add edges
        edges = load_edges(link_cond, ids)
        nx_graph.add_edges_from(edges.values)

        # Return pytorch-geometric object from the graph
        return from_networkx(nx_graph)

if __name__ == '__main__':
    main()
