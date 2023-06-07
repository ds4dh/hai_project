import os
import numpy as np
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


TRAIN_MODE = 'transductive'  # 'transductive', 'inductive' (not working for now)
N_TRAIN_EPOCHS = 400
LR = 1e-2
N_HIDDEN = 32
LAYER_TYPE = 'sage'  # 'gcn', 'sage', 'gat'
BALANCED_COND = 'non'  # 'non', 'over', 'under'
LINK_COND = 'wards'  # 'all', 'wards', 'caregivers'
DATA_DIR = os.path.join('data', 'processed')
LINK_PATH = os.path.join(DATA_DIR, 'graph_links_%s.csv' % LINK_COND)
CKPT_DIR = os.path.join('models', 'GNN', 'ckpts', '%s_balanced' % BALANCED_COND)
CKPT_PATH = os.path.join(CKPT_DIR, 'graph_links_%s' % LINK_COND)
TEST_ONLY = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    """ Train a GCN to predict HAI from patient nodes and worker edges
    """
    # Initialize data and model
    ckpt_path = os.path.join(CKPT_DIR, CKPT_PATH)
    dataset = IPCPredict()
    model = Net(dataset)

    # Train model if required
    if not TEST_ONLY:
        train_loss_plot, dev_loss_plot = train_network(model, dataset)
        os.makedirs(os.path.split(ckpt_path)[0], exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
    else:
        train_loss_plot, dev_loss_plot = [], []
    
    # Reload checkpoint and test model
    model.load_state_dict(torch.load(ckpt_path))
    result_text = test_network(model, dataset)

    # Plot training process (if any) and testing metrics
    plot_model_results(train_loss_plot, dev_loss_plot, result_text)
 

def train_network(model: nn.Module, dataset: InMemoryDataset):
    """ Train a network to predict bacterial colonisation in hospitals
    """    
    # Initialize data variables
    train_mask, dev_mask = dataset.split_ids['train'], dataset.split_ids['dev']
    whole_data = dataset[0]
    if TRAIN_MODE == 'inductive':
        train_data = whole_data[train_mask]
        dev_data = whole_data[dev_mask]
    train_golds = whole_data.y[train_mask]
    dev_golds = whole_data.y[dev_mask]
    wb = ((train_golds == 0).sum() / (train_golds == 1).sum()).item()
    train_weights = torch.tensor([1 if g == 0 else wb for g in train_golds])
    dev_weights = torch.tensor([1 if g == 0 else wb for g in dev_golds])

    # Initialize model and learning
    model.train()
    train_criterion = nn.BCEWithLogitsLoss(weight=train_weights.to(DEVICE))
    dev_criterion = nn.BCEWithLogitsLoss(weight=dev_weights.to(DEVICE))
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, LR, total_steps=N_TRAIN_EPOCHS)
    
    # Start training model
    print('Training GNN model (%s-conv layers)' % LAYER_TYPE)
    train_loss_plot, dev_loss_plot = [], []
    for epoch in range(N_TRAIN_EPOCHS):
        optimizer.zero_grad()

        # Different ways of including features at training time
        if TRAIN_MODE == 'inductive':
            # TODO: define an inductive dataset that is composed of *several* graphs (one for each split)
            raise NotImplementedError('Not working for now. Stay tuned!')
            train_logits = model(train_data)
            dev_logits = model(dev_data)
            preds = torch.sigmoid(logits).view(-1)  # no mask for inductive, because data is just the training graph
        elif TRAIN_MODE == 'transductive':
            logits = model(whole_data.x, whole_data.edge_index)
            preds = torch.sigmoid(logits)
            train_preds = preds[train_mask].view(-1)
            with torch.no_grad():  # required?
                dev_preds = preds[dev_mask].view(-1)
        
        # Compute loss and perform backpropagation
        train_loss = train_criterion(train_preds, train_golds)
        train_loss.backward()
        with torch.no_grad():  # required?
            dev_loss = dev_criterion(dev_preds, dev_golds)
        optimizer.step()
        scheduler.step()

        # Record and report training loss
        train_loss_plot.append(train_loss.item())
        dev_loss_plot.append(dev_loss.item())
        print('\r - Epoch %03d: loss = %.03f' % (epoch, train_loss), end='')
    
    # Return the output of training (and also: model has been updated)
    print('\nTraining finished')
    return train_loss_plot, dev_loss_plot


def test_network(model, dataset, thresh=0.9):
    """ Generates predictions with a trained network and report various metrics
    """
    # Initialize model, data and labels
    print('Testing model')
    model.eval()
    test_mask = dataset.split_ids['dev']
    whole_data = dataset[0]
    golds = whole_data.y[test_mask]

    # Compute model predictions (using the whole graph, i.e., transductive mode)
    logits = model(whole_data.x, whole_data.edge_index)
    probs = torch.sigmoid(logits)[test_mask]
    preds = [0 if p[0] < thresh else 1 for p in probs]  # do different thresholds?
    
    # Compute all metrics using model predictions
    return classification_report(golds, preds)


def plot_model_results(train_loss_plot: list[float],
                       dev_loss_plot: list[float],
                       result_text: str) -> None:
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
    result_path = os.path.join(CKPT_DIR, CKPT_PATH.replace('.pt', '.png'))
    plt.legend()
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
    def __init__(self, transform=None):
        super(IPCPredict, self).__init__('.', transform, None, None)  # what????
        print('Creating data graph')
        graph, split_ids = self.create_graph_and_split_indices()
        data = Data(x=graph.x, y=graph.y, edge_index=graph.edge_index)
        data, _ = self.collate([data])
        data.to(DEVICE)
        self.data = data
        self.split_ids = split_ids

    @staticmethod
    def create_graph_and_split_indices() -> nx.Graph:
        """ Retrieve patient ward data and create a graph, where patients are
            nodes and edges are hospital workers (?)
        """
        # Load features and labels, then initialize the graph
        X_train, X_dev, X_test, y_train, y_dev, y_test =\
             load_features_and_labels(BALANCED_COND)
        node_features = np.concatenate((X_train, X_dev, X_test))
        node_labels = np.concatenate((y_train, y_dev, y_test))
        
        # Retrieve split indices for node-edge correspondance
        split_ids = {
            'train': range(0, len(X_train)),
            'dev': range(len(X_train), len(X_train) + len(X_dev)),
            'test': range(len(X_train) + len(X_dev), len(node_features)),
        }
        
        # Creates graph nodes and edges
        nx_graph = nx.Graph()
        for node_id, node in enumerate(node_features):
            nx_graph.add_node(node_id, x=node.tolist())
        edges = load_edges(link_cond=LINK_COND)
        nx_graph.add_edges_from(edges.values)
        
        # Create and return pytorch-geometric object from the graph
        pyg_graph = from_networkx(nx_graph)
        pyg_graph.y = torch.tensor(node_labels, dtype=torch.float)
        return pyg_graph, split_ids


if __name__ == '__main__':
    main()
