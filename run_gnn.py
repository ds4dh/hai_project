import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.data_utils import load_data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils.convert import from_networkx
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, roc_auc_score


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
    dataset = IPCPredict(balanced=True)
    model = Net(dataset)

    # Train model if required
    plot_losses = []
    if not TEST_ONLY:
        plot_losses = train_network(model, dataset)
        torch.save(model.state_dict(), ckpt_path)
    
    # Reload checkpoint and test model
    model.load_state_dict(torch.load(ckpt_path))
    result_text = test_network(model, dataset)

    # Plot training process (if any) and testing metrics
    plot_model_results(plot_losses, result_text)


def train_network(model: nn.Module, dataset: InMemoryDataset):
    """ Train a network to predict bacterial colonisation in hospitals
    """    
    # Initialize data variables
    loss_plot = []
    data = dataset[0].to(DEVICE)
    golds = data.y[data['train_mask']].to(torch.float)
    weight_balance = ((golds == 0).sum() / (golds == 1).sum()).item()
    weights = torch.tensor([1 if g == 0 else weight_balance for g in golds])

    # Initialize model and learning
    model.train()
    criterion = nn.BCEWithLogitsLoss(weight=weights.to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, LR, total_steps=N_TRAIN_EPOCHS)

    # Start training model
    print('Training GNN model (%s-conv layers)' % LAYER_TYPE)
    for epoch in range(N_TRAIN_EPOCHS):
        optimizer.zero_grad()

        # Different ways of including features at training time
        if TRAIN_MODE == 'inductive':
            # TODO: define an inductive dataset that is composed of *several* graphs (one for each split)
            raise NotImplementedError('Not working for now. Stay tuned!')
            logits = model(data)
            preds = torch.sigmoid(logits).view(-1)  # no mask for inductive, because data is just the training graph
        elif TRAIN_MODE == 'transductive':
            logits = model(data)
            preds = torch.sigmoid(logits)[data['train_mask']].view(-1)
        
        # Backpropagation
        loss = criterion(preds, golds)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Record and report training loss
        loss_plot.append(loss.item())
        print('\r - Epoch %03d: loss = %.03f' % (epoch, loss), end='')
    
    # Return the output of training (and also: model has been updated)
    print('\nTraining finished')
    return loss_plot


def test_network(model, dataset, thresh=0.9):
    """ Generates predictions with a trained network and report various metrics
    """
    # Initialize model, data and labels
    print('Testing model')
    model.eval()
    data = dataset[0].to(DEVICE)
    golds = data.y[data['test_mask']].tolist()

    # Compute model predictions (using the whole graph, i.e., transductive mode)
    logits = model(data)
    probs = torch.sigmoid(logits)[data['test_mask']].tolist()
    preds = [0 if p[0] < thresh else 1 for p in probs]  # do different thresholds?
    
    # Compute all metrics using model predictions
    return classification_report(golds, preds)


def plot_model_results(plot_losses: list[float], result_text: str) -> None:
    """ Plot a summary of the training process (training loss vs epoch)
    """
    _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(plot_losses)), plot_losses)
    ax.text(0.95, 0.95, result_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='square', facecolor='white'))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training loss')
    result_path = os.path.join(CKPT_DIR, CKPT_PATH.replace('.pt', '.png'))
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

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.to(torch.float32)  # TODO: put this in dataset, not here
        edge_index = edge_index.long()  # TODO: put this in dataset, not here
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class IPCPredict(InMemoryDataset):
    """ Dataset containing graph data, and node indices for train, dev and test
    """
    def __init__(self, transform=None, balanced=True):
        super(IPCPredict, self).__init__('.', transform, None, None)  # check path???
        print('Creating data graph')
        pyg_graph = self.create_graph()
        data = Data(x=pyg_graph.x, y=pyg_graph.y, edge_index=pyg_graph.edge_index)
        data.num_nodes = len(pyg_graph.y)
        data.num_classes = 2  # used?
        self.balanced = balanced
        print('Computing data splits')
        self.add_train_dev_test_masks_to_data(data)
        print('Building datasets')
        self.data, _ = self.collate([data])
    
    def add_train_dev_test_masks_to_data(self, data):
        """ Add boolean tensors to the data to get train, dev and test splits 
        """
        # Load split indices
        df_train = pd.read_pickle(os.path.join(PKL_DIR, 'X_train.pkl'))
        df_dev = pd.read_pickle(os.path.join(PKL_DIR, 'X_dev.pkl'))
        df_test = pd.read_pickle(os.path.join(PKL_DIR, 'X_test.pkl'))

        # # This is because 'index' is a special word for pandas
        # df_train.reset_index(inplace=True)
        # df_train = df_train.rename(columns={'index': 'id'})
        # if not BALANCED:  # why this???
        #     df_dev.reset_index(inplace=True)
        #     df_dev = df_dev.rename(columns={'index': 'id'})
        #     df_test.reset_index(inplace=True)
        #     df_test = df_test.rename(columns={'index': 'id'})
        
        # Retrieve train, dev and test masks from the split indices
        train_ids, dev_ids, test_ids = df_train.id.unique(), df_dev.id.unique(), df_test.id.unique()
        data['train_mask'] = torch.tensor([i in train_ids for i in range(data.num_nodes)])
        data['dev_mask'] = torch.tensor([i in dev_ids for i in range(data.num_nodes)])
        data['test_mask'] = torch.tensor([i in test_ids for i in range(data.num_nodes)])
    
    @staticmethod
    def create_graph() -> nx.Graph:
        """ Retrieve patient ward data and create a graph, where patients are
            nodes and edges are hospital workers (?)
        """
        # Initialize graph and retrieve raw data
        G = nx.Graph()
        X_train, X_dev, X_test, y_train, y_dev, y_test = load_data(BALANCED_COND)
        import pdb; pdb.set_trace()
        inductive_node_features = 
        transductive_node_features = np.concatenate((X_train, X_dev, X_test))

        # # Join all node data files in a single dataframe
        # node_df.reset_index(inplace=True)
        # node_df = node_df.rename(columns={'index': 'id'})
        # node_df = node_df.fillna(0)
        # node_df = node_df.loc[node_df['id'] != -1]
        # node_df = node_df.filter(items=DF_FEATURES)
        # node_df = node_df.drop_duplicates()
        
        # # Transform string column values to one hot encoding
        # for column_name in STRING_FEATURES:
        #     node_df[column_name] = pd.Categorical(node_df[column_name])
        #     one_hot = pd.get_dummies(node_df[column_name], prefix=column_name)
        #     node_df = pd.concat([node_df, one_hot], axis=1)
        #     node_df = node_df.drop(column_name, axis=1)
            
        # # Standardize numerical columns
        # for column_name in NUMERICAL_FEATURES:
        #     mean, std = node_df[column_name].mean(), node_df[column_name].std()
        #     node_df[column_name] = (node_df[column_name] - mean) / std
        # node_df['id'] = node_df['id'].astype(int)
        # TODO: LOAD ALL DATA FEATURES AND JOIN THEM (OR NOT FOR INDUCTIVE LEARNING)
        # TODO: TRANSFORM INTO INTEGERS ONE HOT ETC STRING VS NUMERICAL USING DATA FUNCTION
                
        # Creates graph nodes and labels
        # TODO: GET NODE FEATURES FROM LOADED DATAFRAME (BEFORE TRANSFORM)
        labels = []
        for node in node_df.itertuples():
            features = [node.__getattribute__(feat) for feat in NODE_FEATURES]
            G.add_nodes_from([(node.id, {'x': features})])
            labels.append(node.colonised)
        
        # Add ward edges to graph
        links_df = pd.read_csv(LINK_PATH)
        links_df = links_df[links_df['src'].isin(node_df.id.unique())]
        links_df['src'] = links_df['src'].astype(int)
        links_df['dest'] = links_df['dest'].astype(int)
        for edge in links_df.itertuples():
            G.add_edges_from([(edge.src, edge.dest)])
        
        # Create and return pytorch-geometric object from the graph
        pyg_graph = from_networkx(G)
        scaler = RobustScaler().fit(pyg_graph.x)
        # TODO: CHECK IF SCALE TWICE IS OK???
        pyg_graph.x = torch.tensor(scaler.transform(pyg_graph.x))
        pyg_graph.y = torch.as_tensor(labels)
        return pyg_graph


if __name__ == '__main__':
    main()
