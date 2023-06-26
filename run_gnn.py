import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import InMemoryDataset
from data.graph_utils import IPCDataset
from sklearn.metrics import classification_report


N_TRAIN_EPOCHS = 400
LR = 1e-2
N_HIDDEN = 32
LAYER_TYPE = 'sage'  # 'gcn', 'sage', 'gat'
SETTING_CONDS = ['inductive', 'transductive']
BALANCED_CONDS = ['non', 'under', 'over']
LINK_CONDS = ['all', 'wards', 'caregivers', 'no']
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
                print('New run: %s setting, %s-balanced data, %s link(s)' %
                      (setting_cond, balanced_cond, link_cond))
                train_model(setting_cond, balanced_cond, link_cond, ckpt_path)
                

def train_model(setting_cond: str,
                balanced_cond: str,
                link_cond: str,
                ckpt_path: str):
    """ Train a GNN model to predict HAI from patient nodes and worker edges
    """
    # Initialize data and model
    dataset = IPCDataset(setting_cond, balanced_cond, link_cond)
    model = Net(dataset)
    
    # Train model if required
    if not TEST_ONLY:
        train_loss_plot, dev_loss_plot = train_net(model, dataset, setting_cond)
        os.makedirs(os.path.split(ckpt_path)[0], exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
    else:
        train_loss_plot, dev_loss_plot = [], []
    
    # Reload checkpoint and test model
    model.load_state_dict(torch.load(ckpt_path))
    report = evaluate_net(model, dataset, setting_cond)

    # Plot training process (if any) and testing metrics
    save_model_results(train_loss_plot, dev_loss_plot, report, ckpt_path)


def train_net(model: nn.Module,
              dataset: InMemoryDataset,
              setting_cond: str
              ) -> tuple[list[float]]:
    """ Train a network to predict bacterial colonisation in hospitals
    """    
    # Load data (inductive or transductive setting)
    if setting_cond == 'inductive':
        train_data = dataset.get_split('train').to(DEVICE)
        dev_data = dataset.get_split('dev').to(DEVICE)
        train_golds = train_data.y
        dev_golds = dev_data.y
    elif setting_cond == 'transductive':
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
        if setting_cond == 'inductive':
            train_logits = model(train_data.x, train_data.edge_index)
            train_probs = torch.sigmoid(train_logits).view(-1)
        elif setting_cond == 'transductive':
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
            if setting_cond == 'inductive':
                dev_logits = model(dev_data.x, dev_data.edge_index)
                dev_probs = torch.sigmoid(dev_logits).view(-1)
            elif setting_cond == 'transductive':
                dev_probs = whole_probs[dev_mask].view(-1)
            dev_loss = dev_criterion(dev_probs, dev_golds)

        # Record and report training loss
        train_loss_plot.append(train_loss.item())
        dev_loss_plot.append(dev_loss.item())
        print('\r ---- Epoch %03d: loss = %.03f' % (epoch, train_loss), end='')
    
    # Return the output of training (and also: model has been updated)
    print('\n - Training finished')
    return train_loss_plot, dev_loss_plot


def evaluate_net(model: nn.Module,
                 dataset: InMemoryDataset,
                 setting_cond: str,
                 thresh: float=0.9):
    """ Generates predictions with a trained network and report various metrics
    """
    # TODO: DO LIKE IN RUN_CONTROLS WITH COMPUTATION OF BEST THRESHOLD ETC!
    # Initialize model, data and labels
    print(' - Testing model')
    model.eval()
    if setting_cond == 'inductive':
        test_data = dataset.get_split('dev').to(DEVICE)  # with dev for now
        test_golds = test_data.y
    elif setting_cond == 'transductive':
        whole_data = dataset.get_split('whole').to(DEVICE)
        test_mask = whole_data.masks['dev']  # with dev for now
        test_golds = whole_data.y[test_mask]

    # Compute model predictions
    with torch.no_grad():
        if setting_cond == 'inductive':
            test_logits = model(test_data.x, test_data.edge_index)
            test_probs = torch.sigmoid(test_logits).view(-1)
        elif setting_cond == 'transductive':
            whole_logits = model(whole_data.x, whole_data.edge_index)
            whole_preds = torch.sigmoid(whole_logits)
            test_probs = whole_preds[test_mask].view(-1)

    # Compute all metrics using model predictions
    if DEVICE.type != 'cpu':
        test_probs = test_probs.cpu().numpy()
        test_golds = test_golds.cpu().numpy()
    test_preds = [0 if p < thresh else 1 for p in test_probs]
    return classification_report(test_golds, test_preds)


def save_model_results(train_loss_plot: list[float],
                       dev_loss_plot: list[float],
                       report: str,
                       ckpt_path
                       ) -> None:
    """ Plot a summary of the training process (training loss vs epoch)
    """
    # Plot training report
    plot_path = ''.join((os.path.splitext(ckpt_path)[0], '.png'))
    _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(train_loss_plot)), train_loss_plot, label='Train loss')
    ax.plot(range(len(dev_loss_plot)), dev_loss_plot, label='Dev loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training loss')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    
    # Write classification report
    report_path = ''.join((os.path.splitext(ckpt_path)[0], '.txt'))
    with open(report_path, 'a') as f: f.write(report)


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

    
if __name__ == '__main__':
    main()
