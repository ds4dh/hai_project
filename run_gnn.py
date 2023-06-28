import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import InMemoryDataset, Data
from data.graph_utils import IPCDataset
from ray import tune
from ray.air import session
from functools import partial
from run_utils import generate_report
from sklearn.metrics import roc_auc_score


N_TRAIN_EPOCHS = 500
N_GPUS = torch.cuda.device_count()
N_CPUS = os.cpu_count() - 2
SETTING_CONDS = ['inductive', 'transductive']
BALANCED_CONDS = ['non', 'under', 'over']
LINK_CONDS = ['all', 'wards', 'caregivers', 'no']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEARCH_SPACE = {
    'hidden_dim': tune.choice([16, 32, 64, 128]),
    'n_layers': tune.choice([2, 3, 4, 5]),
    'dropout': tune.choice([0.0, 0.1, 0.3, 0.5]),
    'layer_type': tune.choice(['gcn', 'sage', 'gat']),
    'n_heads': tune.sample_from(lambda spec:
        None if spec.config.layer_type != 'gat'
        else np.random.choice([4, 8, 16])),
    'lr': tune.loguniform(1e-4, 1e-1),
}


def main():
    """ Train a GNN in different settings, data balance and link conditions
    """
    for setting_cond in SETTING_CONDS:
        for balanced_cond in BALANCED_CONDS:
            for link_cond in LINK_CONDS:
                # Define dataset and ckpt path, given conditions
                print('New run: %s setting, %s-balanced data, %s link(s)' %
                      (setting_cond, balanced_cond, link_cond))
                dataset = IPCDataset(setting_cond, balanced_cond, link_cond)
                report_path = os.path.abspath(os.path.join(
                    'models', 'gnn',
                    '%s_setting' % setting_cond,
                    '%s_balanced' % balanced_cond,
                    'links_%s.txt' % link_cond
                ))
                
                # Define dataset and start hyper-parameter tuning
                analysis = tune.run(
                    partial(tune_net, dataset=dataset, setting_cond=setting_cond),
                    resources_per_trial={'cpu': N_CPUS, 'gpu': N_GPUS},
                    config=SEARCH_SPACE,
                    num_samples=1,  # (?)
                )
                
                # Test model with best hyperparameters and write report locally
                best_config = analysis.get_best_config(metric='auroc', mode='max')
                model = Net(in_features=dataset.num_features, **best_config)
                model = model.to(DEVICE)
                metric = evaluate_net(model, dataset, setting_cond, split='test')
                final_report = metric['report']
                with open(report_path, 'w') as f:
                    f.write(final_report)
                    
                    
def tune_net(config: dict, dataset: Data, setting_cond: str) -> None:
    """ Tune hyper-parameters of a GNN for HAI prediction task
    """
    # Initialize model, optimizer, and scheduler, given hyper-parameters
    model = Net(in_features=dataset.num_features, **config).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, config['lr'], total_steps=N_TRAIN_EPOCHS)
    
    # Train model and generate checkpoint (and track losses)
    train_losses, dev_losses = train_net(
        model=model,
        dataset=dataset,
        setting_cond=setting_cond,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    
    # Evaluate model with dev set and report results for hyper-parameter tuning
    tune_loss = dev_losses[-1]
    auroc = evaluate_net(model, dataset, setting_cond, split='dev')['auroc']
    session.report({'loss': tune_loss, 'auroc': auroc})
    
    
def train_net(model: nn.Module,
              dataset: InMemoryDataset,
              setting_cond: str,
              optimizer: torch.optim.Optimizer,
              scheduler: torch.optim.lr_scheduler,
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

    # Set class-balanced loss functions
    wb = ((train_golds == 0).sum() / (train_golds == 1).sum()).item()
    train_weights = torch.tensor([1 if g == 0 else wb for g in train_golds])
    dev_weights = torch.tensor([1 if g == 0 else wb for g in dev_golds])  # ??
    train_criterion = nn.BCEWithLogitsLoss(weight=train_weights.to(DEVICE))
    dev_criterion = nn.BCEWithLogitsLoss(weight=dev_weights.to(DEVICE))  # ??
    
    # Start training model
    train_losses, dev_losses = [], []
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
        train_losses.append(train_loss.item())
        dev_losses.append(dev_loss.item())
    
    # Return losses
    return train_losses, dev_losses


def evaluate_net(model: nn.Module,
                 dataset: InMemoryDataset,
                 setting_cond: str,
                 split: str,
                 ) -> str:
    """ Generates predictions with a trained network and report various metrics
    """
    # Initialize model, data and labels
    model.eval()
    if setting_cond == 'inductive':
        data_dev = dataset.get_split('dev').to(DEVICE)
        data_test = dataset.get_split(split).to(DEVICE)  # test or dev
        y_dev = data_dev.y
        y_test = data_test.y
    elif setting_cond == 'transductive':
        data_whole = dataset.get_split('whole').to(DEVICE)
        dev_mask = data_whole.masks['dev']
        test_mask = data_whole.masks[split]  # test or dev
        y_dev = data_whole.y[dev_mask]
        y_test = data_whole.y[test_mask]
        
    # Compute model predictions
    with torch.no_grad():
        if setting_cond == 'inductive':
            logits_dev = model(data_dev.x, data_dev.edge_index)
            logits_test = model(data_test.x, data_test.edge_index)
            y_prob_dev = torch.sigmoid(logits_dev).view(-1)
            y_prob_test = torch.sigmoid(logits_test).view(-1)
        elif setting_cond == 'transductive':
            logits_whole = model(data_whole.x, data_whole.edge_index)
            y_prob_whole = torch.sigmoid(logits_whole)
            y_prob_dev = y_prob_whole[dev_mask].view(-1)
            y_prob_test = y_prob_whole[test_mask].view(-1)

    #  Put model results back on cpu if needed
    if DEVICE.type != 'cpu':
        y_prob_dev = y_prob_dev.cpu().numpy()
        y_dev = y_dev.cpu().numpy()
        y_prob_test = y_prob_test.cpu().numpy()
        y_test = y_test.cpu().numpy()
    
    # Compute metrics using model predictions
    report = generate_report(y_prob_dev, y_prob_test, y_dev, y_test)
    auroc = roc_auc_score(y_test, y_prob_test)
    return {'report': report, 'auroc': auroc}


class Net(nn.Module):
    def __init__(self,
                 in_features: int,  # not a hyper-parameter
                 hidden_dim: int,
                 layer_type: str,
                 dropout: float=0.1,
                 n_layers: int=2,
                 n_heads: int=8,  # only for gat layers
                 *args, **kwags,
                 ) -> None:
        """ Graph neural network that takes patient network into account
        """
        super(Net, self).__init__()
        assert n_layers >= 2
        self.dropout = dropout
        self.layers = nn.ModuleList()
        
        # Graph convolutional network layers
        if layer_type == 'gcn':
            self.layers.append(GCNConv(in_features, hidden_dim))
            for _ in range(n_layers - 2):
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.layers.append(GCNConv(hidden_dim, 1))
        
        # Graph-sage framework layers
        elif layer_type == 'sage':
            self.layers.append(SAGEConv(in_features, hidden_dim))
            for _ in range(n_layers - 2):
                self.layers.append(SAGEConv(hidden_dim, hidden_dim))
            self.layers.append(SAGEConv(hidden_dim, 1))
        
        # Graph attention network layers
        elif layer_type == 'gat':
            assert hidden_dim % n_heads == 0
            n_out = hidden_dim // n_heads
            self.layers.append(GATConv(in_features, n_out, heads=n_heads))
            for _ in range(n_layers - 2):
                self.layers.append(GATConv(hidden_dim, n_out, heads=n_heads))
            self.layers.append(GATConv(hidden_dim, 1))
            
    def forward(self, x, edge_index):
        """ Forward pass of the graph neural network
        """
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)  # last layer: no activation / dropout
        return x
    
    
if __name__ == '__main__':
    main()
