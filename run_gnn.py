import os
import json
import shutil
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl  # lightning.pytorch as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data
from data.graph_utils import IPCDataset
from optuna.integration import PyTorchLightningPruningCallback as PLPruningCallback
from functools import partial
from sklearn.metrics import roc_auc_score
from run_utils import FocalLoss, generate_report, find_optimal_threshold
from warnings import filterwarnings
filterwarnings('ignore', category=UserWarning, module='pytorch_lightning')
filterwarnings('ignore', category=RuntimeWarning, module='pytorch_lightning')


ABS_JOIN = lambda *args: os.path.abspath(os.path.join(*args))  # helper function
DO_HYPER_OPTIM = False
N_TRAIN_EPOCHS = 500
N_TRIALS = 100
SETTING_CONDS = ['inductive', 'transductive']
BALANCED_CONDS = ['under', 'non', 'over']
LINK_CONDS = ['all', 'wards', 'caregivers', 'no']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_GPUS = torch.cuda.device_count()
N_CPUS = 1  # os.cpu_count() // 2 - 1  # -> 1, else GPU goes OOM
N_DEVICES = N_GPUS if DEVICE == 'cuda' else N_CPUS


def main():
    """ Train a GNN in different settings, data balance and link conditions
    """
    for setting_cond in SETTING_CONDS:
        for balanced_cond in BALANCED_CONDS:
            for link_cond in LINK_CONDS:
                # Initialize dataset and result directory, given conditions
                print('New run: %s setting, %s-balanced data, %s link(s)' %
                    (setting_cond, balanced_cond, link_cond))
                dataset = IPCDataset(setting_cond, balanced_cond, link_cond)
                log_dir = ABS_JOIN(
                    'models', 'gnn',
                    '%s_setting' % setting_cond,
                    '%s_balanced' % balanced_cond,
                    '%s_links' % link_cond
                )
                
                # Hyper-optimization loop to find best model
                if DO_HYPER_OPTIM == True:
                    if os.path.exists(log_dir): shutil.rmtree(log_dir)  # needed?
                    os.makedirs(log_dir, exist_ok=True)
                    objective = partial(
                        tune_net, dataset=dataset, balanced_cond=balanced_cond,
                        setting_cond=setting_cond, log_dir=log_dir,
                    )
                    study = optuna.create_study(
                        study_name='run_gnn_pl',
                        direction='maximize',
                        pruner=optuna.pruners.MedianPruner(n_warmup_steps=50),
                        sampler=optuna.samplers.TPESampler(),
                    )
                    optuna.pruners.SuccessiveHalvingPruner()
                    study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_CPUS)
                    best_params = study.best_trial.params
                
                # Load from the result of a previous hyper-optimization run
                else:
                    try:
                        param_path = ABS_JOIN(log_dir, 'best_params.json')
                        with open(param_path, 'r') as f:
                            best_params = json.load(f)
                    except:
                        continue
                
                # Evaluate best model and report best metric
                test_eval = evaluate_net(
                    dataset, best_params, setting_cond, log_dir)
                with open(ABS_JOIN(log_dir, 'gnn_report.json'), 'w') as f:
                    json.dump(test_eval['report'], f, indent=4)
                with open(ABS_JOIN(log_dir, 'gnn_best_params.json'), 'w') as f:
                    json.dump(best_params, f, indent=4)


def evaluate_net(dataset: Data,
                 params: dict,
                 setting_cond: str,
                 log_dir: str
                 ) -> dict:
    """ Train a model and evaluate it with test dataset
    """
    pl_model = PLWrapperNet(params, dataset, setting_cond)
    trainer = train_model(pl_model, log_dir)  # retrain model (very short)
    pl_model.optimize_threshold = True  # set threshold optimization
    trainer.validate(pl_model, ckpt_path='best')  # find best threshold
    trainer.test(pl_model, ckpt_path='best')  # evaluate model
    return pl_model.test_evaluation


def tune_net(trial: optuna.trial.Trial,
             dataset: Data,
             balanced_cond: str,
             setting_cond: str,
             log_dir: str,
             ) -> float:
    """ Tune hyper-parameters of a GNN for HAI prediction task
    """
    # Initialize pytorch-lightning instance (model, data, optimizer, scheduler)
    if balanced_cond == 'over':  # avoid OOM
        torch.set_float32_matmul_precision('medium')
        hidden_dim_choice = [32, 64, 128]
        layer_choice = ['gcn', 'sage']
    else:  # try more parameters
        torch.set_float32_matmul_precision('high')
        hidden_dim_choice = [32, 64, 128, 256]
        layer_choice = ['gcn', 'sage', 'gat']
    config = {
        'hidden_dim': trial.suggest_categorical('hidden_dim', hidden_dim_choice),
        'n_layers': trial.suggest_categorical('n_layers', [2, 3, 4, 5]),
        'layer': trial.suggest_categorical('layer', layer_choice),
        'dropout': trial.suggest_categorical('dropout', [0.0, 0.1, 0.3, 0.5]),
        'n_heads': trial.suggest_categorical('n_heads', [4, 8, 16]),
        'lr': trial.suggest_float('lr', 1e-3, 1e-0, log=True),
        'w_balance': trial.suggest_float('w_balance', 1e1, 1e3, log=True),
    }
    pl_model = PLWrapperNet(config, dataset, setting_cond)
    trainer = train_model(pl_model, log_dir, trial)
    
    # Report objective value (auroc) using best checkpoint during training
    objective_value = trainer.validate(ckpt_path='best')[0]['dev_auroc']
    return objective_value


def train_model(pl_model: pl.LightningModule,
                logdir: str,
                trial: optuna.trial.Trial=None,
                )-> pl.Trainer:
    """ Define logger, callbacks, and trainer, then train the model
    """
    # Define logger and callbacks (trial used only if used during optuna study)
    logger = TensorBoardLogger(logdir, name='logs')
    callbacks = [EarlyStopping(monitor='dev_loss', mode='min', patience=5)]
    if trial is not None:
        callbacks.append(PLPruningCallback(trial, monitor='dev_auroc'))
    else:
        print('\n/!\ Retraining best model identified by optuna study /!\ \n')
    
    # Train model and return whole trainer object
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=N_TRAIN_EPOCHS,
        accelerator=DEVICE, 
        devices=N_DEVICES,
        log_every_n_steps=1,
        check_val_every_n_epoch=10,
        num_sanity_val_steps=0,
        callbacks=callbacks,
    )
    trainer.fit(pl_model)
    return trainer

    
class PLWrapperNet(pl.LightningModule):
    def __init__(self,
                 config: dict,
                 dataset: Data,
                 setting_cond: str,
                 ) -> None:
        """ Pytorch-lightning object wrapping around model config and training
        """
        super().__init__()
        self.lr = config['lr']
        self.dataset = dataset
        self.setting_cond = setting_cond
        self.optimize_threshold = False
        self.decision_threshold = 0.5  # initialization
        self.net = Net(dataset.num_features, **config)
        self.criterions = self.init_criterions(config['w_balance'])
        
    def forward(self, batch):
        """ Process nodes and edges to node infection probability
        """
        y_logits = self.net(batch.x, batch.edge_index)
        if self.setting_cond == 'transductive':
            y_logits = y_logits[batch.mask]
        y_scores = torch.sigmoid(y_logits).view(-1)
        return y_scores
    
    def compute_loss(self, batch, split):
        """ Compute loss for train or dev step
        """
        y_score = self.forward(batch)
        y_true = batch.y
        if self.setting_cond == 'transductive':
            y_true = y_true[batch.mask]
        return self.criterions[split](y_score, y_true)
    
    def training_step(self, batch, batch_idx):
        """ Training step using either inductive or transductive setting
        """
        return self.compute_loss(batch, split='train')
        
    def validation_step(self, batch, batch_idx):
        """ Validation step using either inductive or transductive setting
        """
        loss = self.compute_loss(batch, split='dev')
        evaluation = self.evaluate_net(batch)
        self.log('dev_loss', loss, batch_size=1)
        self.log('dev_auroc', evaluation['report']['auroc'], batch_size=1)
    
    def test_step(self, batch, batch_idx):
        """ Evaluate network with testing data
        """
        self.test_evaluation = self.evaluate_net(batch, test_mode=True)
        
    def evaluate_net(self, batch, test_mode=False):
        """ Final evaluation of fine-tuned and trained model
        """
        # Generate predictions and load true labels
        y_score = self.forward(batch).cpu().numpy()
        y_true = batch.y
        if self.setting_cond == 'transductive':
            y_true = y_true[batch.mask]
        y_true = y_true.cpu().numpy()
        
        # Optimize (or not) decision threshold before using testing data
        if self.optimize_threshold and not test_mode:  # using validation data
            self.decision_threshold = find_optimal_threshold(y_true, y_score)
        
        # Evaluate model with 0.5 or f1-score optimized decision threshold)
        report = generate_report(y_true, y_score, 0.5)
        report_optim = generate_report(y_true, y_score, self.decision_threshold)
        report.update({'%s_optim' % k: v for k, v in report_optim.items()})
        
        # Return metrics, true labels, and model predictions
        return {'report': report, 'y_true': y_true, 'y_score': y_score}
        
    def configure_optimizers(self) -> tuple[list[dict]]:
        optim = AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = OneCycleLR(optim, self.lr, total_steps=N_TRAIN_EPOCHS)
        return [optim], [{'scheduler': sched, 'interval': 'epoch'}]
    
    def get_dataloader(self, split: str) -> DataLoader:
        """ Generic function to initialize and return an iterable data split
        """
        if self.setting_cond == 'transductive':
            data = self.dataset.get_split('whole')
            data.mask = data.masks[split]
        elif self.setting_cond == 'inductive':
            data = self.dataset.get_split(split)
        return DataLoader(dataset=[data], batch_size=None)
        
    def train_dataloader(self) -> DataLoader:
        """ Return the training dataloader
        """
        return self.get_dataloader('train')
    
    def val_dataloader(self) -> DataLoader:
        """ Return the validation dataloader
        """
        return self.get_dataloader('dev')
    
    def test_dataloader(self) -> DataLoader:
        """ Return the testing dataloader
        """
        return self.get_dataloader('test')
    
    def init_criterions(self,
                        w_balance: float,
                        loss_type: str='focal'
                        ) -> dict:
        """ Initialize class-weighted train and dev criterion (better way?)
        """
        # Get train and dev labels
        if self.setting_cond == 'inductive':
            y_train = self.dataset.get_split('train').y
            y_dev = self.dataset.get_split('dev').y
        elif self.setting_cond == 'transductive':
            whole_data = self.dataset.get_split('whole')
            y_train = whole_data.y[whole_data.masks['train']]
            y_dev = whole_data.y[whole_data.masks['dev']]
        
        # Create train and dev balanced weights (now: w_balance = parameter)
        # w_balance = ((y_train == 0).sum() / (y_train == 1).sum()).item()
        w_train = torch.tensor([1 if g == 0 else w_balance for g in y_train])
        w_dev = torch.tensor([1 if g == 0 else w_balance for g in y_dev])
        
        # Initialize criterion with the balanced weights
        loss_cls = FocalLoss if loss_type == 'focal' else nn.BCEWithLogitsLoss
        crit_train = loss_cls(weight=w_train.to(DEVICE))
        crit_dev = loss_cls(weight=w_dev.to(DEVICE))
        return {'train': crit_train, 'dev': crit_dev}
        
        
class Net(nn.Module):
    def __init__(self,
                 in_features: int,  # not a hyper-parameter
                 hidden_dim: int,
                 layer: str,
                 n_layers: int=2,
                 n_heads: int=8,  # only for gat layers
                 dropout: float=0.1,
                 *args, **kwags,
                 ) -> None:
        """ Graph neural network that takes patient network into account
        """
        super(Net, self).__init__()
        assert n_layers >= 2
        self.dropout = dropout
        self.layers = nn.ModuleList()
        
        # Graph convolutional network layers
        if layer == 'gcn':
            self.layers.append(GCNConv(in_features, hidden_dim))
            for _ in range(n_layers - 2):
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.layers.append(GCNConv(hidden_dim, 1))
        
        # Graph-sage framework layers
        elif layer == 'sage':
            self.layers.append(SAGEConv(in_features, hidden_dim))
            for _ in range(n_layers - 2):
                self.layers.append(SAGEConv(hidden_dim, hidden_dim))
            self.layers.append(SAGEConv(hidden_dim, 1))
        
        # Graph attention network layers
        elif layer == 'gat':
            assert hidden_dim % n_heads == 0
            n_out = hidden_dim // n_heads
            self.layers.append(GATConv(in_features, n_out, heads=n_heads))
            for _ in range(n_layers - 2):
                self.layers.append(GATConv(hidden_dim, n_out, heads=n_heads))
            self.layers.append(GATConv(hidden_dim, 1))
            
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor
                ) -> torch.Tensor:
        """ Forward pass of the graph neural network
        """
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)  # last layer: no activation / dropout
        return x
    
    
if __name__ == '__main__':
    main()
