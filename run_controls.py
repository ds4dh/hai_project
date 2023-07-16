import os
import json
import numpy as np
import optuna
from functools import partial
from data.data_utils import load_features_and_labels
from data.graph_utils import load_graph_features_and_labels
from run_utils import generate_dict_report, find_optimal_threshold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import ConvergenceWarning
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Helper function
def ABS_JOIN(*args):
    return os.path.abspath(os.path.join(*args))

DO_HYPER_OPTIM = False
N_CPUS = 12
POSITIVE_ID = 1  # label id for 'infected' label
SETTING_CONDS = ['inductive', 'transductive']  # only used if graph features
LINK_CONDS = ['all', 'wards', 'caregivers', 'no']  # only used if graph features
BALANCED_CONDS = ['under', 'non', 'over']
MODELS = {
    'logistic_regression': LogisticRegression,
    'random_forest': RandomForestClassifier,
    'catboost': CatBoostClassifier,
    'knn': KNeighborsClassifier,
}
GRIDS = {
    'logistic_regression': {
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'C': [0.0, 0.01, 0.1, 1, 10, 100, 200, 1000],
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'max_iter': [100, 200, 500, 1000],
        'l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'class_weight': [None, 'balanced'],
    },
    'random_forest': {
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2, 5, 10],
        'max_depth': [5, 10, 15, 20, 30, 50, 100],
        'n_estimators': [1, 10, 50, 100, 1000],
        'class_weight': [None, 'balanced', 'balanced_subsample'],
    },
    'catboost': {
        'depth': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'learning_rate': [0.010, 0.025, 0.050, 0.075, 0.100],
        'iterations': [500, 1000, 1500, 2000, 2500, 3000],
    },
    'knn': {
        'n_neighbors': list(range(1, 30)),
        'weights': ['uniform', 'distance'],
        'p': [1, 2, 3, 4],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    },
}


def main():
    """ Run all models for any set of conditions, keep results in a ckpt path
    """
    # First use node features only
    for balanced_cond in BALANCED_CONDS:
        ckpt_dir = ABS_JOIN('models', 'controls', 'node_features',
                             '%s_balanced' % balanced_cond)
        os.makedirs(os.path.split(ckpt_dir)[0], exist_ok=True)
        run_all_models(ckpt_dir, 'nodes', balanced_cond)
            
    # Second, use node and edge features (generated with node2vec)
    for setting_cond in SETTING_CONDS:
        for balanced_cond in BALANCED_CONDS:
            for link_cond in LINK_CONDS:
                ckpt_dir = ABS_JOIN('models', 'controls', 'edge_features',
                                    '%s_setting' % setting_cond,
                                    '%s_balanced' % balanced_cond,
                                    'links_%s' % link_cond)
                os.makedirs(os.path.split(ckpt_dir)[0], exist_ok=True)
                run_all_models(ckpt_dir, 'edges',
                               balanced_cond, setting_cond, link_cond)
                
                
def run_all_models(ckpt_dir: str,
                   feat_cond: str,
                   balanced_cond: str,
                   setting_cond: str=None,
                   link_cond: str=None,
                   ) -> None:
    """ Train and test all control models, after hyper-parameter grid search
    """
    print('New run: %s features, %s setting, %s-balanced data, %s link(s)' %
          (feat_cond, setting_cond, balanced_cond, link_cond))
    for model_name in MODELS.keys():
        print('\nNow running %s algorithm' % model_name)
        run_one_model(model_name, ckpt_dir, feat_cond, balanced_cond,
                      setting_cond, link_cond)
    print('\nAll models were run!\n')
    

def run_one_model(model_name: str,
                  ckpt_dir: str,
                  feature_cond: str,
                  balanced_cond: str,
                  setting_cond: str=None,
                  link_cond: str=None
                  ) -> None:
    """ Train and test one model, after hyper-parameter grid search
    """
    # Load the correct set of data samples
    print('Loading data features and labels')
    if feature_cond == 'nodes':
        X, y, _ = load_features_and_labels(balanced_cond)
    elif feature_cond == 'edges':
        X, y = load_graph_features_and_labels(
            setting_cond, balanced_cond, link_cond)
    
    # Hyper-optimization loop to find best set of model hyper-parameters
    if DO_HYPER_OPTIM == True:        
        print('Finding best model hyper-parameters using random search')
        objective = partial(tune_model, X=X, y=y, model_name=model_name)
        study = optuna.create_study(
            study_name='run_control_pl',
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=50),
            sampler=optuna.samplers.TPESampler(),
        )
        optuna.pruners.SuccessiveHalvingPruner()
        study.optimize(objective, n_trials=500, n_jobs=N_CPUS)
        best_params = study.best_trial.params
        
    # Load from the result of a previous hyper-optimization run
    else:
        try:
            params_path = ABS_JOIN(ckpt_dir, '%s_best_params.json' % model_name)
            with open(params_path, 'r') as f:
                best_params = json.load(f)
        except:
            return
    
    # Re-train and evaluate the best model
    model = initialize_model(model_name, best_params)
    model.fit(X['train'], y['train'])  # add dev data here?
    y_prob_dev = model.predict_proba(X['dev'])[:, POSITIVE_ID]
    y_prob_test = model.predict_proba(X['test'])[:, POSITIVE_ID]
    optimal_threhsold = find_optimal_threshold(y['dev'], y_prob_dev)
    report = generate_dict_report(y['test'], y_prob_test, optimal_threhsold)
    write_report(report, model_name, best_params, ckpt_dir)
    

def tune_model(trial: optuna.trial.Trial,
               X: np.ndarray,
               y: np.ndarray,
               model_name: str,
               ) -> float:
    """ Find best catboost model with grid-search and k-fold cross-validation,
        then train the best model with the whole data and save it
    """
    # Define model with suggested parameters
    model_params = {k: trial.suggest_categorical(k, v)
                    for k, v in GRIDS[model_name].items()}
    model = initialize_model(model_name, model_params)
    
    # Train model and return auroc computed with the validation set
    try:
        model.fit(X['train'], y['train'])
        y_prob_dev = model.predict_proba(X['dev'])[:, POSITIVE_ID]
        roc_auc = roc_auc_score(y['dev'], y_prob_dev)
    except:
        roc_auc = 0.0  # error score (since auroc should be big)
    return roc_auc
    

def write_report(report: dict,
                 model_name: str,
                 best_params: dict,
                 ckpt_dir: str,
                 ) -> None:
    """ Write classification report (micro/macro precision/recall/f1-score)
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    print('Writing result report to %s' % ckpt_dir)
    report_path = ABS_JOIN(ckpt_dir, '%s_report.json' % model_name)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)  # f.write(report)
    best_params_path = ABS_JOIN(ckpt_dir, '%s_best_params.json' % model_name)
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=4)
        
        
def initialize_model(model_name: str,
                     params: dict):
    """ Load model in silence mode if possible, and load it anyways if not
    """
    try:
        return MODELS[model_name](**params, verbose=False)
    except:
        return MODELS[model_name](**params)
    

if __name__ == '__main__':
    main()
