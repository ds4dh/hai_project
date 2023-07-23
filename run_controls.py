import os
import json
import numpy as np
import optuna
from functools import partial
from data.data_utils import load_features_and_labels
from data.graph_utils import load_graph_features_and_labels
from run_utils import generate_report, find_optimal_threshold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import ConvergenceWarning
from catboost import CatBoostClassifier
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler, BruteForceSampler
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)


ABS_JOIN = lambda *args: os.path.abspath(os.path.join(*args))  # helper function
DO_HYPER_OPTIM = True
N_CPUS = 1
POSITIVE_ID = 1  # label id for 'infected' label
N_TRIALS = 100
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
        'depth': [5, 6, 7, 8, 9, 10],
        'learning_rate': [0.001, 0.0025, 0.005, 0.010, 0.025, 0.050],
        'iterations': [100, 250, 500, 1000, 2500],
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
    # # First use node features only
    # for balanced_cond in BALANCED_CONDS:
    #     conds = {'feat_cond': 'nodes', 'balanced_cond': balanced_cond,
    #              'setting_cond': None, 'link_cond': None}
    #     for model_name in MODELS.keys():
    #         run_one_model(conds, model_name)
                            
    # Second, use node and edge features (generated with node2vec)
    for setting_cond in SETTING_CONDS:
        for balanced_cond in BALANCED_CONDS:
            for link_cond in LINK_CONDS:
                conds = {'feat_cond': 'edges', 'balanced_cond': balanced_cond,
                         'setting_cond': setting_cond, 'link_cond': link_cond}
                X, y = load_correct_data(conds)
                for model_name in MODELS.keys():
                    run_one_model(conds, model_name, X, y)
                    
                    
def run_one_model(conds: dict[str, str],
                  name: str,
                  X: np.ndarray,
                  y: np.ndarray,
                  ) -> None:
    """ Train and test one model, after hyper-parameter grid search
    """
    # Load data and load or find best parameters for the models. Note that the
    # 'edges' conditions will simply use the best parameters of the corresponding
    # 'features' condition, since it would take too long to tune hyper-parameters
    # for all settings, data balance, and link condition (mainly because of the
    # knn and catboost algorithms). After a pilot study, we determined that this
    # method produces similar performance levels for the control models.
    print(' - Running %s with conditions %s' % (name, conds))
    if DO_HYPER_OPTIM and conds['feat_cond'] != 'edges':
        best_params = find_best_params(X, y, conds, name)
    else:
        best_params = load_best_params(conds, name)
        if best_params == None: return  # model that have not been run yet
    
    # Generate performance report from model predictions
    y_score_dev, y_score_test = evaluate_model(X, y, name, best_params)
    optimal_threhsold = find_optimal_threshold(y['dev'], y_score_dev)
    report = generate_report(y['test'], y_score_test, 0.5)
    report_optim = generate_report(y['test'], y_score_test, optimal_threhsold)
    report.update({'%s_optim' % k: v for k, v in report_optim.items()})
    
    # Write report and best parameters to ckpt file
    ckpt_dir = get_ckpt_dir(conds)
    os.makedirs(os.path.split(ckpt_dir)[0], exist_ok=True)    
    write_report(report, name, best_params, ckpt_dir)


def find_best_params(X: np.ndarray,
                     y: np.ndarray,
                     conds: dict[str, str],
                     name: str,
                     ) -> dict:
    """ Find best model hyper-parameters using random search
    """
    objective = partial(tune_model, X=X, y=y, conds=conds, name=name)
    study = optuna.create_study(
        study_name='run_control_pl',
        direction='maximize',
        pruner=MedianPruner(n_warmup_steps=50),
        sampler=TPESampler(),
    )
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_CPUS)
    return study.best_trial.params
    
    
def load_correct_data(conds: dict[str, str],
                      ) -> tuple[np.ndarray, np.ndarray]:
    """ Load correct set of data for a given set of conditions
    """
    if conds['feat_cond'] == 'nodes':
        X, y, _ = load_features_and_labels(conds['balanced_cond'])
    elif conds['feat_cond'] == 'edges':
        node2vec_dim = 128  # performance does not increase past this value
        X, y = load_graph_features_and_labels(conds, node2vec_dim)
    return X, y


def load_best_params(conds: dict[str, str],
                     name: str,
                     ) -> dict:
    """ Load best parameters for a model, and return None if the model was not
        found, e.g., if it has not been hyper-parameter-optimized yet
    """
    try:
        # Generate correct checkpoint (as explained in run_one_model())
        if conds['feat_cond'] == 'edges':
            node_conds = dict(conds)
            node_conds.update({'feat_cond': 'nodes'})
            ckpt_dir = get_ckpt_dir(node_conds)
        else:
            ckpt_dir = get_ckpt_dir(conds)
        
        # Load corresponding parameters
        params_path = ABS_JOIN(ckpt_dir, '%s_best_params.json' % name)
        with open(params_path, 'r') as f:
            return json.load(f)
    except:
        return None


def get_ckpt_dir(conds: dict[str, str]) -> str:
    """ Get correct model checkpoint directory for a set of conditions
    """
    if conds['feat_cond'] == 'edges':
        return ABS_JOIN(
            'models', 'controls', 'edge_features',
            '%s_setting' % conds['setting_cond'],
            '%s_balanced' % conds['balanced_cond'],
            '%s_links' % conds['link_cond'],
        )
    elif conds['feat_cond'] == 'nodes':
        return ABS_JOIN(
            'models', 'controls', 'node_features',
            '%s_balanced' % conds['balanced_cond'],
        )
    else:
        raise ValueError('Invalid feature condition in condition dictionary.')


def evaluate_model(X: np.ndarray,
                   y: np.ndarray,
                   name: str,
                   best_params: dict
                   ) -> tuple[np.ndarray, np.ndarray]:
    """ Re-train a model with the best model parameters and generate predictions
    """
    model = initialize_model(name, best_params)
    model.fit(X['train'], y['train'])
    y_score_dev = model.predict_proba(X['dev'])[:, POSITIVE_ID]
    y_score_test = model.predict_proba(X['test'])[:, POSITIVE_ID]
    return y_score_dev, y_score_test


def tune_model(trial: optuna.trial.Trial,
               X: np.ndarray,
               y: np.ndarray,
               conds: dict[str, str],
               name: str,
               ) -> float:
    """ Find best catboost model with grid-search and k-fold cross-validation,
        then train the best model with the whole data and save it
    """
    # Suggest model parameters and load corresponding model
    params = {k: trial.suggest_categorical(k, v) for k, v in GRIDS[name].items()}
    model = initialize_model(name, params)
    
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
    print(' - Writing report for model %s to %s with params %s' %\
         (model_name, ckpt_dir, best_params))
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
