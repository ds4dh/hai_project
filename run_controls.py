import os
import numpy as np
from joblib import dump, load
from data.data_utils import load_features_and_labels
from data.graph_utils import load_graph_features_and_labels
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import ConvergenceWarning
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)


LOAD_MODELS = False
USE_GRAPH_FEATURES = True
SETTING_CONDS = ['inductive', 'transductive']  # only used if graph features
LINK_CONDS = ['all', 'wards', 'caregivers', 'no']  # only used if graph features
BALANCED_CONDS = ['non', 'under', 'over']
MODELS = {
    # 'catboost': CatBoostClassifier(verbose=False),
    # 'knn': KNeighborsClassifier(),
    'logistic_regression': LogisticRegression(verbose=False),
    # 'random_forest': RandomForestClassifier(verbose=False),
}
GRIDS = {
    'catboost': {
        'depth': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'learning_rate': [0.010, 0.025, 0.050, 0.075, 0.100],
        'iterations': [1000, 1500, 2000, 2500, 3000, 3500],
    },
    'knn': {
        'n_neighbors': list(range(1, 30)),
        'weights': ['uniform', 'distance'],
        'p': [1, 2, 3, 4],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    },
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
}
CKPT_DIR = os.path.join('models', 'controls', 'ckpts')
REPORT_PATH = os.path.join('models', 'controls', 'results')


def main():
    """ Run all models for any set of conditions, keep results in a ckpt path
    """
    # # First use node features only
    # for balanced_cond in BALANCED_CONDS:
    #     for balanced_cond in BALANCED_CONDS:
    #         ckpt_path = os.path.join('models', 'controls', 'node_features',
    #                                  '%s_balanced' % balanced_cond)
    #         run_all_models(ckpt_path, 'nodes', balanced_cond)
                
    # Second, use edge features only
    for setting_cond in SETTING_CONDS:
        for balanced_cond in BALANCED_CONDS:
            for link_cond in LINK_CONDS:
                ckpt_path = os.path.join('models', 'controls', 'edge_features',
                                         '%s_setting' % setting_cond,
                                         '%s_balanced' % balanced_cond,
                                         'links_%s' % link_cond)
                run_all_models(ckpt_path, 'edges',
                               balanced_cond, setting_cond, link_cond)
                
                
def run_all_models(ckpt_path: str,
                   feat_cond: bool,
                   balanced_cond: bool,
                   setting_cond: bool=False,
                   link_cond: bool=False,
                   ) -> None:
    """ Train and test all control models, after hyper-parameter grid search
    """
    print('New run: %s features, %s setting, %s-balanced data, %s link(s)' %
          (feat_cond, setting_cond, balanced_cond, link_cond))
    for model_name in MODELS.keys():
        model = MODELS[model_name]
        grid = GRIDS[model_name]
        print('\nNow running %s algorithm' % model_name)
        run_one_model(model_name, model, grid, ckpt_path, feat_cond,
                      balanced_cond, setting_cond, link_cond)
    print('\nAll models were run!\n')
    

def run_one_model(model_name, model, grid, ckpt_path, feature_cond,
                  balanced_cond, setting_cond=None, link_cond=None):
    """ Train and test one model, after hyper-parameter grid search
    """
    # Load the correct set of data samples
    print('Loading data features and labels')
    if feature_cond == 'nodes':
        X, y, _ = load_features_and_labels(balanced_cond)
    elif feature_cond == 'edges':
        X, y = load_graph_features_and_labels(setting_cond, balanced_cond, link_cond)
                    
    # Find the best set of hyperparameters or load best model
    if not LOAD_MODELS:
        print('Finding best model hyper-parameters using random search')
        model = find_best_hyperparams(X['train'], y['train'], model, grid)
        os.makedirs(os.path.split(ckpt_path)[0], exist_ok=True)
        save_model(model, ckpt_path)
    else:
        print('Loading model from best model checkpoint path')
        model = load_model(ckpt_path)
    
    # Evaluate the best model
    print('Evaluating best model with the dev and test data')
    report = evaluate_model(model, X['dev'], y['dev'], X['test'], y['test'])
    write_report(report, model_name, model, ckpt_path)
    

def find_best_hyperparams(X_train, y_train, model, grid,
                          cross_val_count=5, n_jobs=None, scoring='roc_auc'):
    """ Find best catboost model with grid-search and k-fold cross-validation,
        then train the best model with the whole data and save it
    """
    random_search = RandomizedSearchCV(
        model,
        param_distributions=grid,
        cv=cross_val_count,
        n_jobs=n_jobs,
        scoring=scoring,
        error_score=0.0,  # in case of invalid combination of hyper-parameters
    )
    model = random_search.fit(X_train, y_train)
    return model


def save_model(model, ckpt_path):
    """ Create checkpoint and save trained model
    """
    os.makedirs(os.path.split(ckpt_path)[0], exist_ok=True)
    dump(model, ckpt_path)


def load_model(ckpt_path):
    """ Load trained model from existing checkpoint
    """
    model = load(ckpt_path)
    return model
    

def write_report(report, model_name, model, ckpt_path):
    """ Write classification report (micro/macro precision/recall/f1-score)
    """
    report_path = ''.join((os.path.splitext(ckpt_path)[0], '.txt'))
    with open(report_path, 'a') as f:
        print('Writing result report to %s' % report_path)
        f.write('Results for %s\n' % model_name)
        f.write('Best hyperparameters: %s\n' % model.best_params_)
        f.write(report + '\n\n')
        
        
def evaluate_model(model, X_dev, y_dev, X_test, y_test, positive_id=1):
    """ Evaluate a trained model using the test data, after identifying the
        optimal threshold using the validation data
    """
    threshold = find_optimal_threshold(model, X_dev, y_dev)
    y_prob_test = model.predict_proba(X_test)[:, positive_id]
    y_pred_test = (y_prob_test >= threshold).astype(int)
    report = classification_report(y_test, y_pred_test)
    report += 'AUROC confidence interval: %s' % auroc_ci(y_test, y_prob_test)
    return report


def find_optimal_threshold(model, X_dev, y_dev, positive_id=1):
    """ Find optimal decision threshold using the validation set
    """
    y_prob_dev = model.predict_proba(X_dev)[:, positive_id]
    thresholds = np.linspace(0, 1, 100)
    scores = []
    for t in thresholds:
        y_pred_dev = (y_prob_dev >= t).astype(int)
        score = f1_score(y_dev, y_pred_dev)
        scores.append(score)
    return thresholds[np.argmax(scores)]


def auroc_ci(y_true, y_score, t_value=1.96):
    """ Compute confidence interval of auroc score using Racha's method
    """
    auroc = roc_auc_score(y_true, y_score)
    n1 = sum(y_true == 1)
    n2 = sum(y_true == 0)
    p1 = (n1 - 1) * (auroc / (2 - auroc) - auroc ** 2)
    p2 = (n2 - 1) * (2 * auroc ** 2 / (1 + auroc) - auroc ** 2)
    std_auroc = ((auroc * (1 - auroc) + p1 + p2) / (n1 * n2)) ** 0.5
    low, high = (auroc - t_value * std_auroc, auroc + t_value * std_auroc)
    return '%.3f (%.3f, %.3f)' % (auroc, low, high)


if __name__ == '__main__':
    main()
