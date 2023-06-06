import os
import numpy as np
from data.data_utils import load_data
from joblib import dump, load
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)


LOAD_MODELS = False
BALANCED_COND = 'non'  # 'non', 'under', 'over'
MODELS = {
    'catboost': CatBoostClassifier(verbose=False),
    'knn': KNeighborsClassifier(),
    'logistic_regression': LogisticRegression(verbose=False),
    'random_forest': RandomForestClassifier(verbose=False),
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
        'penalty': ['elasticnet', 'none'],
        'max_iter': [100, 200, 500, 1000],
        'l1_ratio': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
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
    """ Train and test all control models, after hyper-parameter grid search
    """
    new_report_path = find_new_path(REPORT_PATH)
    for model_name in MODELS.keys():
        model = MODELS[model_name]
        grid = GRIDS[model_name]
        print('\nNow running %s algorithm' % model_name)
        run_one_model(model_name, model, grid, new_report_path)
    print('\nAll models were run!\n')
    
    
def find_new_path(report_path):
    """ Generate a report path that does not exist yet
    """
    iteration = 0
    while os.path.exists('%s_%02i' % (report_path, iteration)):
        iteration += 1
        if iteration >= 100:
            error_msg = "Too many %s's! Delete some." % report_path
            raise FileExistsError(error_msg)
    return '%s_%02i' % (report_path, iteration)
    

def run_one_model(model_name, model, grid, new_report_path):
    """ Train and test one model, after hyper-parameter grid search
    """
    print('Loading data features and labels')
    X_train, X_dev, X_test, y_train, y_dev, y_test = load_data(BALANCED_COND)
    if not LOAD_MODELS:
        print('Finding best model hyper-parameters using random search')
        model = find_best_hyperparams(X_train, y_train, model_name, model, grid)
    else:
        print('Loading model from best model checkpoint path')
        model = load_model(model_name)
    print('Evaluating best model with the dev and test data')
    report = evaluate_model(model, X_dev, y_dev, X_test, y_test)
    print('Writing result report to %s' % new_report_path)
    write_report(report, model_name, model, new_report_path)
    

def find_best_hyperparams(X_train, y_train, model_name, model, grid,
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
    )
    model = random_search.fit(X_train, y_train)
    save_model(model, model_name)
    best_parameters = random_search.best_params_
    return model, best_parameters


def save_model(model, model_name):
    """ Create checkpoint and save trained model
    """
    ckpt_path = os.path.join(CKPT_DIR, 'best_%s.joblib' % model_name)
    os.makedirs(os.path.split(ckpt_path)[0], exist_ok=True)
    dump(model, ckpt_path)


def load_model(model_name):
    """ Load trained model from existing checkpoint
    """
    ckpt_path = os.path.join(CKPT_DIR, 'best_%s.joblib' % model_name)
    model = load(ckpt_path)
    return model
    

def write_report(report, model_name, model, new_report_path):
    """ Write classification report (micro/macro precision/recall/f1-score)
    """
    with open(new_report_path, 'a') as f:
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
