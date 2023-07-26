import os
import json
import numpy as np
from run_utils import generate_report, find_optimal_threshold
from run_gnn import (
    IPCDataset,
    load_best_params as load_best_params_gnn,
    evaluate_model as evaluate_model_gnn,
    get_ckpt_dir as get_ckpt_dir_gnn,
)
from run_controls import (
    load_best_params as load_best_params_ctrl,
    evaluate_model as evaluate_model_ctrl,
    load_correct_data as load_correct_data_ctrl,
    get_ckpt_dir as get_ckpt_dir_ctrl,
)


POOL_STRATEGIES = ['unanimity', 'majority', 'average']
SETTING_CONDS = ['inductive', 'transductive']
BALANCED_CONDS = ['under', 'non', 'over']
LINK_CONDS = ['all', 'wards', 'caregivers']  # , 'no']
CONTROL_MODELS = ['random_forest', 'knn', 'catboost', 'logistic_regression']
POSSIBLE_CONTROL_MODELS_FOR_ENSEMBLE = CONTROL_MODELS
POSSIBLE_GNN_MODELS_FOR_ENSEMBLE = ['gnn-inductive', 'gnn-transductive']
SELECTED_MODELS = ['gnn-inductive', 'random_forest', 'catboost']


def main():
    # run_gnn_ensemble()  # pooled over link conditions
    # run_control_ensemble()  # pooled over control models
    run_selected_ensemble()  # pooled over selection of models
    
            
def run_gnn_ensemble():
    """ Train a GNN in different settings, data balance and link conditions and
        generate predictions with all models using different voting strategies
    """
    for setting_cond in SETTING_CONDS:
        for balanced_cond in BALANCED_CONDS:
            y_true_list, y_score_list, thresh_optim_list = [], [], []
            for link_cond in LINK_CONDS:
                
                # Initialize dataset and model parameters, given conditions
                print('New run: %s setting, %s-balanced data, %s link(s)' %
                     (setting_cond, balanced_cond, link_cond))
                conds = {'feat_cond': 'edges', 'balanced_cond': balanced_cond,
                         'setting_cond': setting_cond, 'link_cond': link_cond}
                dataset = IPCDataset(**conds)
                log_dir = get_ckpt_dir_gnn(conds)
                params = load_best_params_gnn(log_dir)
                
                # Generate predictions and collect f1-optimal threshold
                test_eval = evaluate_model_gnn(
                    dataset, params, setting_cond, log_dir)
                thresh_optim = find_optimal_threshold(
                    test_eval['y_true'], test_eval['y_score'])
                y_true_list.append(test_eval['y_true'])
                y_score_list.append(test_eval['y_score'])
                thresh_optim_list.append(thresh_optim)
                
                # Use individual predictions to evaluate ensemble predictions
                out_dir = log_dir.replace('%s_links' % link_cond, 'ensemble_#S#')
                out_path = os.path.join(out_dir, 'gnn_report.json')
                evaluate_ensemble_predictions(
                    y_true_list, y_score_list, thresh_optim_list, out_path)
                
                
def run_control_ensemble():
    """ Train control models in different data balance conditions and generate
        predictions with all models using different voting strategies
    """
    for balanced_cond in BALANCED_CONDS:
        y_true_list, y_score_list, thresh_optim_list = [], [], []
        for model in CONTROL_MODELS:
            # Initialize dataset and model parameters, given conditions
            print('New run: %s-balanced data, %s model' % (balanced_cond, model))
            conds = {'feat_cond': 'nodes', 'balanced_cond': balanced_cond}
            best_params = load_best_params_ctrl(conds, name=model)
            X, y = load_correct_data_ctrl(conds)
            
            # Generate y-scores with the model, using the testing dataset
            _, y_score = evaluate_model_ctrl(X, y, model, best_params)
            thresh_optim = find_optimal_threshold(y['test'], y_score)
            y_true_list.append(y['test'])
            y_score_list.append(y_score)
            thresh_optim_list.append(thresh_optim)
            
        # Use individual predictions to evaluate ensemble predictions
        out_dir = get_ckpt_dir_ctrl(conds)
        out_path = os.path.join(out_dir, 'ensemble_#S#_report.json')
        evaluate_ensemble_predictions(
            y_true_list, y_score_list, thresh_optim_list, out_path)
            
            
def run_selected_ensemble():
    """ Train control and gnn models in different data balance conditions and
        generate predictions with all models using different voting strategies
    """
    for balanced_cond in ['non']:  # others turned out to be worse in most cases
        y_true_list, y_score_list, thresh_optim_list = [], [], []
        for model in SELECTED_MODELS:
            
            # Initialize dataset and result directory, given conditions
            print('New run: %s-balanced data, %s model' % (balanced_cond, model))
            setting_cond = model.split('-')[-1]
            conds = {'feat_cond': 'nodes', 'balanced_cond': balanced_cond,
                     'link_cond': 'wards', 'setting_cond': setting_cond}
            if model in POSSIBLE_CONTROL_MODELS_FOR_ENSEMBLE:
                out_dir = get_ckpt_dir_ctrl(conds)
                params = load_best_params_ctrl(conds, name=model)
                X, y = load_correct_data_ctrl(conds)
                y_true = y['test']
            elif model in POSSIBLE_GNN_MODELS_FOR_ENSEMBLE:
                out_dir = get_ckpt_dir_gnn(conds)
                params = load_best_params_gnn(out_dir)
                dataset = IPCDataset(**conds)
            
            # Generate y-scores with the model, using the testing dataset
            if model in POSSIBLE_CONTROL_MODELS_FOR_ENSEMBLE:
                _, y_score = evaluate_model_ctrl(X, y, model, params)
            elif model in POSSIBLE_GNN_MODELS_FOR_ENSEMBLE:
                test_eval = evaluate_model_gnn(
                    dataset, params, conds['setting_cond'], out_dir)
                y_true = test_eval['y_true']
                y_score = test_eval['y_score']
            thresh_optim = find_optimal_threshold(y_true, y_score)
            y_true_list.append(y_true)
            y_score_list.append(y_score)
            thresh_optim_list.append(thresh_optim)
            
        # Use individual predictions to evaluate ensemble predictions
        out_path = os.path.join('models', 'all', 'ensemble_#S#_report.json')
        evaluate_ensemble_predictions(
            y_true_list, y_score_list, thresh_optim_list, out_path)
        
        
def evaluate_ensemble_predictions(y_true_list: list[list[int]],
                                  y_score_list: list[list[float]],
                                  thresh_optim_list: list[float],
                                  out_path: str,
                                  ) -> None:
        """ Generate ensemble predictions and evaluate them
        """
        # Check trues alignement and try different ensemble pooling strategies
        y_true = pool_trues(y_true_list)
        for strategy in POOL_STRATEGIES:
            
            # All individual thresholds = 0.5
            thresh_base_list = [0.5 for _ in thresh_optim_list]
            y_score, threshold = generate_ensemble_predictions(
                y_score_list, thresh_base_list, strategy)
            report = generate_report(y_true, y_score, threshold)
            report.update({'y_score': y_score.tolist()})
            
            # F1-score optimized individual thresholds
            y_score, threshold = generate_ensemble_predictions(
                y_score_list, thresh_optim_list, strategy)
            report_optim = generate_report(y_true, y_score, threshold)
            report_optim.update({'y_score': y_score.tolist()})
            report.update({'%s_optim' % k: v for k, v in report_optim.items()})
            
            # Write whole report
            out_path_ = out_path.replace('#S#', strategy)
            out_dir = os.path.split(out_path)[0]
            os.makedirs(out_dir, exist_ok=True)
            with open(out_path_, 'w') as f:
                json.dump(report, f, indent=4)


def pool_trues(y_true_list):
    """ Check all is good and return y_true
    """
    assert all(all(y_true_list[0] == rest) for rest in y_true_list[1:])
    return y_true_list[0]


def generate_ensemble_predictions(y_score_list: list[list[float]],
                                  thresh_optim_list: list[float],
                                  strategy: str,
                                  ) -> list[int]:
    """ Pool model predictions using specific pooling strategy
    """
    y_scores = []
    transposed_scores = list(map(list, zip(*y_score_list)))  # zipped scores
    for model_scores in transposed_scores:
        
        # Pooled prediction is 0 if scores of all models are < threshold
        if strategy == 'unanimity':
            zipped = zip(model_scores, thresh_optim_list)
            all_votes_neg = all(score < thresh for score, thresh in zipped)
            y_scores.append(int(not all_votes_neg))
            pooled_threshold = 0.5  # any number between 0 and 1 will do
        
        # Pooled prediction is 0 if scores of a majority of models are < threshold
        elif strategy == 'majority':
            zipped = zip(model_scores, thresh_optim_list)
            n_votes_neg = sum(score < thresh for score, thresh in zipped)
            y_scores.append(int(n_votes_neg < (len(thresh_optim_list) / 2)))
            pooled_threshold = 0.5  # any number between 0 and 1 will do
            
        # Pooled prediction uses average of scores across models
        elif strategy == 'average':
            average_score = sum(model_scores) / len(model_scores)
            y_scores.append(average_score)
            pooled_threshold = sum(thresh_optim_list) / len(thresh_optim_list)        
        
        else:
            raise ValueError('Invalid pooling strategy')

    return np.array(y_scores), pooled_threshold


if __name__ == '__main__':
    main()
    