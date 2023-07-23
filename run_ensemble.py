import os
import json
import numpy as np
from run_gnn import IPCDataset, evaluate_net
from run_utils import generate_report

POOL_STRATEGIES = ['unanimity', 'majority', 'average']
SETTING_CONDS = ['inductive', 'transductive']
BALANCED_CONDS = ['under', 'non', 'over']
LINK_CONDS = ['all', 'wards', 'caregivers']  # , 'no']


def main():
    # run_control_ensemble()
    # run_control_edge_ensemble()
    run_gnn_ensemble()


def run_control_ensemble():
    ...


def run_control_edge_ensemble():
    """ Train control models that use node features, as well as node2vec edge
        features in different settings, data balance and link conditions
    """
    for setting_cond in SETTING_CONDS:
        for balanced_cond in BALANCED_CONDS:
            y_true_list, y_score_list, thresh_optim_list = [], [], []
            for link_cond in LINK_CONDS:
                
                # Initialize dataset and result directory, given conditions
                print('New run: %s setting, %s-balanced data, %s link(s)' %
                    (setting_cond, balanced_cond, link_cond))
                dataset = IPCDataset(setting_cond, balanced_cond, link_cond)
                log_dir = os.path.join(
                    'models', 'edge_features',
                    '%s_setting' % setting_cond,
                    '%s_balanced' % balanced_cond,
                    '%s_links' % link_cond
                )
                
                # Retrieve model
                try:
                    param_path = os.path.join(log_dir, 'best_params.json')
                    with open(param_path, 'r') as f:
                        params = json.load(f)
                except:
                    pass
                
                # Generate predictions
                test_eval = evaluate_net(dataset, params, setting_cond, log_dir)
                y_true_list.append(test_eval['y_true'])
                y_score_list.append(test_eval['y_score'])
                thresh_optim_list.append(test_eval['report']['threshold_optim'])
                
                # Use individual predictions to evaluate ensemble predictions
                out_dir = log_dir.replace('%s_links' % link_cond, 'ensemble')
                evaluate_ensemble_predictions(
                    y_true_list, y_score_list, thresh_optim_list, out_dir)
    

def run_gnn_ensemble():
    """ Train a GNN in different settings, data balance and link conditions
    """
    for setting_cond in SETTING_CONDS:
        for balanced_cond in BALANCED_CONDS:
            y_true_list, y_score_list, thresh_optim_list = [], [], []
            for link_cond in LINK_CONDS:
                
                # Initialize dataset and result directory, given conditions
                print('New run: %s setting, %s-balanced data, %s link(s)' %
                    (setting_cond, balanced_cond, link_cond))
                dataset = IPCDataset(setting_cond, balanced_cond, link_cond)
                log_dir = os.path.join(
                    'models', 'gnn',
                    '%s_setting' % setting_cond,
                    '%s_balanced' % balanced_cond,
                    '%s_links' % link_cond
                )
                
                # Retrieve model
                try:
                    param_path = os.path.join(log_dir, 'best_params.json')
                    with open(param_path, 'r') as f:
                        params = json.load(f)
                except:
                    pass
                
                # Generate predictions
                test_eval = evaluate_net(dataset, params, setting_cond, log_dir)
                y_true_list.append(test_eval['y_true'])
                y_score_list.append(test_eval['y_score'])
                thresh_optim_list.append(test_eval['report']['threshold_optim'])
                
                # Use individual predictions to evaluate ensemble predictions
                out_dir = log_dir.replace('%s_links' % link_cond, 'ensemble')
                evaluate_ensemble_predictions(
                    y_true_list, y_score_list, thresh_optim_list, out_dir)


def evaluate_ensemble_predictions(y_true_list: list[list[int]],
                                  y_score_list: list[list[float]],
                                  thresh_optim_list: list[float],
                                  out_dir: str,
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
            
            # F1-score optimized individual thresholds
            y_score, threshold = generate_ensemble_predictions(
                y_score_list, thresh_optim_list, strategy)
            report_optim = generate_report(y_true, y_score, threshold)
            report.update({'%s_optim' % k: v for k, v in report_optim.items()})
            
            # Write whole report
            out_dir_ = '%s_%s' % (out_dir, strategy)
            os.makedirs(out_dir_, exist_ok=True)
            report_path = os.path.join(out_dir_, 'gnn_report.json')
            with open(report_path, 'w') as f:
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
    