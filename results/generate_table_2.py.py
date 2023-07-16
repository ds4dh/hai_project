import os
import csv
import json

OUTPUT_PATH = os.path.join('results', 'table_2.csv')
USE_OPTIMAL_THRESHOLD = True  # if False, use results with threshold = 0.5
SETTING_CONDS = ['inductive', 'transductive']
BALANCED_CONDS = ['under', 'non', 'over']
LINK_CONDS = ['all', 'wards', 'caregivers', 'no']
CONTROL_MODEL_NAMES = [
    'logistic_regression',
    'random_forest',
    'catboost',
    'knn',
]


def main():
    # Initialize result file
    os.makedirs(os.path.split(OUTPUT_PATH)[0], exist_ok=True)
    result_dicts = []
    
    # First, check control models that use node features only
    for balanced_cond in BALANCED_CONDS:
        ckpt_dir = os.path.join('models', 'controls', 'node_features')
        for model_name in CONTROL_MODEL_NAMES:
            try:
                model_row = get_model_row(ckpt_dir, model_name, balanced_cond)
                result_dicts.append(model_row)
            except:
                pass
    
    # Second, check control models that use both node and edges features
    for setting_cond in SETTING_CONDS:
        for balanced_cond in BALANCED_CONDS:
            for link_cond in LINK_CONDS:
                ckpt_dir = os.path.join('models', 'controls', 'edge_features')
                for model_name in CONTROL_MODEL_NAMES:
                    try:
                        model_row = get_model_row(ckpt_dir, model_name,
                            balanced_cond, setting_cond, link_cond)
                        result_dicts.append(model_row)
                    except:
                        pass
                
    # Third, check GNN networks
    for setting_cond in SETTING_CONDS:
        for balanced_cond in BALANCED_CONDS:
            for link_cond in LINK_CONDS:
                ckpt_dir = os.path.join('models', 'gnn')
                try:
                    model_row = get_model_row(
                        ckpt_dir, 'gnn', balanced_cond, setting_cond, link_cond)
                    result_dicts.append(model_row)
                except:
                    pass
    
    # Build final table
    write_table(result_dicts)
    

def get_model_row(ckpt_dir: str,
                  model_type: str,
                  balanced_cond: str,
                  setting_cond: str='-',
                  link_cond: str='-',
                  ) -> None:
    """ Lalalala
    """
    # Load report data
    report_filename = os.path.join(
        ckpt_dir, 
        '%s_setting' % setting_cond if setting_cond != '-' else '',
        '%s_balanced' % balanced_cond,
        '%s_links' % link_cond if link_cond != '-' else '',
        model_type + '_report.json'
    )
    with open(report_filename, 'r') as f: report = json.load(f)
    
    # Return a dict formatted for a table ({column_name: metric value})
    accuracy_key = 'accuracy_optim' if USE_OPTIMAL_THRESHOLD else 'accuracy'
    macro_avg_key = 'macro avg_optim' if USE_OPTIMAL_THRESHOLD else 'macro avg'
    to_return = {
        'Model': model_type,
        'Balanced': balanced_cond,
        'Setting': setting_cond,
        'Links': link_cond,
        'Accuracy (%)': '%.2f' % (report[accuracy_key] * 100),
        'Precision (%)': '%.2f' % (report[macro_avg_key]['precision'] * 100),
        'Recall (%)': '%.2f' % (report[macro_avg_key]['recall'] * 100),
        'F1-score (%)': '%.2f' % (report[macro_avg_key]['f1-score'] * 100),
        'AUROC (95% CI)': '%.2f (%.2f-%.2f)' % tuple([report[k] * 100
            for k in ['auroc', 'auroc-low', 'auroc-high']]),
    }
    return to_return
    

def write_table(list_of_dicts: list[dict]):
    """ Lalalala
    """
    headers = list(list_of_dicts[0].keys())
    with open(OUTPUT_PATH, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        for d in list_of_dicts:
            writer.writerow(d)
    

if __name__ == '__main__':
    main()
    