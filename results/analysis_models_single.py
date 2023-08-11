import os
import csv
import json


OUTPUT_BASE_PATH = os.path.join('results', 'tables', 'table')
ALL_SETTING_CONDS = ['inductive', 'transductive']
ALL_BALANCED_CONDS = ['under', 'non', 'over']
ALL_LINK_CONDS = [
    'all',
    'wards',
    'caregivers',
    'no',
    'ensemble_average',
    'ensemble_majority',
    'ensemble_unanimity',
]
ALL_CONTROL_MODELS = [
    'logistic_regression',
    'random_forest',
    'catboost',
    'knn',
    'ensemble_average',
    'ensemble_majority',
    'ensemble_unanimity',
]
CONDS_TABLE_2 = [
    
    # GNN models (transductive, all possible link types)
    {'Model': 'gnn', 'Setting': 'transductive', 'Links': 'all'},
    {'Model': 'gnn', 'Setting': 'transductive', 'Links': 'wards'},
    {'Model': 'gnn', 'Setting': 'transductive', 'Links': 'caregivers'},
    
    # GNN models (transductive, all possible link types)
    {'Model': 'gnn', 'Setting': 'inductive', 'Links': 'all'},
    {'Model': 'gnn', 'Setting': 'inductive', 'Links': 'wards'},
    {'Model': 'gnn', 'Setting': 'inductive', 'Links': 'caregivers'},
    
    # Control models (without node embeddings only)
    {'Model': 'catboost', 'Setting': '-',},
    {'Model': 'random_forest', 'Setting': '-',},
    {'Model': 'knn', 'Setting': '-',},   
    {'Model': 'logistic_regression', 'Setting': '-',},
    
    # Ensemble model (gnn-inductive, catboost, random forest)
    {'Model': 'all_ensemble_average'},    
    
]
CONDS_TABLE_3 = [
    
    # GNN and ensemble models (repeated from table 3)
    {'Model': 'gnn', 'Setting': 'inductive', 'Links': 'no'},
    {'Model': 'gnn', 'Setting': 'inductive'},
    {'Model': 'gnn', 'Setting': 'transductive'},
    
    # Catboost, with (in/trans-ductive) and without node embeddings
    {'Model': 'catboost', 'Setting': '-',},
    {'Model': 'catboost', 'Setting': 'inductive'},
    {'Model': 'catboost', 'Setting': 'transductive'},
    
    # Random forest, with (in/trans-ductive) and without node embeddings
    {'Model': 'random_forest', 'Setting': '-',},
    {'Model': 'random_forest', 'Setting': 'inductive'},
    {'Model': 'random_forest', 'Setting': 'transductive'},
    
    # KNN, with (in/trans-ductive) and without node embeddings
    {'Model': 'knn', 'Setting': '-',},
    {'Model': 'knn', 'Setting': 'inductive'},
    {'Model': 'knn', 'Setting': 'transductive'},
    
    # Logistic regression, with (in/trans-ductive) and without node embeddings
    {'Model': 'logistic_regression', 'Setting': '-',},
    {'Model': 'logistic_regression', 'Setting': 'inductive'},
    {'Model': 'logistic_regression', 'Setting': 'transductive'},
    
]


def main():
    os.makedirs(os.path.split(OUTPUT_BASE_PATH)[0], exist_ok=True)
    for use_optim_threshold in [True, False]:
        optim_str = '_optim' if use_optim_threshold else ''
        raw_table_path = '%s%s_raw.csv' % (OUTPUT_BASE_PATH, optim_str)
        table_2_path = '%s%s_2.csv' % (OUTPUT_BASE_PATH, optim_str)
        table_3_path = '%s%s_3.csv' % (OUTPUT_BASE_PATH, optim_str)
        compute_raw_results(raw_table_path, use_optim_threshold)
        write_spec_table(raw_table_path, table_2_path, CONDS_TABLE_2)
        write_spec_table(raw_table_path, table_3_path, CONDS_TABLE_3)
    
    
def compute_raw_results(raw_table_path, use_optim_threshold):
    """ Check in the logs of all trained models and write a table with all the
        results for all different conditions for all models
    """
    # First, check control models that use node features only
    dicts_to_write = []
    for balanced_cond in ALL_BALANCED_CONDS:
        ckpt_dir = os.path.join('models', 'controls', 'node_features')
        for model_name in ALL_CONTROL_MODELS:
            try:
                model_row = get_model_row(
                    ckpt_dir, model_name, use_optim_threshold,
                    balanced_cond=balanced_cond)
                dicts_to_write.append(model_row)
            except:
                pass  # for conditions that were not run yet
    
    # Second, check control models that use both node and edges features
    for setting_cond in ALL_SETTING_CONDS:
        for balanced_cond in ALL_BALANCED_CONDS:
            for link_cond in [c for c in ALL_LINK_CONDS if c != 'no']:
                ckpt_dir = os.path.join('models', 'controls', 'edge_features')
                for model_name in ALL_CONTROL_MODELS:
                    try:
                        model_row = get_model_row(ckpt_dir, model_name,
                            use_optim_threshold, balanced_cond=balanced_cond,
                            setting_cond=setting_cond, link_cond=link_cond)
                        dicts_to_write.append(model_row)
                    except:
                        pass  # for conditions that were not run yet
                
    # Third, check GNN networks
    model_name = 'gnn'
    for setting_cond in ALL_SETTING_CONDS:
        for balanced_cond in ALL_BALANCED_CONDS:
            for link_cond in ALL_LINK_CONDS:
                ckpt_dir = os.path.join('models', model_name)
                try:
                    model_row = get_model_row(
                        ckpt_dir, model_name, use_optim_threshold,
                        balanced_cond=balanced_cond, setting_cond=setting_cond,
                        link_cond=link_cond)
                    dicts_to_write.append(model_row)
                except:
                    pass  # for conditions that were not run yet
    
    # Finally, check ensemble model (all)
    ckpt_dir = os.path.join('models', 'all')
    model_name = 'ensemble_average'
    model_row = get_model_row(ckpt_dir, 'ensemble_average', use_optim_threshold)
    dicts_to_write.append(model_row)
    
    # Build final table
    write_table(dicts_to_write, raw_table_path)
    

def get_model_row(ckpt_dir: str,
                  model_name: str,
                  use_optim_threshold: bool,
                  balanced_cond: str='-',
                  setting_cond: str='-',
                  link_cond: str='-',
                  ) -> None:
    """ Print one row that summarizes one model results
    """
    # Load report data
    report_filename = os.path.join(
        ckpt_dir, 
        '%s_setting' % setting_cond if setting_cond != '-' else '',
        '%s_balanced' % balanced_cond if balanced_cond != '-' else '',
        link_cond if 'ensemble' in link_cond else\
            '%s_links' % link_cond if link_cond != '-' else '',
        model_name + '_report.json'
    )
    with open(report_filename, 'r') as f: report = json.load(f)
    if 'ensemble' in link_cond: model_name = '%s_%s' % (model_name, link_cond)
    if 'ensemble' in model_name: model_name = 'all_%s' % model_name
    
    # Return a dict formatted for a table ({column_name: metric value})
    accuracy_key = 'accuracy_optim' if use_optim_threshold else 'accuracy'
    macro_avg_key = 'macro avg_optim' if use_optim_threshold else 'macro avg'
    to_return = {
        'Model': model_name,
        'Balanced': balanced_cond,
        'Setting': setting_cond,
        'Links': link_cond,
        'Accuracy (%)': '%.2f' % (report[accuracy_key] * 100),
        'Precision (%)': '%.2f' % (report[macro_avg_key]['precision'] * 100),
        'Recall (%)': '%.2f' % (report[macro_avg_key]['recall'] * 100),
        'F1-score (%)': '%.2f' % (report[macro_avg_key]['f1-score'] * 100),
        # 'AUROC (%)': '%.2f' % (report['auroc'] * 100),
        'AUROC (%) (95% CI)': '%.2f (%.2f-%.2f)' % tuple([report[k] * 100
            for k in ['auroc', 'auroc-low', 'auroc-high']]),
    }
    return to_return


def write_spec_table(raw_table_path, table_path, cond_dicts):
    """ Read the raw results table to write a selection of rows to a new table
    """
    # Read raw data and initialize dict list that will be written
    raw_data = read_table(raw_table_path)
    dicts_to_write = []
    
    # Fill the dict list with the best model of each set of conditions
    for cond_dict in cond_dicts:
        to_select = lambda d, conds: all(d.get(k) == v for k, v in conds.items())
        # to_maximize = lambda d: d['AUROC (%)'].split(' ')[0]
        to_maximize = lambda d: d['AUROC (%) (95% CI)'].split(' ')[0]
        selected_data = [d for d in raw_data if to_select(d, cond_dict)]
        max_auroc_dict = max(selected_data, key=to_maximize)
        dicts_to_write.append(max_auroc_dict)
    
    # Build final table
    write_table(dicts_to_write, table_path)


def write_table(list_of_dicts: list[dict],
                table_path: str
                ) -> None:
    """ Write final results to a csv file
    """
    headers = list(list_of_dicts[0].keys())
    with open(table_path, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        for d in list_of_dicts:
            writer.writerow(d)


def read_table(table_path: str
               ) -> list[dict]:
    """ Read table as a list of dict where keys are headers
    """
    with open(table_path, 'r') as f:
        csv_dict_reader = csv.DictReader(f)
        data = []
        for row in csv_dict_reader:
            data.append(row)
    return data


if __name__ == '__main__':
    main()
    