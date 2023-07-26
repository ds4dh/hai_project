import os
import sys
sys.path.append(os.path.abspath('.'))
import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from data.data_utils import load_features_and_labels
from run_gnn import IPCDataset, evaluate_model as evaluate_model_gnn
from run_controls import load_correct_data, evaluate_model as evaluate_model_ctrl
from sklearn.metrics import roc_curve, roc_auc_score


OUTPUT_PATH = os.path.join('results', 'figures', 'figure_5.png')
TABLE_PATH = os.path.join('results', 'tables', 'table_figure_5.csv')
RUN_MODELS = {
    'logistic-regression': {
        'model_dir': os.path.join('models', 'controls', 'node_features'),
        'type': 'logistic_regression',
        'feat_cond': 'nodes',
        'setting_cond': None,
        'balanced_cond': 'non',
        'link_cond': None,
        'plot_color': (0.995, 0.695, 0.389),
    },
    'knn': {
        'model_dir': os.path.join('models', 'controls', 'node_features'),
        'type': 'knn',
        'feat_cond': 'nodes',
        'setting_cond': None,
        'balanced_cond': 'under',  # only model to improve with data balance
        'link_cond': None,
        'plot_color': (0.576, 0.806, 0.564),
    },
    'catboost': {
        'model_dir': os.path.join('models', 'controls', 'node_features'),
        'type': 'catboost',
        'feat_cond': 'nodes',
        'setting_cond': None,
        'balanced_cond': 'non',
        'link_cond': None,
        'plot_color': (0.753, 0.576, 0.777),
    },
    'random-forest': {
        'model_dir': os.path.join('models', 'controls', 'node_features'),
        'type': 'random_forest',
        'feat_cond': 'nodes',
        'setting_cond': None,
        'balanced_cond': 'non',
        'link_cond': None,
        'plot_color': (0.977, 0.700, 0.812),
    },
    'ensemble-all': {
        'model_dir': os.path.join('models', 'all'),
        'type': 'ensemble_average',
        'balanced_cond': 'non',
        'plot_color': (0.753, 0.753, 0.753),
    },
    'gnn-inductive': {
        'model_dir': os.path.join('models', 'gnn'),
        'type': 'gnn',
        'feat_cond': None,
        'setting_cond': 'inductive',
        'balanced_cond': 'non',
        'link_cond': 'wards',
        'plot_color': (1.000, 0.350, 0.350),
    },
    'gnn-transductive': {
        'model_dir': os.path.join('models', 'gnn'),
        'type': 'gnn',
        'feat_cond': None,
        'setting_cond': 'transductive',
        'balanced_cond': 'non',
        'link_cond': 'wards',
        'plot_color': (0.530, 0.695, 0.830),
    },
}
UNUSED_PLOT_COLORS = [  # in case needed
    (0.995, 0.995, 0.524),
    (0.783, 0.600, 0.494),
]
PLOT_COND_DICT = {
    'a - AMS': lambda x: x['MDR_STATUS'] == 'AMS',
    'b - AMR': lambda x: x['MDR_STATUS'] == 'AMR',
    'c - MDR': lambda x: x['MDR_STATUS'] == 'MDR',
    'd - MDR E. coli': lambda x:\
        (x['MDR_STATUS'] == 'MDR') and
        (x['ORG_NAME'] == 'ESCHERICHIA COLI'),
    'e - MDR K. pneumoniae': lambda x:\
        (x['MDR_STATUS'] == 'MDR') and
        (x['ORG_NAME'] in [
            'ENTEROBACTER CLOACAE',
            'ENTEROBACTER CLOACAE COMPLEX'
        ]),
    'f - MDR E. cloacae': lambda x:\
        (x['MDR_STATUS'] == 'MDR') and
        (x['ORG_NAME'] in [
            'ENTEROBACTER CLOACAE',
            'ENTEROBACTER CLOACAE COMPLEX'
        ]),
}


def main():
    """ Retrieve dataset and result directory, then run model, for all models
    """
    # Run all models and draw on the same figure
    fig, axs = plt.subplots(2, len(PLOT_COND_DICT) // 2, figsize=(10, 6.3))
    handles = []
    for model_key, run_model in RUN_MODELS.items():
        if 'ensemble' in run_model['type']:
            log_dir = run_model['model_dir']
        else:
            log_dir = os.path.join(
                run_model['model_dir'],
                '%s_setting' % run_model['setting_cond']
                    if run_model['setting_cond'] is not None else '',
                '%s_balanced' % run_model['balanced_cond']
                    if run_model['balanced_cond'] is not None else '',
                '%s_links' % run_model['link_cond']
                    if run_model['link_cond'] is not None else '',
            )
        run_one_model(log_dir, run_model, model_key, axs.flatten())
        handles.append(mlines.Line2D(
            [], [], color=run_model['plot_color'], marker='s', linestyle='None',
            markersize=6, markeredgecolor='black', label=model_key,
        ))
        
    # Save final figure
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               columnspacing=0.5, handletextpad=0.1)
    fig.subplots_adjust(
        left=0.06, right=0.965, bottom=0.12, top=0.95, wspace=0.33, hspace=0.33)
    fig.savefig(OUTPUT_PATH, dpi=300)


def run_one_model(log_dir, run_model, model_key, axs):
    """ Train best model in the best setting, data balance, and link condition,
        then check performance for different MDR categories
    """
    # Retrieve model best parameters
    print('\nEvaluating model with the following parameters %s' % run_model)
    if 'ensemble' in run_model['type']:
        best_params = None
    else:
        path = os.path.join(log_dir, '%s_best_params.json' % run_model['type'])
        with open(path, 'r') as f:
            best_params = json.load(f)
    
    # Retrieve relevant data info (by sample index)
    _, _, ids = load_features_and_labels(balanced=run_model['balanced_cond'])
    data_dir = os.path.join('data', 'processed')
    sample_info = pd.read_csv(os.path.join(data_dir, 'patient-ward_info.csv'))
    test_info = sample_info.loc[ids['test']]
    
    # Generate predictions and plot them for different set of categories
    y_score = compute_y_score(run_model, best_params, log_dir)
    fprs, tprs, aurocs = compute_metrics(test_info, y_score, PLOT_COND_DICT)
    
    # Plot roc curves for each set of categories
    titles = list(PLOT_COND_DICT.keys())
    color = run_model['plot_color']
    write_table(TABLE_PATH, model_key, titles, aurocs)
    for fpr, tpr, axs, title in zip(fprs, tprs, axs, titles):
        axs.plot(fpr, tpr, lw=1.0, color=color, label='_nolegend_')
        axs.plot([0, 1], [0, 1], lw=1.0, color='gray', label='_nolegend_')
        polish_figure(axs, title)
        
        
def compute_y_score(run_model, best_params, log_dir):
    """ Compute prediction scores for the testing dataset, for the correct model
    """
    if 'ensemble' in run_model['type']:
        report_path = os.path.join(log_dir, '%s_report.json' % run_model['type'])
        with open(report_path, 'r') as f: report = json.load(f)
        y_score = np.array(report['y_score_optim'])
    elif run_model['type'] == 'gnn':
        dataset = IPCDataset(**run_model)
        y_score = evaluate_model_gnn(
            dataset, best_params, run_model['setting_cond'], log_dir)['y_score']
    else:
        keys = [k + '_cond' for k in ['feat', 'setting', 'balanced', 'link']]
        conds = {k: run_model[k] for k in keys}
        X, y = load_correct_data(conds)
        _, y_score = evaluate_model_ctrl(X, y, run_model['type'], best_params)
    return y_score
    
    
def compute_metrics(test_info, y_score, cond_dict):
    """ Compute auroc and confidence interval for different categories that
        belong to a specific category set
    """
    # Retrieve relevant information
    test_label_info = test_info['COLONISED']
    test_cond_info_pos = test_info[test_label_info == 1]  #[cond_key]
    y_score_pos = y_score[test_label_info == 1]
    y_score_neg = y_score[test_label_info == 0]
    
    # Go through all condition of the condition set defined by cond_dict
    fprs, tprs, aurocs = [], [], []
    for _, cat_cond in cond_dict.items():
        
        # Retrieve positive and negative samples to check, taking into account
        # categories have no negative samples. For this reason, use all negative
        # samples and condition positive samples.
        y_score_pos_cat = y_score_pos[test_cond_info_pos.apply(cat_cond, axis=1)]
        y_score_check = np.concatenate([y_score_pos_cat, y_score_neg])
        y_true_check = np.array(
            [1] * len(y_score_pos_cat) + [0] * len(y_score_neg)
        )
        
        # Compute and record results (auroc and confidence interval)
        fpr, tpr, _ = roc_curve(y_true_check, y_score_check)
        auroc = roc_auc_score(y_true_check, y_score_check)
        fprs.append(fpr), tprs.append(tpr); aurocs.append(auroc)
        
    # Return results
    return fprs, tprs, aurocs


def polish_figure(ax, title):
    """ Reproduce cool R behaviour with denser grid lines around 0 and 1
    """
    # Set minor ticks
    minor_ticks = np.concatenate([
        np.arange(0.0, 0.1, 0.01),
        np.arange(0.9, 1.0, 0.01)
    ])
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)
    
    # Set major ticks
    major_ticks = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    major_tick_labels = [f'{tick:.02f}' for tick in major_ticks]
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.set_xticklabels(major_tick_labels, fontsize='smaller')
    ax.set_yticklabels(major_tick_labels, fontsize='smaller')
        
    # Enable grid for major and minor ticks
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.5)
    ax.grid(which='major', alpha=0.5)
    
    # Remove what is around the plot
    ax.tick_params(axis='both', which='both', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add custom axis labels and titles
    ax.text(0.99, -0.14, 'False positive rate', transform=ax.transAxes,
            ha='right', fontsize=10)
    ax.text(-0.2, 0.98, 'True positive rate', transform=ax.transAxes,
            va='top', fontsize=10, rotation='vertical')
    ax.text(-0.21, 1.11, title, transform=ax.transAxes,
            fontsize=12, va='top', ha='left')
    
    
def write_table(table_path: str,
                model_key: str,
                titles: list[str],
                aurocs: list[float]
                ) -> None:
    """" Record the auroc results for reporting in the manuscript
    """
    with open(table_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for title, auroc in zip(titles, aurocs):
            writer.writerow([model_key, title, auroc])
            
    
if __name__ == '__main__':
    main()
    