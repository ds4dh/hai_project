import os
import sys
sys.path.append(os.path.abspath('.'))
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from data.data_utils import load_features_and_labels
from run_gnn import IPCDataset, evaluate_model, get_ckpt_dir, load_best_params
from run_utils import auroc_ci


MODEL_RUN = {
    'setting_cond': 'ensemble',  # ensemble, inductive, transductive
    'balanced_cond': 'non',  # only relevant with setting_cond = in/trans-ductive
    'link_cond': 'wards',  # only relevant with setting_cond = in/trans-ductive
}
FIGURE_FILE = 'figure_4_%s.png' % MODEL_RUN['setting_cond']
OUTPUT_PATH = os.path.join('results', 'figures', FIGURE_FILE)
BAR_PLOT_COLORS = [
    (0.627, 0.306, 0.310),
    (0.353, 0.463, 0.553),
    (0.384, 0.537, 0.376),
    (0.502, 0.384, 0.518),
    (0.663, 0.463, 0.259),
    (0.663, 0.663, 0.349),
    (0.522, 0.400, 0.329),
    (0.651, 0.467, 0.541),
    (0.502, 0.502, 0.502),
]
BAR_PLOT_COLORS = [tuple([a * 1.5 for a in c]) for c in BAR_PLOT_COLORS]
CAT_CONDS = {
    'ORG_NAME': {
        'C. amanolaticus': lambda x: x == 'CITROBACTER AMALONATICUS',
        'C. freundii': lambda x: x == 'CITROBACTER FREUNDII COMPLEX',
        'C. koseri': lambda x: x == 'CITROBACTER KOSERI',
        'E. absurdiae': lambda x: x == 'ENTEROBACTER ASBURIAE',
        'E. cloacae': lambda x: x in [
            'ENTEROBACTER CLOACAE',
            'ENTEROBACTER CLOACAE COMPLEX',
        ],
        'E. coli': lambda x: x == 'ESCHERICHIA COLI',
        'K. oxytoca': lambda x: x == 'KLEBSIELLA OXYTOCA',
        'K. pneumoniae': lambda x: x == 'KLEBSIELLA PNEUMONIAE',
        'S. enterica': lambda x: x in [
            'SALMONELLA DUBLIN',
            'SALMONELLA ENTERITIDIS',
            'SALMONELLA DUBLIN',
            'SALMONELLA ENTERITIDIS',
            'SALMONELLA HADAR',
        ],
        # 'C. youngae': lambda x: x == 'CITROBACTER YOUNGAE',
        # 'E. fergunosi': lambda x: x == 'ESCHERICHIA FERGUSONII',
        # 'R. ornithinolytica': lambda x: x == 'RAOULTELLA ORNITHINOLYTICA',
        # 'L. adecarboxylata': lambda x : x == 'LECLERCIA ADECARBOXYLATA',
        # 'S. flexneri': lambda x: x == 'SHIGELLA FLEXNERI',
    },
    'SPEC_TYPE_DESC': {
        'bal': lambda x: x in ['BRONCHOALVEOLAR LAVAGE', 'BRONCHIAL WASHINGS'],
        'blood': lambda x: x in [
            'BLOOD CULTURE',
            'BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)',
            'FLUID RECEIVED IN BLOOD CULTURE BOTTLES',
            'BLOOD CULTURE - NEONATE',
            'BLOOD CULTURE (POST-MORTEM)',
        ],
        'sputum': lambda x: x == 'SPUTUM',
        'swab': lambda x: x == 'SWAB',
        'urine': lambda x: x in ['URINE', 'URINE,KIDNEY'],
        'other': lambda x: x in [
            'ABSCESS',
            'ASPIRATE',
            'BILE',
            'BIOPSY',
            'CATHETER TIP-IV',
            'CSF;SPINAL FLUID',
            'DIALYSIS FLUID',
            'EAR',
            'EYE',
            'FLUID WOUND',
            'FLUID,OTHER',
            'FOREIGN BODY',
            'Isolate',
            'JOINT FLUID',
            'PERITONEAL FLUID',
            'PLEURAL FLUID',
            'TRACHEAL ASPIRATE',
            'FOOT CULTURE',
        ],
    },
    'LOSH': {
        '0-4': lambda x: x <= 4,
        '4-10': lambda x: 4 < x <= 10,
        '10-50': lambda x: 10 < x <= 50,
        '50-100': lambda x: 50 < x <= 100,
        '>100': lambda x: 100 < x,
    },
    'MDR_STATUS': {
        'AMS': lambda x: x == 'AMS',
        'AMR': lambda x: x == 'AMR',
        'MDR': lambda x: x == 'MDR',
    },
}
CAT_TITLES = {
    'ORG_NAME': {'label': 'organism', 'letter': 'a'},
    'SPEC_TYPE_DESC': {'label': 'specimen', 'letter': 'b'},
    'LOSH': {'label': 'length of stay', 'letter': 'c'},
    'MDR_STATUS': {'label': 'resistance profile', 'letter': 'd'},
}


def main():
    """ Train best model in the best setting, data balance, and link condition,
        then check performance for different MDR categories
    """
    # Initialize dataset and model hyper-parameters, given conditions
    if 'ensemble' in MODEL_RUN['setting_cond']:
        log_dir = os.path.join('models', 'all')
    else:
        dataset = IPCDataset(**MODEL_RUN)
        log_dir = get_ckpt_dir(MODEL_RUN)
        params = load_best_params(log_dir)
    
    # Retrieve relevant data info (by sample index)
    _, _, ids = load_features_and_labels(balanced=MODEL_RUN['balanced_cond'])
    data_dir = os.path.join('data', 'processed')
    sample_info = pd.read_csv(os.path.join(data_dir, 'patient-ward_info.csv'))
    test_info = sample_info.loc[ids['test']]
    
    # Generate predictions and plot them for different set of categories
    if 'ensemble' in MODEL_RUN['setting_cond']:
        report_file = 'ensemble_average_report.json'
        report_path = os.path.join(log_dir, report_file)
        with open(report_path, 'r') as f: report = json.load(f)
        y_score = np.array(report['y_score_optim'])
    else:
        metrics = evaluate_model(
            dataset, params, MODEL_RUN['setting_cond'], log_dir)
        y_score = metrics['y_score']
    plot_results_by_category(test_info, y_score)
    
    
def plot_results_by_category(test_info, y_score):
    """ Compute performance for different sets of categories and show the results
        on several bar plots
    """
    # Gather test results and plot them for each set of categories
    _, axs = plt.subplots(1, len(CAT_CONDS), figsize=(len(CAT_CONDS) * 4.5, 5))
    for i, (cond_key, cond_dict) in enumerate(CAT_CONDS.items()):
        
        # Compute metrics for this category
        aurocs, auroc_lows, auroc_highs = compute_metrics(
            test_info, y_score, cond_key, cond_dict
        )
        
        # Show results as a bar plot
        cats = list(cond_dict.keys())
        colors = BAR_PLOT_COLORS[:len(cats)]
        axs[i].bar(
            cats, aurocs, yerr=[auroc_lows, auroc_highs], zorder=10, alpha=0.85,
            color=colors, capsize=5, label=cats, align='center', ecolor='black',
        )
        
        # Polish figure
        axs[i].set_ylabel('AUROC', fontsize='x-large')
        axs[i].set_xlabel(CAT_TITLES[cond_key]['label'], fontsize='x-large')
        axs[i].set_ylim([0.0, 1.05])
        axs[i].set_xticklabels([])
        axs[i].tick_params(axis='y', labelsize='large')
        axs[i].yaxis.grid(True, color=(0.8, 0.8, 0.8))
        axs[i].xaxis.grid(True, color=(0.8, 0.8, 0.8))
        handles = [mlines.Line2D(
            [], [], color=colors[i], marker='s', linestyle='None',
            markersize=10, markeredgecolor='black') for i, _ in enumerate(cats)]
        axs[i].text(-0.17, 1.0, CAT_TITLES[cond_key]['letter'], va='top',
                    ha='right', fontsize='xx-large', transform=axs[i].transAxes)
        axs[i].legend(handles, cats, fontsize=14, ncol=2, loc='upper center',
                      bbox_to_anchor=(0.5, -0.1, 0.0, 0.0), columnspacing=0.5,
                      handletextpad=0.1)
        
    # Save figure
    plt.subplots_adjust(wspace=0.3)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    
    
def compute_metrics(test_info, y_score, cond_key, cond_dict):
    """ Compute auroc and confidence interval for different categories that
        belong to a specific category set
    """
    # Retrieve relevant information
    test_label_info = test_info['COLONISED']
    test_cond_info_pos = test_info[test_label_info == 1][cond_key]
    test_cond_info_neg = test_info[test_label_info == 0][cond_key]
    y_score_pos = y_score[test_label_info == 1]
    y_score_neg = y_score[test_label_info == 0]
    
    # Go through all condition of the condition set defined by cond_dict
    aurocs, auroc_lows, auroc_highs = [], [], []
    for _, cat_cond in cond_dict.items():
        
        # Retrieve positive and negative samples to check, taking into account
        # that most categories (all but 'LOSH') have no negative samples. In this
        # case, use all negative samples and condition negative samples
        y_score_pos_cat = y_score_pos[test_cond_info_pos.apply(cat_cond)]
        y_score_neg_cat = y_score_neg[test_cond_info_neg.apply(cat_cond)]
        if cond_key != 'LOSH':
            y_score_check = np.concatenate([y_score_pos_cat, y_score_neg])
            y_true_check = np.array(
                [1] * len(y_score_pos_cat) + [0] * len(y_score_neg)
            )
        else:
            y_score_check = np.concatenate([y_score_pos_cat, y_score_neg_cat])
            y_true_check = np.array(
                [1] * len(y_score_pos_cat) + [0] * len(y_score_neg_cat)
            )
        
        # Compute and record results (auroc and confidence interval)
        auroc, auroc_low, auroc_high = auroc_ci(y_true_check, y_score_check)
        aurocs.append(auroc)
        auroc_lows.append(auroc - max(0.5, auroc_low))
        auroc_highs.append(min(1.0, auroc_high) - auroc)
    
    # Return results
    return aurocs, auroc_lows, auroc_highs
    

if __name__ == '__main__':
    main()
    