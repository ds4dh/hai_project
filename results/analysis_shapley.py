import os
import sys
sys.path.append(os.path.abspath('.'))
import shap
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from typing import Callable
from tempfile import NamedTemporaryFile as temp_file
from run_controls import load_best_params, load_correct_data, initialize_model


OUTPUT_PATH = os.path.join('results', 'figures', 'figure_6')
DATA_SPLIT = 'train'  # 'test', 'train'
MASKER_KEY = 'indep'  # 'indep', 'part'
BALANCE = 'minor'  # 'none', 'minor', 100  # int sets both neg and pos to the int
PICKLE_PATH = '%s_%s_%s_%s' % (OUTPUT_PATH, DATA_SPLIT, MASKER_KEY, BALANCE)
LOAD_DATA = True
POSITIVE_ID = 1  # for label 'colonised'
MASKER_ALGORITHM = {
    'indep': shap.maskers.Independent,
    'part': shap.maskers.Partition,
}[MASKER_KEY]
MODEL_RUN = {
    'model_name': 'random_forest',  # 'logistic_regression', 'random_forest',
    'conds': {
        'feat_cond': 'nodes',
        'setting_cond': None,
        'balanced_cond': 'non',
        'link_cond': None,
    }
}
FEATURE_LABELS = {
    'PREV_WARDID': 'Previous ward',
    'CURR_WARDID': 'Current ward',
    'GENDER_F': 'Sex (female)',
    'GENDER_M': 'Sex (male)',
    'DIAG_ID': 'Diagnosis',
    'LOS': 'Length of stay in ward',
    'LOSH': 'Length of stay in hospital',
    'N_CONTACTS': 'Number of patients in ward',
    'N_COLONISED': 'Colonisation pressure (absolute)',
    'CP': 'Colonisation pressure (relative)',
    'PREV_CAREUNIT_CCU': 'Previous care-unit (CCU)',
    'PREV_CAREUNIT_CSRU': 'Previous care-unit (CSRU)',
    'PREV_CAREUNIT_MICU': 'Previous care-unit (MICU)',
    'PREV_CAREUNIT_NICU': 'Previous care-unit (NICU)',
    'PREV_CAREUNIT_NWARD': 'Previous care-unit (NWARD)',
    'PREV_CAREUNIT_SICU': 'Previous care-unit (SICU)',
    'PREV_CAREUNIT_TSICU': 'Previous care-unit (TSICU)',
    'CURR_CAREUNIT_CCU': 'Current care-unit (CCU)',
    'CURR_CAREUNIT_CSRU': 'Current care-unit (CSRU)',
    'CURR_CAREUNIT_MICU': 'Current care-unit (MICU)',
    'CURR_CAREUNIT_NICU': 'Current care-unit (NICU)',
    'CURR_CAREUNIT_NWARD': 'Current care-unit (NWARD)',
    'CURR_CAREUNIT_SICU': 'Current care-unit (SICU)',
    'CURR_CAREUNIT_TSICU': 'Current care-unit (TSICU)',
}


def main():
    """ Train best model in the best setting, data balance, and link condition,
        then check performance for different MDR categories
    """
    # Retrieve correct data and balance negative and positive samples if required
    print('Loading data')
    X, y = load_correct_data(MODEL_RUN['conds'])
    pos_data = X[DATA_SPLIT][y[DATA_SPLIT] == 1]
    neg_data = X[DATA_SPLIT][y[DATA_SPLIT] == 0]
    if BALANCE == 'none':
        n_pos_samples = len(pos_data)
        n_neg_samples = len(neg_data)
    elif BALANCE == 'minor':
        n_pos_samples = n_neg_samples = min(len(pos_data), len(neg_data))
    elif isinstance(BALANCE, int):
        assert 0 < BALANCE < min(len(pos_data), len(neg_data))
        n_pos_samples = n_neg_samples = BALANCE
    else:
        raise ValueError('Invalid balance parameter (none, minor or integer)')
    pos_ids = np.random.choice(len(pos_data), size=n_pos_samples, replace=False)
    neg_ids = np.random.choice(len(neg_data), size=n_neg_samples, replace=False)
    shap_data = np.concatenate([pos_data[pos_ids], neg_data[neg_ids]])
    
    # Run entire Shapley analysis if required
    if not LOAD_DATA:
        
        # Re-train best model (always with full training data)
        print('Re-training best %s model' % MODEL_RUN['model_name'])
        best_params = load_best_params(MODEL_RUN['conds'], MODEL_RUN['model_name'])
        model = initialize_model(MODEL_RUN['model_name'], best_params)
        model.fit(X['train'], y['train'])
        
        # Run shapley analysis (with required data)
        print('Running Shapley analysis')
        masker = MASKER_ALGORITHM(shap_data)
        shap_fn = lambda x: model.predict_proba(x)[:, POSITIVE_ID]
        explainer = shap.Explainer(shap_fn, masker)
        shap_vals = explainer(shap_data)  # long step
        with open('%s.pkl' % PICKLE_PATH, 'wb') as f:
            pickle.dump(shap_vals, f)
    
    # Load data from a previous run if required
    else:
        print('Loading shapley values from %s' % PICKLE_PATH)
        with open('%s.pkl' % PICKLE_PATH, 'rb') as f:
            shap_vals = pickle.load(f)
        assert shap_vals.shape == shap_data.shape
        
    # Plot results of the analysis
    print('Plotting results')
    balanced = '%s_balanced' % MODEL_RUN['conds']['balanced_cond']
    test_info_path = os.path.join('data', 'processed', balanced, 'X_test.pkl')
    feature_names = pd.read_pickle(test_info_path).columns.tolist()
    shap_vals.feature_names = [FEATURE_LABELS[n] for n in feature_names]
    summary_plot_path = save_temp_figure(shap.summary_plot, shap_vals, shap_data,
                                         max_display=11, plot_size=(7, 6.5))
    bar_plot_path = save_temp_figure(shap.plots.bar, shap_vals,
                                     max_display=12, show=False)
    merge_figures(summary_plot_path, bar_plot_path, '%s.png' % OUTPUT_PATH)
    
    
def save_temp_figure(plot_fn: Callable,
                     *args, **kwargs
                     ) -> None:
    """ Save one figure to a temporary path by invoking the plot function and its
        arguments, then return the temporary path for further processing
    """
    plot_path = temp_file(suffix='.png').name
    plt.figure()
    plot_fn(*args, **kwargs)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    return plot_path
    
    
def merge_figures(first_figure_path: str,
                  second_figure_path: str,
                  fig_path: str,
                  ) -> None:
    """ Load two figures and create a single figure with two subplots
    """
    first_figure = mpimg.imread(first_figure_path)
    second_figure = mpimg.imread(second_figure_path)
    _, axs = plt.subplots(1, 2)
    axs[0].imshow(first_figure)
    axs[0].axis('off')
    axs[0].text(0.02, 0.99, 'a', transform=axs[0].transAxes, fontsize=10, va='top')
    axs[1].imshow(second_figure)
    axs[1].axis('off')
    axs[1].text(0.02, 0.99, 'b', transform=axs[1].transAxes, fontsize=9, va='top')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.0, wspace=0.1)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(fig_path, dpi=300, bbox_inches = 'tight', pad_inches = 0)
    
    
if __name__ == '__main__':
    main()
    