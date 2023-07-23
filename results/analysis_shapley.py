import os
import sys
sys.path.append(os.path.abspath('.'))
import shap
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tempfile import NamedTemporaryFile as temp_file
from run_controls import load_best_params, load_correct_data, initialize_model


OUTPUT_PATH = os.path.join('results', 'figures', 'figure_6.png')
PICKLE_PATH = OUTPUT_PATH.replace('.png', '_data.pkl')
LOAD_DATA = False
MODEL_RUN = {
    'model_name': 'random_forest',  # 'logistic_regression', 'random_forest',
    'conds': {
        'feat_cond': 'nodes',
        'setting_cond': None,
        'balanced_cond': 'non',
        'link_cond': None,
    }
}
POSITIVE_ID = 1


def main():
    """ Train best model in the best setting, data balance, and link condition,
        then check performance for different MDR categories
    """
    # Retrieve correct data
    print('Loading data')
    X, y = load_correct_data(MODEL_RUN['conds'])
    pos_data = X['test'][y['test'] == 1]
    neg_data = X['test'][y['test'] == 0]
    n_pos_samples = len(pos_data)  # 10, "len(pos_data)" for no sub-sampling
    n_neg_samples = len(pos_data)  # 10, "len(pos_data)" for balanced shap data
    pos_ids = np.random.choice(len(pos_data), size=n_pos_samples, replace=False)
    neg_ids = np.random.choice(len(neg_data), size=n_neg_samples, replace=False)
    shap_data = np.concatenate([pos_data[pos_ids], neg_data[neg_ids]])
    
    # Run entire Shapley analysis if required
    if not LOAD_DATA:
        
        # Re-train best model
        print('Re-training best %s model' % MODEL_RUN['model_name'])
        best_params = load_best_params(MODEL_RUN['conds'], MODEL_RUN['model_name'])
        model = initialize_model(MODEL_RUN['model_name'], best_params)
        model.fit(X['train'], y['train'])
        
        # Run shapley analysis
        print('Running Shapley analysis')
        masker = shap.maskers.Independent(shap_data)
        shap_fn = lambda x: model.predict_proba(x)[:, POSITIVE_ID]
        explainer = shap.Explainer(shap_fn, masker)
        shap_values = explainer(shap_data)  # long step
        with open(PICKLE_PATH, 'wb') as f:
            pickle.dump(shap_values, f)
    
    # Load data from a previous run if required
    else:
        print('Loading shapley values from %s' % PICKLE_PATH)
        with open(PICKLE_PATH, 'rb') as f:
            shap_values = pickle.load(f)
        
    # Plot results of the analysis
    print('Plotting results')
    balanced = '%s_balanced' % MODEL_RUN['conds']['balanced_cond']
    test_info_path = os.path.join('data', 'processed', balanced, 'X_test.pkl')
    shap_values.feature_names = pd.read_pickle(test_info_path).columns.tolist()
    summary_plot_path = temp_file(suffix='.png').name
    bar_plot_path = temp_file(suffix='.png').name
    save_one_figure(summary_plot_path, shap.summary_plot,
                    shap_values, shap_data, max_display=11, plot_size=(7, 6.5))
    save_one_figure(bar_plot_path, shap.plots.bar,
                    shap_values, max_display=12, show=False)
    merge_figures(summary_plot_path, bar_plot_path, OUTPUT_PATH)
    
    
def save_one_figure(plot_path, plot_fn, *args, **kwargs):
    """ Save one figure, invoking the plot function and its arguments
    """
    plt.figure()
    plot_fn(*args, **kwargs)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    
    
def merge_figures(first_figure_path: str,
                  second_figure_path: str,
                  fig_path: str,
                  ) -> None:
    """ Load two figures and create a single figure with two subplots
    """
    first_figure = mpimg.imread(first_figure_path)
    second_figure = mpimg.imread(second_figure_path)
    _, axs = plt.subplots(1, 2)  #, figsize=(16, 8))
    axs[0].imshow(first_figure)
    axs[0].axis('off')
    axs[0].text(0.05, 0.98, 'a', transform=axs[0].transAxes, fontsize=12, va='top')
    axs[1].imshow(second_figure)
    axs[1].axis('off')
    axs[1].text(0.05, 0.98, 'b', transform=axs[1].transAxes, fontsize=11, va='top')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.0, wspace=0.1)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(fig_path, dpi=300, bbox_inches = 'tight', pad_inches = 0)
    
    
if __name__ == '__main__':
    main()
    