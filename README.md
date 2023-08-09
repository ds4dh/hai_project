# Healthcare-associated infections

Repository for the manuscript entitled "Detection of Patients at Risk of Enterobacteriaceae Infection Using Graph Neural Networks: a Retrospective Study"

### How to reproduce the results

* First, upload MIMIC-III data to data/physionet.org/files/mimiciii/1.4 (i.e., where .gz files are). Note that you should have access to the MIMIC-III database in order to fullfil this step.
* Don't forget to install the required libraries (see environment.yml)
* Then, pre-process the data to generate the relevant datasets
```
python data/data_utils.py
```
* Second, run the control models (best hyper-parameters were already computed in models/controls but you re-do the tuning process by setting "DO_HYPER_OPTIM" to True in run_controls.py)
```
python run_controls.py
```
* Third, run the GNN models (best hyper-parameters were already computed in models/gnn but you re-do the tuning process by setting "DO_HYPER_OPTIM" to True in run_gnn.py)
```
python run_gnn.py
```
* Fourth, run the ensemble model (composed of inductive-gnn, random forest, and catboost)
```
python run_ensemble.py
```
* Finally, generate all result figures and tables
```
python results/analysis_models_single.py  # Tables 2 and 3
python results/analysis_by_category.py  # Figure 4
python results/analysis_roc_curves.py  # Figure 5
python results/analysis_shapley.py  # Figure 6
```
