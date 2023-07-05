# Healthcare-associated infections

Repository for the manuscript entitled "Detection of Patients at Risk of Enterobacteriaceae Infection Using Graph Neural Networks: a Retrospective Study"

### Dependencies

* Coming soon

### How to reproduce the results

* First, upload MIMIC-III data to data/physionet.org/files/mimiciii/1.4 (i.e., where .gz files are). Note that you should have access to the MIMIC-III database in order to fullfil this step.
* Don't forget to install the required libraries (see dependencies)
* Then, pre-process the data
```
python data/data_utils.py
```
* Second, run the control models
```
python run_controls.py
```
* Third, run the GNN models
```
python run_gnn.py
```
* Finally, generate all figures
```
python generate_all_figures.py  # coming soon
```
