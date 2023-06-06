import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import warnings
from pandas.errors import DtypeWarning
warnings.filterwarnings(action='ignore', category=DtypeWarning)

# Input file paths
PATH_ICU_STAYS = 'data/physionet.org/files/mimiciii/1.4/ICUSTAYS.csv.gz'
PATH_ADMISSIONS = 'data/physionet.org/files/mimiciii/1.4/ADMISSIONS.csv.gz'
PATH_PATIENTS = 'data/physionet.org/files/mimiciii/1.4/PATIENTS.csv.gz'
PATH_TRANSFERS = 'data/physionet.org/files/mimiciii/1.4/TRANSFERS.csv.gz'
PATH_MICROBIOLOGY_EVENTS = 'data/physionet.org/files/mimiciii/1.4/MICROBIOLOGYEVENTS.csv.gz'
PATH_CHART_EVENTS = 'data/physionet.org/files/mimiciii/1.4/CHARTEVENTS.csv.gz'

# Output file paths (features)
PROCESSED_DATA_DIR = os.path.join('data', 'processed')
PATH_PATIENT_WARDS = os.path.join(PROCESSED_DATA_DIR, 'patient-wards.csv')
PATH_COLONISATION_LABELS = os.path.join(PROCESSED_DATA_DIR, 'patient-ward_colonisation_labels.csv')
PATH_DIAGNOSE_DATA = os.path.join(PROCESSED_DATA_DIR, 'patient-ward_diagnose_data.csv')
PATH_COLUMNS_AND_LABELS = os.path.join(PROCESSED_DATA_DIR, 'patient-ward_columns_and_labels.csv')
PATH_FEATURES_AND_LABELS = os.path.join(PROCESSED_DATA_DIR, 'patient-ward_features_and_labels.csv')

# Ouput file paths (links)
PATH_PATIENT_WARD_CAREGIVER_MAPPING = os.path.join(PROCESSED_DATA_DIR, 'patient-ward_caregiver_mapping.csv')
PATH_WARD_LINKS = os.path.join(PROCESSED_DATA_DIR, 'graph_links_wards.csv')
PATH_CAREGIVER_LINKS = os.path.join(PROCESSED_DATA_DIR, 'graph_links_caregivers.csv')
PATH_ALL_LINKS = os.path.join(PROCESSED_DATA_DIR, 'graph_links_all.csv')

# Relevant column names in various data files
ENTEROBACTERIAE = [
    'KLEBSIELLA PNEUMONIAE', 'ESCHERICHIA COLI', 'ENTEROBACTER CLOACAE',
    'KLEBSIELLA OXYTOCA', 'CITROBACTER KOSERI', 'CITROBACTER FREUNDII COMPLEX',
    'ENTEROBACTER ASBURIAE', 'ENTEROBACTER CLOACAE COMPLEX',
    'CITROBACTER AMALONATICUS', 'CITROBACTER YOUNGAE', 'SALMONELLA ENTERITIDIS',
    'SHIGELLA FLEXNERI', 'SALMONELLA HADAR', 'ESCHERICHIA FERGUSONII',
    'LECLERCIA ADECARBOXYLATA', 'RAOULTELLA ORNITHINOLYTICA', 'SALMONELLA DUBLIN'
]
SPECIMENS = [  # ????? seems not useful
    'BRONCHOALVEOLAR LAVAGE', 'SPUTUM', 'BLOOD CULTURE', 'URINE',
    'BLOOD CULTURE - NEONATE', 'SEROLOGY/BLOOD', 'EYE',
    'BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)', 'PLEURAL FLUID', 'SWAB',
    'PERITONEAL FLUID', 'CATHETER TIP-IV', 'CSF;SPINAL FLUID', 'FLUID,OTHER',
    'DIALYSIS FLUID', 'FLUID RECEIVED IN BLOOD CULTURE BOTTLES', 'ASPIRATE',
    'BRONCHIAL WASHINGS', 'ABSCESS', 'WORM', 'JOINT FLUID', 'FOOT CULTURE',
    'FOREIGN BODY', 'BILE', 'TRACHEAL ASPIRATE', 'POSTMORTEM CULTURE',
    'STOOL (RECEIVED IN TRANSPORT SYSTEM)', 'THROAT FOR STREP', 'SKIN SCRAPINGS',
    'BIOPSY', 'SWAB, R/O GC', 'BRONCHIAL BRUSH', 'THROAT CULTURE',
    'BRONCHIAL BRUSH - PROTECTED', 'FLUID WOUND', 'EAR', 'URINE,SUPRAPUBIC ASPIRATE',
    'URINE,KIDNEY', 'SWAB - R/O YEAST', 'BLOOD CULTURE (POST-MORTEM)', 'THROAT',
    'STERILITY CULTURE', 'CRE Screen', 'NOSE', 'FECAL SWAB', 'CORNEAL EYE SCRAPINGS',
    'TRANSTRACHEAL ASPIRATE', 'GASTRIC ASPIRATE', 'SCOTCH TAPE PREP/PADDLE',
    'RECTAL - R/O GC', 'NAIL SCRAPINGS', 'URINE,PROSTATIC MASSAGE', 'BLOOD',
    'ANORECTAL/VAGINAL CULTURE', 'Isolate', 'BLOOD BAG FLUID',
]
ADMISSION_COLUMNS = ['SUBJECT_ID', 'HADM_ID']
MICROB_COLUMNS = ['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'ORG_NAME', 'SPEC_TYPE_DESC']
TRANSFER_COLUMNS = [
    'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'PREV_CAREUNIT', 'CURR_CAREUNIT',
    'PREV_WARDID', 'CURR_WARDID','INTIME', 'OUTTIME', 'LOS',
]
MISSING_MERGE_COLUMNS = ['SUBJECT_ID', 'HADM_ID', 'COLONISED_DATE', 'SPEC_TYPE_DESC', 'ORG_NAME']
TIME_COLUMNS = ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME']
DIAGNOSIS_COLUMNS = ['SUBJECT_ID', 'HADM_ID', 'DIAGNOSIS']
GENDER_COLUMNS = ['SUBJECT_ID', 'GENDER']
NON_FEATURE_COLUMNS = [
    'DEATHTIME', 'ICUSTAY_ID', 'DIAGNOSIS', 'ORG_NAME', 'SPEC_TYPE_DESC',
    'DISCHTIME', 'COLONISED_DATE', 'CHARTDATE', 'INTIME', 'OUTTIME',
    'ADMITTIME', 'SUBJECT_ID', 'HADM_ID',  # 'EXPIRE_FLAG', <-- ????
]
CHARTEVENT_COLUMNS = ['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'CGID']
CAREGIVER_COLUMNS = ['SUBJECT_ID', 'HADM_ID', 'INTIME', 'OUTTIME', 'CURR_WARDID', 'CHARTTIME', 'CGID']
WARD_COLUMNS = ['SUBJECT_ID', 'HADM_ID', 'INTIME', 'OUTTIME', 'CURR_WARDID']
STRING_FEATURES = ['PREV_CAREUNIT', 'CURR_CAREUNIT', 'GENDER']
NUMERICAL_FEATURES = ['DIAG_ID', 'CURR_WARDID', 'PREV_WARDID', 'LOS', 'LOSH']
REGENERATE_DATASET_FROM_SCRATCH = True


def main():
    """ Build all datasets and link sets needed for GNN and control experiments
    """
    if REGENERATE_DATASET_FROM_SCRATCH:
        # Generate labels and basic features for all patients-wards
        # generate_patient_colonisation_labels()  # (62580 x 6)
        # print(pd.read_csv(PATH_COLONISATION_LABELS, index_col=[0]).shape)
        # generate_patient_wards()  # (261857 x 13)
        # print(pd.read_csv(PATH_PATIENT_WARDS, index_col=[0]).shape)
        # generate_diagnose_data()  # (261857 x 15)
        # print(pd.read_csv(PATH_DIAGNOSE_DATA, index_col=[0]).shape)
        
        # # Generate features for all patients (and keep label in same file)
        # initialize_data_with_diagnoses_and_labels()  # (281179 x 19)
        # print(pd.read_csv(PATH_COLUMNS_AND_LABELS, index_col=[0]).shape)
        # handle_missing_data()  # (274323 x 20)
        # print(pd.read_csv(PATH_COLUMNS_AND_LABELS, index_col=[0]).shape)
        # add_gender_to_data()  # (274323 x 21)
        # print(pd.read_csv(PATH_COLUMNS_AND_LABELS, index_col=[0]).shape)
        # # add_colonisation_pressure_to_data()  # (274323, 24) -> this step takes long (approx. 4 hours)
        # # print(pd.read_csv(PATH_COLUMNS_AND_LABELS, index_col=[0]).shape)
        # add_losh_to_data()  # (274323 x 25)
        # print(pd.read_csv(PATH_COLUMNS_AND_LABELS, index_col=[0]).shape)
        # generate_features_and_labels()  # (274323, 12) -> 267100 non-colonised, 7612 colonised (UPDATE THESE NUMBERS)
        # print(pd.read_csv(PATH_FEATURES_AND_LABELS, index_col=[0]).shape)

        # Generate graph links for graph-based algorithms
        # link_patient_wards_to_caregivers()  # long step!!
        generate_caregiver_links()  # (672656 x 2); approx. 3 minutes
        generate_ward_links()  # (184272 x 2); approx. 5 minutes
        merge_ward_and_caregiver_links()  # (1674559 x 2), i.e., [src, dest]
    
    # Save sets for balanced and non-balanced (training???) samples
    for balanced in ['non', 'under', 'over']:
        save_data_set_splits(balanced)
        load_data(balanced)  # check everything went good
        # X_train shape: (168758, 26), y_train shape: (168758,)
        # X_dev shape: (56253, 26), y_dev shape: (56253,)
        # X_test shape: (56253, 26), y_test shape: (56253,)
        # X_train shape: (8658, 26), y_train shape: (8658,)
        # X_dev shape: (2887, 26), y_dev shape: (2887,)
        # X_test shape: (2887, 26), y_test shape: (2887,)
        # X_train shape: (328857, 26), y_train shape: (328857,)
        # X_dev shape: (109619, 26), y_dev shape: (109619,)
        # X_test shape: (109620, 26), y_test shape: (109620,)


def generate_patient_colonisation_labels():
    """ Label patients based on whether they were colonised by selected organisms
    """
    print('Creating colonisation labels based on microbiolgy events')    
    # Find patients that were infected, if relevant organism + specimen detected
    df_microb = pd.read_csv(PATH_MICROBIOLOGY_EVENTS, usecols=MICROB_COLUMNS)
    df_microb = df_microb[df_microb['ORG_NAME'].isin(ENTEROBACTERIAE)]
    df_microb = df_microb[df_microb['SPEC_TYPE_DESC'].isin(SPECIMENS)]
    df_microb['COLONISED'] = int(1)
    df_microb = df_microb.rename(columns={'CHARTTIME': 'COLONISED_DATE'})
    
    # Find non-infected patients by loading patients and filtering colonised ones
    df_admissions = pd.read_csv(PATH_ADMISSIONS, usecols=ADMISSION_COLUMNS)
    unique_hadm_ids = df_microb['HADM_ID'].unique()
    no_colonisation_mask = ~df_admissions['HADM_ID'].isin(unique_hadm_ids)
    df_admissions = df_admissions[no_colonisation_mask]
    df_admissions['COLONISED'] = int(0)
    df_admissions['COLONISED_DATE'] = ''
    df_admissions['ORG_NAME'] = ''
    df_admissions['SPEC_TYPE_DESC'] = ''

    # Build final label collection
    df_final = pd.concat([df_microb, df_admissions])
    df_final = df_final.drop_duplicates()
    df_final.to_csv(PATH_COLONISATION_LABELS)


def generate_patient_wards():
    """ Link patients with time of ward entry/leave, ward id and transfer status
    """
    print('Collecting in-ward location(s) for each patients')
    df_times = pd.read_csv(PATH_ADMISSIONS, usecols=TIME_COLUMNS)
    df_transfers = pd.read_csv(PATH_TRANSFERS, usecols=TRANSFER_COLUMNS)
    df_final = pd.merge(df_transfers, df_times, how='left',
                        left_on=ADMISSION_COLUMNS,
                        right_on=ADMISSION_COLUMNS)
    df_final.sort_values(['HADM_ID', 'INTIME'], ascending=False)
    df_final.groupby('HADM_ID')
    df_final = df_final.drop_duplicates()
    df_final.to_csv(PATH_PATIENT_WARDS, encoding='utf-8')


def generate_diagnose_data():
    """ Generate patient-ward data as integers based on patient diagnoses
    """
    print('Link patients to integer colonisation labels')
    # Get patient and in-ward data
    df_diagnoses = pd.read_csv(PATH_ADMISSIONS, usecols=DIAGNOSIS_COLUMNS)
    df_patient_wards = pd.read_csv(PATH_PATIENT_WARDS, index_col=[0])
    df_final = pd.merge(df_patient_wards, df_diagnoses, how='left',
                        left_on=ADMISSION_COLUMNS,
                        right_on=ADMISSION_COLUMNS)
    
    # Encode all possible diagnoses as integers
    unique_diagnoses = df_final.DIAGNOSIS.unique()
    encoded = LabelEncoder().fit_transform(unique_diagnoses)
    mapping = {d: e for d, e in zip(unique_diagnoses, encoded)}
    df_final['DIAG_ID'] = df_final['DIAGNOSIS'].map(mapping)
    df_final = df_final.drop_duplicates()
    df_final.to_csv(PATH_DIAGNOSE_DATA, encoding='utf-8')
    

def initialize_data_with_diagnoses_and_labels():
    """ Add labels to patient data (useful for over / under sampling)
    """
    print('Initializing main data file with colonisation labels')
    df_labels = pd.read_csv(PATH_COLONISATION_LABELS, index_col=[0])
    df_data = pd.read_csv(PATH_DIAGNOSE_DATA, index_col=[0])
    df_final = pd.merge(df_labels, df_data, how='left',
                        left_on=ADMISSION_COLUMNS,
                        right_on=ADMISSION_COLUMNS)
    df_final = df_final.drop_duplicates()
    df_final.to_csv(PATH_COLUMNS_AND_LABELS, encoding='utf-8')


def handle_missing_data():
    """ Add chartdate, and replace non existing times by default ones
    """
    print('Handling missing data')
    # Load data and colonisation information
    df_data = pd.read_csv(PATH_COLUMNS_AND_LABELS, index_col=[0])
    cols_to_load = MICROB_COLUMNS + ['CHARTDATE']
    df_microb = pd.read_csv(PATH_MICROBIOLOGY_EVENTS, usecols=cols_to_load)
    df_microb = df_microb.rename(columns={'CHARTTIME': 'COLONISED_DATE'})

    # Replace non existing times by default ones
    df_final = pd.merge(df_data, df_microb, how='left',
                        left_on=MISSING_MERGE_COLUMNS,
                        right_on=MISSING_MERGE_COLUMNS)
    df_final.INTIME.fillna(df_final.ADMITTIME, inplace=True)
    df_final.OUTTIME.fillna(df_final.DISCHTIME, inplace=True)
    df_final.COLONISED_DATE.fillna(df_final.CHARTDATE, inplace=True)
    
    # Set idle values back to non-colonised patients as they were modified before (what???)
    df_final.loc[
        (df_final.COLONISED_DATE < df_final.INTIME) |
        (df_final.COLONISED_DATE > df_final.OUTTIME), 'COLONISED'] = int(0)
    df_final.loc[df_final['COLONISED'] == 0, 'COLONISED_DATE'] = ''
    df_final.loc[df_final['COLONISED'] == 0, 'ORG_NAME'] = ''
    df_final.loc[df_final['COLONISED'] == 0, 'SPEC_TYPE_DESC'] = ''
    df_final = df_final.drop_duplicates()
    df_final.to_csv(PATH_COLUMNS_AND_LABELS, encoding='utf-8')


def add_gender_to_data():
    """ Add patient gender to the patient data
    """
    print('Adding patient gender to the patient data')
    df_data = pd.read_csv(PATH_COLUMNS_AND_LABELS, index_col=[0])
    df_patients = pd.read_csv(PATH_PATIENTS, usecols=GENDER_COLUMNS)
    df_final = pd.merge(df_data, df_patients, on=['SUBJECT_ID'])
    df_final = df_final.drop_duplicates()
    df_final.to_csv(PATH_COLUMNS_AND_LABELS, encoding='utf-8')


def add_colonisation_pressure_to_data():
    """ Add colonisation pressure to the patient data
    """
    print('Adding colonisation pressure to the patient data')
    # Load already existing data
    df_data = pd.read_csv(PATH_COLUMNS_AND_LABELS, index_col=[0])
    for patient in tqdm(list(df_data.itertuples()),
                        desc='Computing colonisation pressure'):
        
        # Identify patients that were in the same room at the same time
        others = df_data.loc[
            (df_data['SUBJECT_ID'] != patient.SUBJECT_ID) &
            (df_data['HADM_ID'] != patient.HADM_ID)]
        others = others.loc[others['CURR_WARDID'] == patient.CURR_WARDID]
        others = others.loc[
            ((others['INTIME'] > patient.INTIME) & (others['INTIME'] < patient.OUTTIME)) |
            ((others['OUTTIME'] > patient.INTIME) & (others['OUTTIME'] < patient.OUTTIME))]
        
        # Compute colonisation pressure and this information to the patient data
        n_contacts = len(others.SUBJECT_ID.unique())
        n_colonised = len(others.loc[others['COLONISED'] == 1].SUBJECT_ID.unique())
        colonisation_pressure = n_colonised / n_contacts if n_contacts > 0 else 0
        df_data.at[patient[0], 'N_CONTACTS'] = n_contacts  # very inefficient???
        df_data.at[patient[0], 'N_COLONISED'] = n_colonised  # very inefficient???
        df_data.at[patient[0], 'CP'] = colonisation_pressure  # very inefficient???
    
    # Save updated dataset
    df_data = df_data.drop_duplicates()
    df_data.to_csv(PATH_COLUMNS_AND_LABELS, encoding='utf-8')
    

def add_losh_to_data():
    """ Add length-of-stay in hospital data, in days (difference between LOS and LOSH?)
    """
    print('Adding hospital length-of-stay to the patient data')
    df_data = pd.read_csv(PATH_COLUMNS_AND_LABELS, index_col=[0])
    df_data['DISCHTIME'] = pd.to_datetime(df_data['DISCHTIME'])
    df_data['ADMITTIME'] = pd.to_datetime(df_data['ADMITTIME'])
    df_data['LOSH'] = df_data['DISCHTIME'] - df_data['ADMITTIME']
    df_data['LOSH'] = df_data['LOSH'] / np.timedelta64(1, 'D')
    df_data = df_data.drop_duplicates()
    df_data.to_csv(PATH_COLUMNS_AND_LABELS, encoding='utf-8')


def generate_features_and_labels():
    """ Drop duplicates and features that were used only for computations
    """
    print('Generating features and label dataset')
    df_data = pd.read_csv(PATH_COLUMNS_AND_LABELS, index_col=[0])
    df_data = df_data.drop_duplicates()
    df_data = df_data.drop(NON_FEATURE_COLUMNS, axis=1)
    df_data.to_csv(PATH_FEATURES_AND_LABELS, encoding='utf-8')


def link_patient_wards_to_caregivers():
    """ Add links between patients that were visited by the same caregiver at the same time
    """
    print('Linking caregivers to patient-wards')
    # Load caregiver data
    import time
    t0 = time.time()
    df_caregivers = pd.read_csv(PATH_CHART_EVENTS, usecols=CHARTEVENT_COLUMNS) # takes approx. 8 minutes (330_712_483 x 7)
    df_caregivers['CHARTTIME'] = pd.to_datetime(df_caregivers['CHARTTIME'])
    df_caregivers['day'] = df_caregivers['CHARTTIME'].dt.day
    df_caregivers['month'] = df_caregivers['CHARTTIME'].dt.month
    df_caregivers['year'] = df_caregivers['CHARTTIME'].dt.year

    # Load patient-ward location data
    df_wards = pd.read_csv(PATH_PATIENT_WARDS, usecols=WARD_COLUMNS)  # takes a few seconds (261_857 x 5)
    df_wards['INTIME'] = pd.to_datetime(df_wards['INTIME'])
    df_wards['OUTTIME'] = pd.to_datetime(df_wards['OUTTIME'])

    # Link caregivers to patient-wards
    t1 = time.time()
    print(t1 - t0, df_caregivers.shape, df_wards.shape)
    df_final = pd.merge(df_caregivers, df_wards, how='left',
                        left_on=ADMISSION_COLUMNS, 
                        right_on=ADMISSION_COLUMNS)  # takes approx 1.8 hours (1_664_662_896 x 10)
    t2 = time.time()
    print(t2 - t1, df_final.shape)
    df_final = df_final[(df_final['CHARTTIME'] >= df_final['INTIME']) &
                        (df_final['CHARTTIME'] <= df_final['OUTTIME'])]  # takes approx. 2 hours (329366946 x 10)
    t3 = time.time()
    print(t3 - t2, df_final.shape)
    df_final = df_final.drop_duplicates()  # ????
    df_final.to_csv(PATH_PATIENT_WARD_CAREGIVER_MAPPING, encoding='utf-8')
    print(time.time() - t3, df_final.shape)
    exit()
    

def generate_caregiver_links():
    """ Generate links between patients that had the same caregiver at the same time
    """
    print('Generating graph links using patient-caregiver data')
    # Load patient-ward and caregiver data and merge them in a single dataframe
    df_data = pd.read_csv(PATH_COLUMNS_AND_LABELS, usecols=WARD_COLUMNS)
    df_data['PWARD_ID'] = df_data.index  # to keep track of row ids
    df_ward_caregiver = pd.read_csv(PATH_PATIENT_WARD_CAREGIVER_MAPPING,
                                    usecols=CAREGIVER_COLUMNS)
    df_patients = pd.merge(df_ward_caregiver, df_data, how='left',  # 18M samples
                           left_on=WARD_COLUMNS, right_on=WARD_COLUMNS)
    
    # Group patients by caregiver ids, only if visited on the same day
    df_patients['CHARTTIME'] = pd.to_datetime(df_patients['CHARTTIME'])
    df_patients['CHARTDATE'] = df_patients['CHARTTIME'].dt.date  # only day
    used_columns = ['SUBJECT_ID', 'HADM_ID', 'PWARD_ID', 'CHARTDATE', 'CGID']
    df_patients = df_patients.filter(items=used_columns)
    df_patients = df_patients.drop_duplicates()  # 250k samples -> faster!
    df_grouped = df_patients.groupby(['CHARTDATE', 'CGID'])  # 1 minute
    
    # Add links between patients visited by the same caregiver on the same day
    links = set()
    for _, group in tqdm(df_grouped, desc='Computing links'):
        src_patient_wards = group['PWARD_ID'].unique().tolist()
        for src_id in src_patient_wards:
            tgt_patient_wards = [i for i in src_patient_wards if i != src_id]
            for tgt_id in tgt_patient_wards:
                links.add(frozenset([src_id, tgt_id]))  # unordered
    # patient_ward_ids = df_data['PWARD_ID'].to_list()  # ????????? mais src_id != pward_id non????
    # admission_ids = df_data['HADM_ID'].to_list()
    # patient_dict = dict(zip(patient_ward_ids, admission_ids))  # useful?
                # if patient_dict[src_id] != patient_dict[tgt_id]:  # why check again?????
            
    # Generate data file containing caregiver links
    import pdb; pdb.set_trace()
    link_dict = {'src': [list(link)[0] for link in links],
                 'dst': [list(link)[1] for link in links]}
    df_caregiver_links = pd.DataFrame.from_dict(link_dict)
    df_caregiver_links.to_csv(PATH_CAREGIVER_LINKS)


def generate_ward_links():
    """ Generate links between patients that visited the same ward at the same time
    """
    print('Generating graph links using in-ward patient data')
    # Load patient-ward data
    df_data = pd.read_csv(PATH_COLUMNS_AND_LABELS, usecols=WARD_COLUMNS)
    df_grouped = df_data.groupby('CURR_WARDID')

    # Add connections between parients that visited the same room at the same time
    links = set()
    for group_name, group in tqdm(df_grouped, desc='Computing links'):
        for row in tqdm(group.itertuples(), desc=' - Group %s' % group_name,
                        leave=False, total=len(group)):
            contact_patients = group[(group['INTIME'] < row.OUTTIME) &
                                     (group['OUTTIME'] > row.INTIME) &
                                     (group.index != row.Index)]
            for contact_patient in contact_patients.index:
                links.add(frozenset([row.Index, contact_patient]))  # unordered

    # Generate data file containing ward links
    link_dict = {'src': [list(link)[0] for link in links],
                 'dst': [list(link)[1] for link in links]}
    df_ward_links = pd.DataFrame.from_dict(link_dict)
    df_ward_links.to_csv(PATH_WARD_LINKS)
    

def merge_ward_and_caregiver_links():
    """ Merge links coming from having a common ward, and having common caregivers
    """
    print('Generating a new set of links combining in-ward and caregiver links')
    df_ward_links = pd.read_csv(PATH_WARD_LINKS, index_col=[0])
    df_caregiver_links = pd.read_csv(PATH_CAREGIVER_LINKS, index_col=[0])
    df_all = pd.concat([df_ward_links, df_caregiver_links], ignore_index=True)
    df_all = df_all.drop_duplicates()
    df_all.to_csv(PATH_ALL_LINKS)
    

def save_data_set_splits(balanced='non'):
    """ Save data set splits from processed dataset file
    """
    print('Saving data set splits using main data file')
    # Get data and balance it by outcome label if required (undersampling)
    df_data = pd.read_csv(PATH_FEATURES_AND_LABELS, index_col=[0])
    if balanced == 'under':
        df_minor = df_data[df_data['COLONISED'] == 1]
        df_major = df_data[df_data['COLONISED'] == 0]
        df_major = df_major.sample(n=len(df_minor))
        df_data = pd.concat([df_minor, df_major])
    if balanced == 'over':
        df_minor = df_data[df_data['COLONISED'] == 1]
        df_major = df_data[df_data['COLONISED'] == 0]
        df_minor = df_minor.sample(n=len(df_major), replace=True)
        df_data = pd.concat([df_major, df_minor])
    
    # Separate input features and labels
    df_y = df_data['COLONISED']
    df_X = df_data.drop('COLONISED', axis=1)
    
    # One-hotize features that are strings
    for feat in STRING_FEATURES:
        df_X[feat] = pd.Categorical(df_X[feat])
        one_hot_features = pd.get_dummies(df_X[feat], prefix=feat)
        df_X = pd.concat([df_X, one_hot_features], axis=1)
        df_X = df_X.drop(feat, axis=1)

    # Standardize features that are numerical
    for feat in NUMERICAL_FEATURES:
        df_X[feat] = (df_X[feat] - df_X[feat].mean()) / df_X[feat].std()
    df_X.replace(np.nan, 0, inplace=True)  # fillna????

    # Create data splits
    balanced_dir = os.path.join(PROCESSED_DATA_DIR, '%s_balanced' % balanced)
    os.makedirs(balanced_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(
        df_X, df_y, test_size=0.2, random_state=2, shuffle=True)
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train, y_train, test_size=0.25, random_state=2, shuffle=True)
    X_train.to_pickle(os.path.join(balanced_dir, 'X_train.pkl'))
    X_dev.to_pickle(os.path.join(balanced_dir, 'X_dev.pkl'))
    X_test.to_pickle(os.path.join(balanced_dir, 'X_test.pkl'))
    y_train.to_pickle(os.path.join(balanced_dir, 'y_train.pkl'))
    y_dev.to_pickle(os.path.join(balanced_dir, 'y_dev.pkl'))
    y_test.to_pickle(os.path.join(balanced_dir, 'y_test.pkl'))


def load_data(balanced='non'):
    """ Get data set splits (separate features and labels)
    """
    print('Loading dataset splits in %s-balanced mode' % balanced)
    # Load data feature splits
    balanced_dir = os.path.join(PROCESSED_DATA_DIR, '%s_balanced' % balanced)
    X_train = pd.read_pickle(os.path.join(balanced_dir, 'X_train.pkl'))
    X_dev = pd.read_pickle(os.path.join(balanced_dir, 'X_dev.pkl'))
    X_test = pd.read_pickle(os.path.join(balanced_dir, 'X_test.pkl'))

    # TEMPORARY UNTIL I GENERATE THE DATASETS AGAIN
    X_train = X_train.drop(['Number_colonised', 'Number_patients'], axis=1)
    X_dev = X_dev.drop(['Number_colonised', 'Number_patients'], axis=1)
    X_test = X_test.drop(['Number_colonised', 'Number_patients'], axis=1)
    # TEMPORARY UNTIL I GENERATE THE DATASETS AGAIN

    # Scale features
    scaler = RobustScaler().fit(X_train)  # pd.DataFrame -> np.ndarray
    X_train = scaler.transform(X_train)
    X_dev = scaler.transform(X_dev)
    X_test = scaler.transform(X_test)

    # Load labels
    y_train = pd.read_pickle(os.path.join(balanced_dir, 'y_train.pkl'))
    y_dev = pd.read_pickle(os.path.join(balanced_dir, 'y_dev.pkl'))
    y_test = pd.read_pickle(os.path.join(balanced_dir, 'y_test.pkl'))

    # Return features and labels in separate objects
    print('Loaded successfully!')
    print(' - X_train shape: %s, y_train shape: %s' % (X_train.shape, y_train.shape))
    print(' - X_dev shape: %s, y_dev shape: %s' % (X_dev.shape, y_dev.shape))
    print(' - X_test shape: %s, y_test shape: %s' % (X_test.shape, y_test.shape))
    return X_train, X_dev, X_test, y_train, y_dev, y_test


if __name__ == '__main__':
    main()
    