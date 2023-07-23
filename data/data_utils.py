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

# Helper function
ABS_JOIN = lambda *args: os.path.abspath(os.path.join(*args))

# Input file paths
PATH_ICU_STAYS = 'data/physionet.org/files/mimiciii/1.4/ICUSTAYS.csv.gz'
PATH_ADMISSIONS = 'data/physionet.org/files/mimiciii/1.4/ADMISSIONS.csv.gz'
PATH_PATIENTS = 'data/physionet.org/files/mimiciii/1.4/PATIENTS.csv.gz'
PATH_TRANSFERS = 'data/physionet.org/files/mimiciii/1.4/TRANSFERS.csv.gz'
PATH_MICROBIOLOGY_EVENTS = 'data/physionet.org/files/mimiciii/1.4/MICROBIOLOGYEVENTS.csv.gz'

# Output file paths (features)
PROCESSED_DATA_DIR = ABS_JOIN('data', 'processed')
PATH_PATIENT_WARDS = ABS_JOIN(PROCESSED_DATA_DIR, 'patient-wards.csv')
PATH_COLONISATION_LABELS = ABS_JOIN(PROCESSED_DATA_DIR, 'patient-ward_colonisation_labels.csv')
PATH_DIAGNOSE_DATA = ABS_JOIN(PROCESSED_DATA_DIR, 'patient-ward_diagnose_data.csv')
PATH_COLUMNS_AND_LABELS = ABS_JOIN(PROCESSED_DATA_DIR, 'patient-ward_columns_and_labels.csv')
PATH_FEATURES_AND_LABELS = ABS_JOIN(PROCESSED_DATA_DIR, 'patient-ward_features_and_labels.csv')

# Relevant column names in various data files
ENTEROBACTERIAE = [
    'KLEBSIELLA PNEUMONIAE', 'ESCHERICHIA COLI', 'ENTEROBACTER CLOACAE',
    'KLEBSIELLA OXYTOCA', 'CITROBACTER KOSERI', 'CITROBACTER FREUNDII COMPLEX',
    'ENTEROBACTER ASBURIAE', 'ENTEROBACTER CLOACAE COMPLEX',
    'CITROBACTER AMALONATICUS', 'CITROBACTER YOUNGAE', 'SALMONELLA ENTERITIDIS',
    'SHIGELLA FLEXNERI', 'SALMONELLA HADAR', 'ESCHERICHIA FERGUSONII',
    'LECLERCIA ADECARBOXYLATA', 'RAOULTELLA ORNITHINOLYTICA', 'SALMONELLA DUBLIN'
]
SPECIMENS = [
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
STRING_FEATURES = ['PREV_CAREUNIT', 'CURR_CAREUNIT', 'GENDER']
REGENERATE_DATASET_FROM_SCRATCH = False


def main():
    """ Build all datasets and link sets needed for GNN and control experiments
    """
    if REGENERATE_DATASET_FROM_SCRATCH:
        # Generate labels and basic features for all patients-wards
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        generate_patient_colonisation_labels()  # (62580 x 6)
        generate_patient_wards()  # (261_857 x 13)
        generate_diagnose_data()  # (261_857 x 15)
        
        # Generate features for all patients (and keep label in same file)
        initialize_data_with_diagnoses_and_labels()  # (281_179 x 19)
        handle_missing_data()  # (274_323 x 20)
        add_gender_to_data()  # (274_323 x 21)
        add_colonisation_pressure_to_data()  # (274_323, 24); approx. 4 hours
        add_losh_to_data()  # (274_323 x 25)
    
    # DEBUG RETAB THIS TO IF LOOP
    generate_features_and_labels()  # (274_323, 12) -> 267_106 non-colonised, 7_617 colonised
        
    # Save different data splittings for different balanced scenarios
    for balanced_cond in ['under', 'over', 'non']:
        if 1:  # REGENERATE_DATASET_FROM_SCRATCH:
            save_data_splits(balanced_cond)
        load_features_and_labels(balanced_cond)


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
    unique_diagnoses = df_final['DIAGNOSIS'].unique()
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
    
    # Set back idle values for non-colonised patients as they were modified (what???)
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
    for pw in tqdm(list(df_data.itertuples()),
                        desc='Computing colonisation pressure'):
        
        # Identify patient-wards that that occured at the same time
        pws = df_data.loc[
            (df_data['SUBJECT_ID'] != pw.SUBJECT_ID) &
            (df_data['HADM_ID'] != pw.HADM_ID)]
        pws = pws.loc[pws['CURR_WARDID'] == pw.CURR_WARDID]
        pws = pws.loc[
            ((pws['INTIME'] > pw.INTIME) & (pws['INTIME'] < pw.OUTTIME)) |
            ((pws['OUTTIME'] > pw.INTIME) & (pws['OUTTIME'] < pw.OUTTIME))]
        
        # Compute colonisation pressure and add features to patient-wards data
        n_contacts = len(pws.SUBJECT_ID.unique())
        n_colonised = len(pws.loc[pws['COLONISED'] == 1].SUBJECT_ID.unique())
        col_pressure = n_colonised / n_contacts if n_contacts > 0 else 0
        df_data.at[pw[0], 'N_CONTACTS'] = n_contacts  # very inefficient??? here append, then set all in a column
        df_data.at[pw[0], 'N_COLONISED'] = n_colonised  # very inefficient??? here append, then set all in a column
        df_data.at[pw[0], 'CP'] = col_pressure  # very inefficient??? here append, then set all in a column
    
    # Save updated dataset
    df_data = df_data.drop_duplicates()
    df_data.to_csv(PATH_COLUMNS_AND_LABELS, encoding='utf-8')
    

def add_losh_to_data():
    """ Add length-of-stay in hospital data, in days (difference between LOS and LOSH?)
    """
    print('Adding hospital length-of-stay to the patient data')
    df_data = pd.read_csv(PATH_COLUMNS_AND_LABELS, index_col=[0])
    df_data['DISCHTIME'] = pd.to_datetime(df_data['DISCHTIME'])  # what????? should not modify df
    df_data['ADMITTIME'] = pd.to_datetime(df_data['ADMITTIME'])  # what????? should not modify df
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
    df_features = df_data.drop(NON_FEATURE_COLUMNS, axis=1)
    df_features.to_csv(PATH_FEATURES_AND_LABELS, encoding='utf-8')


def save_data_splits(balanced='non'):
    """ Save data set splits from processed dataset file
    """
    print('Saving data set splits using main data file')
    # Get data features and labels and keep node ids for graph models
    df_data = pd.read_csv(PATH_FEATURES_AND_LABELS, index_col=[0])
    df_y = df_data['COLONISED']
    df_X = df_data.drop(['COLONISED'], axis=1)
    
    # One-hotize string features and handle missing numerical values 
    for feat in STRING_FEATURES:
        df_X[feat] = pd.Categorical(df_X[feat])
        one_hot_features = pd.get_dummies(df_X[feat], prefix=feat)
        df_X = pd.concat([df_X, one_hot_features], axis=1)
        df_X = df_X.drop(feat, axis=1)
    df_X.fillna(0, inplace=True)
    
    # Create data splits (-> shuffle samples, hence linked node ids and labels)
    balanced_dir = ABS_JOIN(PROCESSED_DATA_DIR, '%s_balanced' % balanced)
    os.makedirs(balanced_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(
        df_X, df_y, test_size=0.2, random_state=2, shuffle=True)
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train, y_train, test_size=0.25, random_state=2, shuffle=True)
    
    # Over-sample or under-sample training data if required
    if balanced in ['under', 'over']:
        X_minor = X_train[y_train == 1]; y_minor = y_train[y_train == 1]
        X_major = X_train[y_train == 0]; y_major = y_train[y_train == 0]
        if balanced == 'under':
            X_major = X_major.sample(n=len(X_minor))
            y_major = y_major[X_major.index]  # corresponding labels
        if balanced == 'over':
            X_minor = X_minor.sample(n=len(X_major), replace=True)
            y_minor = y_minor[X_minor.index]  # corresponding labels
        X_train = pd.concat([X_major, X_minor])
        y_train = pd.concat([y_major, y_minor])
                    
    # Save features (as numpy.array)
    X_train.to_pickle(ABS_JOIN(balanced_dir, 'X_train.pkl'))
    X_dev.to_pickle(ABS_JOIN(balanced_dir, 'X_dev.pkl'))
    X_test.to_pickle(ABS_JOIN(balanced_dir, 'X_test.pkl'))
    
    # Save labels (as dataframes, to keep trak of node ids)
    y_train.to_pickle(ABS_JOIN(balanced_dir, 'y_train.pkl'))
    y_dev.to_pickle(ABS_JOIN(balanced_dir, 'y_dev.pkl'))
    y_test.to_pickle(ABS_JOIN(balanced_dir, 'y_test.pkl'))
    
    
def load_features_and_labels(balanced='non'):
    """ Get data set splits (separate features and labels)
    """
    # Load data feature splits
    balanced_dir = ABS_JOIN(PROCESSED_DATA_DIR, '%s_balanced' % balanced)
    X_train = pd.read_pickle(ABS_JOIN(balanced_dir, 'X_train.pkl'))
    X_dev = pd.read_pickle(ABS_JOIN(balanced_dir, 'X_dev.pkl'))
    X_test = pd.read_pickle(ABS_JOIN(balanced_dir, 'X_test.pkl'))
    
    # Scale features (fitting scaler only with training data)
    scaler = RobustScaler().fit(X_train)  # pd.DataFrame -> np.ndarray
    X_train = scaler.transform(X_train)
    X_dev = scaler.transform(X_dev)
    X_test = scaler.transform(X_test)
    
    # Load labels
    y_train = pd.read_pickle(ABS_JOIN(balanced_dir, 'y_train.pkl'))
    y_dev = pd.read_pickle(ABS_JOIN(balanced_dir, 'y_dev.pkl'))
    y_test = pd.read_pickle(ABS_JOIN(balanced_dir, 'y_test.pkl'))

    # Retrieve node indices
    id_train, id_dev, id_test = y_train.index, y_dev.index, y_test.index

    # Message for debug purpose
    if __name__ == '__main__':
        print('Loaded %s-balanced features and labels successfully!' % balanced)
        print(' - X_train: %s, y_train: %s' % (X_train.shape, y_train.shape))
        print(' - X_dev: %s, y_dev: %s' % (X_dev.shape, y_dev.shape))
        print(' - X_test: %s, y_test: %s' % (X_test.shape, y_test.shape))
    
    # Return data alltogether
    X_data = {'train': X_train, 'dev': X_dev, 'test': X_test}
    y_data = {'train': y_train, 'dev': y_dev, 'test': y_test}
    id_data = {'train': id_train, 'dev': id_dev, 'test': id_test}
    return X_data, y_data, id_data


if __name__ == '__main__':
    main()
    