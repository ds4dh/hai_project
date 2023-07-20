import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Helper function
def ABS_JOIN(*args):
    return os.path.abspath(os.path.join(*args))

# Input/output file paths
PATH_ADMISSIONS = 'data/physionet.org/files/mimiciii/1.4/ADMISSIONS.csv.gz'
PATH_PATIENTS = 'data/physionet.org/files/mimiciii/1.4/PATIENTS.csv.gz'
PATH_MICROBIOLOGY_EVENTS = 'data/physionet.org/files/mimiciii/1.4/MICROBIOLOGYEVENTS.csv.gz'
PATH_PATIENT_WARDS_WITH_LABELS = 'data/processed/patient-ward_colonisation_labels.csv'
PATH_COLUMNS_AND_LABELS = 'data/processed/patient-ward_columns_and_labels.csv'
OUTPUT_PATH = 'data/processed/patient-ward_info.csv'

# MDR and Antibiotics
MDR_FIELDS = ['SUBJECT_ID', 'HADM_ID', 'COLONISED_DATE', 'MDR_STATUS']
MDR_MERGE_FIELDS = ['SUBJECT_ID', 'HADM_ID', 'COLONISED_DATE']
ANTIBIOTIC_FIELDS = ['SUBJECT_ID', 'HADM_ID', 'SPEC_TYPE_DESC', 'ORG_NAME', 'AB_NAME', 'INTERPRETATION']
ANTIBIOTIC_MERGE_FIELDS = ['SUBJECT_ID', 'HADM_ID', 'SPEC_TYPE_DESC', 'ORG_NAME']
ANTIBIOTIC_TYPE_SPECIES = {
    'PENICILLINS': ['PIPERACILLIN/TAZO', 'AMPICILLIN/SULBACTAM', 'AMPICILLIN', 'PIPERACILLIN'],
    'CEPHALOSPORINS': ['CEFAZOLIN', 'CEFEPIME', 'CEFTRIAXONE', 'CEFUROXIME', 'CEFTAZIDIME', 'CEFPODOXIME'],
    'CARBAPENEMS': ['IMIPENEM', 'MEROPENEM'],
    'AMINOGLYCOSIDES': ['TOBRAMYCIN', 'GENTAMICIN', 'AMIKACIN'],
    'FLUOROQUINOLONES': ['CIPROFLOXACIN', 'LEVOFLOXACIN'],
    'TETRACYCLINES': ['TETRACYCLINE'],
    'OTHERS': ['TRIMETHOPRIM/SULFA', 'NITROFURANTOIN'],
}


def main():
    """ Compute MDR status for all patient-wards
    """
    # Read all relevant data files
    admin = pd.read_csv(PATH_ADMISSIONS, index_col=[0])
    patients = pd.read_csv(PATH_PATIENTS, index_col=[0])
    microbiology = pd.read_csv(PATH_MICROBIOLOGY_EVENTS, index_col=[0])
    labels = pd.read_csv(PATH_PATIENT_WARDS_WITH_LABELS, index_col=[0])
    labels['INDEX_COLUMN'] = labels.index
    
    # Merge data files into a single dataframe that will hold mdr status
    df_mdr = admin.groupby('SUBJECT_ID')['ADMITTIME'].min()
    df_mdr = df_mdr.to_frame()
    patients = patients.filter(items=['SUBJECT_ID', 'DOB'])
    df_mdr = df_mdr.merge(patients, how='left', on='SUBJECT_ID')
    df_mdr = labels.merge(df_mdr, how='left', on='SUBJECT_ID')

    # Add antibiotics information
    microbiology = microbiology.filter(items=ANTIBIOTIC_FIELDS)
    df_antibiotics = df_mdr.merge(
        microbiology,
        how='left',
        on=ANTIBIOTIC_MERGE_FIELDS,
    )
    
    # Initialize data fields
    df_mdr['BACTERIA_TYPE'] = ''
    df_mdr['RESISTANCE'] = 0
    df_mdr['MDR_STATUS'] = 'None'
    df_mdr.loc[df_mdr['COLONISED'] == 0, ['BACTERIA_TYPE']] = 'NON_COLONISED'
    
    # Add MDR status by updating the output file
    df_antibiotics = df_antibiotics[df_antibiotics['COLONISED'] == 1]
    for name, group in tqdm(df_antibiotics.groupby('INDEX_COLUMN')):
        group = group[group['INTERPRETATION'] == 'R']
        df_mdr = compute_resistance_profile(df_mdr, name, group)
    
    # Save output file
    df_mdr.drop('INDEX_COLUMN', axis=1)
    df_mdr_linked = link_to_dataset(df_mdr)
    df_mdr_linked.to_csv(OUTPUT_PATH, encoding='utf-8')
    

def compute_resistance_profile(result, name, group):
    """ Count resistance to given antibiotic types for any bacterial infection
    """
    # Count how many antibiotic types were given to this patient ward
    antibiotic_list = group['AB_NAME'].unique()
    resistance_count = 0
    for ab_type, ab_type_species in ANTIBIOTIC_TYPE_SPECIES.items():
        if (any(item in antibiotic_list for item in ab_type_species)):
            resistance_count = resistance_count + 1
    
    # Update MDR status and return
    result.loc[result['INDEX_COLUMN'] == name, ['RESISTANCE']] = resistance_count
    if resistance_count == 0:
        result.loc[result['INDEX_COLUMN'] == name, ['BACTERIA_TYPE']] = 'NDR'
        result.loc[result['INDEX_COLUMN'] == name, ['MDR_STATUS']] = 'AMS'
    if resistance_count > 0:
        result.loc[result['INDEX_COLUMN'] == name, ['BACTERIA_TYPE']] = 'DR'
    if 0 < resistance_count < 3: 
        result.loc[result['INDEX_COLUMN'] == name, ['MDR_STATUS']] = 'AMR'
    if resistance_count >= 3:
        result.loc[result['INDEX_COLUMN'] == name, ['MDR_STATUS']] = 'MDR'  # 1
    return result


def link_to_dataset(df_mdr: pd.DataFrame) -> pd.DataFrame:
    """ Link the built dataframe to patient-ward ids of the pre-processed dataset
        and save final dataframe
    """
    df_data = pd.read_csv(PATH_COLUMNS_AND_LABELS, index_col=[0])
    df_mdr_unique = df_mdr.drop_duplicates(subset=MDR_MERGE_FIELDS)[MDR_FIELDS]
    df_linked = pd.merge(df_data, df_mdr_unique, on=MDR_MERGE_FIELDS, how='left')
    cond = (df_linked['COLONISED'] == 0) | (df_linked['MDR_STATUS'].isna())
    df_linked['MDR_STATUS'] = np.where(cond, 'None', df_linked['MDR_STATUS'])
    return df_linked


if __name__ == '__main__':
    main()
    