from math import isnan
import numpy as np
import pandas as pd

from typing import Tuple, List

"""
Helper methods based on ICD9-CM Diagnosis Codes 
"""

def icd9_to_category(code: str) -> str:
    if code is np.nan or code is None or (isinstance(code, float) and isnan(code)):
        return 'Not Available'
    
    try:
        icd9_code = float(code)
    except ValueError:
        return 'Other'

    icd9_mapping = {
        (1., 139.): 'Other',
        (140., 239.): 'Neoplasms',
        (240., 249.): 'Endocrine',
        (250., 250.99): 'Diabetes',
        (251., 279.): 'Endocrine',
        (280., 289.): 'Other',
        (290., 319.): 'Other',
        (320., 389.): 'Other',
        (390., 459.): 'Circulatory',
        (785., 785.99): 'Circulatory',
        (460., 519.): 'Respiratory',
        (786., 786.99): 'Respiratory',
        (520., 579.): 'Digestive',
        (787., 787.99): 'Digestive',
        (580., 629.): 'Genitourinary',
        (788., 788.99): 'Genitourinary',
        (630., 676.): 'Other',
        (680., 709.): 'Other',
        (710., 739.): 'Musculoskeletal',
        (740., 759.): 'Other',
        (760., 779.): 'Other',
        (780., 784.99): 'Other',
        (789., 799.): 'Other',
        (800., 999.): 'Injury'
    }

    for (l, u), classification in icd9_mapping.items():
        if icd9_code >= l and icd9_code <= u:
            return classification

    return 'Other'
        

def icd9_to_classification(code: str, concise:bool=True) -> str:
    if code is np.nan or code is None or (isinstance(code, float) and isnan(code)):
        return 'Not Available'

    circulatory = 'Circulatory' if concise else 'Diseases of the circulatory system'
    respiratory = 'Respiratory' if concise else 'Diseases of the respiratory system'
    digestive = 'Digestive' if concise else 'Diseases of the digestive system'
    genitourinary = 'Genitourinary' if concise else 'Diseases of the genitourinary system'
    ill_defined = 'Ill-defined' if concise else 'Symptoms, signs and ill-defined conditions'
    injury = 'Injury/Poison' if concise else 'Injury and poisoning'

    try:
        icd9_code = float(code)
    except ValueError:
        if code[0] == 'E':
            return f'(E) {injury}' if concise else 'SC of external causes of injury and poisoning'
        elif code[0] == 'V':
            return f'(V) Health contact' if concise else 'SC of factors influencing health status and contact with health services'
        else:
            return 'Other'
    icd9_mapping = {
        (1., 139.): 'Infectious/Parasitic' if concise else 'Infectious and parasitic diseases',
        (140., 239.): 'Neoplasms',
        (240., 279.): 'Endocrine' if concise else 'Endocrine, nutritional, and metabolic diseases and immunity disorders',
        (280., 289.): 'Blood' if concise else 'Diseases of blood and blood-forming organs',
        (290., 319.): 'Mental' if concise else 'Mental disorders',
        (320., 389.): 'Nervous' if concise else 'Diseases of the nervous system and sense organs',
        (390., 459.): circulatory,
        (785., 785.99): circulatory,
        (460., 519.): respiratory,
        (786., 786.99): respiratory,
        (520., 579.): digestive,
        (787., 787.99): digestive,
        (580., 629.): genitourinary,
        (788., 788.99): genitourinary,
        (630., 676.): 'Pregnancy' if concise else 'Complications of pregnancy, childbirth and the puerperium',
        (680., 709.): 'Skin' if concise else 'Diseases of the skin and subcutaneous tissue',
        (710., 739.): 'Musculoskeletal' if concise else 'Diseases of the musculoskeletal system and connective tissue',
        (740., 759.): 'Congenital' if concise else 'Congenital anomalies',
        (760., 779.): 'Perinatal' if concise else 'Certain conditions originating in the perinatal period',
        (780., 784.99): ill_defined,
        (789., 799.): ill_defined,
        (800., 999.): injury
    }

    if icd9_code < 0.:
        return 'Not Available'

    for (l, u), classification in icd9_mapping.items():
        if icd9_code >= l and icd9_code <= u:
            return classification

    return 'Other'

def is_diabetes_mellitus(code: str) -> bool:
    try:
        icd9_code = float(code)
        return icd9_code >= 250. and icd9_code < 251.
    except ValueError:
        return False

def has_diabetes_diagnosis(*diagnosis_codes: tuple) -> bool:
    return sum(list(map(is_diabetes_mellitus, diagnosis_codes))) > 0


MYOCARDIAL_INFARCTION_CODES = ('410', '412')
CONGESTIVE_HEART_FAILURE_CODES = ('428', '398.91', '402.01', '402.11', '402.91', '404.01', '404.03', '404.11', '404.13', '404.91', '404.93', '425.4', '425.5', '425.6', '425.7', '425.9')
PERIPHERAL_VASCULAR_DISEASE_CODES = ('093.0', '437.3', '443.1', '443.9', '447.1', '557.1', '557.9', '443.2', '443.8', 'V43.4', '440', '441')
CEREBROVASCULAR_DISEASE_CODES = ('362.34', '430', '431', '432', '433', '434', '435', '436', '437', '438')
DEMENTIA_CODES = ('290.0', '290.3', '294.0', '294.8', '331.0', '331.2', '331.7', '797', '290.1', '290.2', '290.4', '294.1', '294.2', '331.1')
CHRONIC_PULMONARY_DISEASE_CODES = ('506.4', '508.1', '508.8', '490', '491', '492', '493', '494', '495', '496', '500', '501', '502', '503', '504', '505')
RHEUMATIC_DISEASE_CODES = ('446.5', '710.0', '710.1', '710.2', '710.3', '710.4', '714.0', '714.1', '714.2', '714.8', '725')
PEPTIC_ULCER_DISEASE_CODES = ('531', '532', '533', '534')
MILD_LIVER_DISEASE_CODES = ('070.22', '070.23', '070.32', '070.33', '070.44', '070.54', '070.6', '070.9', '573.3', '573.4', '573.8', '573.9', '570', '571', 'V42.7')
DIABETES_WITHOUT_COMPLICATIONS_CODES = ('250.8', '250.9', '249.0', '249.1', '249.2', '249.3', '249.9')
MILD_RENAL_DISEASE_CODES = ('403.00', '403.10', '403.90', '404.00', '404.01', '404.10', '404.11', '404.90', '404.91', '585.1', '585.2', '585.3', '585.4', '585.9', 'V42.0', '582', '583')
DIABETES_WITH_CHRONIC_COMPLICATONS_CODES = ('250.4', '250.5', '250.6', '250.7')
HEMIPLEGIA_OR_PARAPLEGIA_CODES = ('342', '343', '344', '334.1')
MALIGNANCY_CODES = (
    '140', '141', '142', '143', '144', '145', '146', '147', '148', '149',
    '150', '151', '152', '153', '154', '155', '156', '157', '158', '159',
    '160', '161', '162', '163', '164', '165', '166', '167', '168', '169',
    '170', '171', '172', '174', '175', '176', '179',
    '180', '181', '182', '183', '184', '185', '186', '187', '188', '189',
    '190', '191', '192', '193', '194', '195', '199.1',
    '200', '201', '202', '203', '204', '205', '206', '207', '208', '238.6'
)
SEVERE_LIVER_DISEASE_CODES = ('456.0', '456.1', '572.2', '572.3', '572.4', '572.8', '456.2')
SEVERE_RENAL_DISEASE_CODES = ('403.01', '403.11', '403.11', '403.91', '404.02', '404.03', '404.12', '404.13', '404.92', '404.93', '585.5', '585.6', '588.0', 'V45.11', 'V45.12', 'V56.0', 'V56.1', 'V56.2', 'V56.31', 'V56.32', 'V56.8', '586')
HIV_CODES = ('042')
METASTATIC_SOLID_TUMOR_CODES = ('199.0', '196', '197', '198')
AIDS_CODES = ('117.5', '078.5', '007.2', '136.3', 'V12.61', '046.3', '003.1', '799.4', '112', '180', '114', '348.3', '054', '115', '176', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '031', '010', '011', '012', '013', '014', '015', '016', '017', '018', '130')


def diagnosis_startswith(df: pd.DataFrame, codes: Tuple[str]) -> pd.Series:
    return ((df.diag_1.str.startswith(codes)) | (df.diag_2.str.startswith(codes)) | (df.diag_3.str.startswith(codes)))

def generate_comorbidity_index_predicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates comorbidity-related features based on the Charlson Comorbidity Index (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6684052/)
    """
    df_copy = df.copy()
    df_copy['has_myocardial_infarction'] = diagnosis_startswith(df_copy, MYOCARDIAL_INFARCTION_CODES).astype(int)
    df_copy['has_congestive_heart_failure'] = diagnosis_startswith(df_copy, CONGESTIVE_HEART_FAILURE_CODES).astype(int)
    df_copy['has_peripheral_vascular_disease'] = diagnosis_startswith(df_copy, PERIPHERAL_VASCULAR_DISEASE_CODES).astype(int)
    df_copy['has_cerebrovascular_disease'] = diagnosis_startswith(df_copy, CEREBROVASCULAR_DISEASE_CODES).astype(int)
    df_copy['has_dementia'] = diagnosis_startswith(df_copy, DEMENTIA_CODES).astype(int)
    df_copy['has_chronic_pulmonary_disease'] = diagnosis_startswith(df_copy, CHRONIC_PULMONARY_DISEASE_CODES).astype(int)
    df_copy['has_rheumatic_disease'] = diagnosis_startswith(df_copy, RHEUMATIC_DISEASE_CODES).astype(int)
    df_copy['has_peptic_ulcer_disease'] = diagnosis_startswith(df_copy, PEPTIC_ULCER_DISEASE_CODES).astype(int)
    df_copy['has_mild_liver_disease'] = diagnosis_startswith(df_copy, MILD_LIVER_DISEASE_CODES).astype(int)
    df_copy['has_diabetes_without_complications'] = diagnosis_startswith(df_copy, DIABETES_WITHOUT_COMPLICATIONS_CODES).astype(int)
    df_copy['has_mild_renal_disease'] = diagnosis_startswith(df_copy, MILD_RENAL_DISEASE_CODES).astype(int)
    df_copy['has_diabetes_with_chronic_complications'] = diagnosis_startswith(df_copy, DIABETES_WITH_CHRONIC_COMPLICATONS_CODES).astype(int)
    df_copy['has_hemiplegia_or_paraplegia'] = diagnosis_startswith(df_copy, HEMIPLEGIA_OR_PARAPLEGIA_CODES).astype(int)
    df_copy['has_malignancy'] = diagnosis_startswith(df_copy, MALIGNANCY_CODES).astype(int)
    df_copy['has_severe_liver_disease'] = diagnosis_startswith(df_copy, SEVERE_LIVER_DISEASE_CODES).astype(int)
    df_copy['has_severe_renal_disease'] = diagnosis_startswith(df_copy, SEVERE_RENAL_DISEASE_CODES).astype(int)
    df_copy['has_metastatic_solid_tumor'] = diagnosis_startswith(df_copy, METASTATIC_SOLID_TUMOR_CODES).astype(int)
    df_copy['has_hiv'] = diagnosis_startswith(df_copy, HIV_CODES).astype(int)
    df_copy['has_aids'] = diagnosis_startswith(df_copy, AIDS_CODES).astype(int)

    return df_copy