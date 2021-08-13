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
CONGESTIVE_HEART_FAILURE_CODES = ('428', '398.91', '402.01', '402.11', '402.91', '404.01', '404.03', '404.11', '404.13', '404.91', '404.93')
PERIPHERAL_VASCULAR_DISEASE_CODES = ('093.0', '437.3', '447.1', '557.1', '557.9', '443.2', '443.8', 'V43.4', '440', '441')
CEREBROVASCULAR_DISEASE_CODES = ('362.34', '430', '431', '432', '433', '434', '435', '436', '437', '438')
DEMENTIA_CODES = ('290', '294.1', '331.2')
CHRONIC_PULMONARY_DISEASE_CODES = ('416.8', '416.9', '506.4', '508.1', '508.8', '490', '491', '492', '493', '494', '495', '496', '500', '501', '502', '503', '504', '505')
RHEUMATIC_DISEASE_CODES = ('446.5', '710.0', '710.1', '710.2', '710.3', '710.4', '714.0', '714.1', '714.2', '714.8', '725')
PEPTIC_ULCER_DISEASE_CODES = ('531', '532', '533', '534')
MILD_LIVER_DISEASE_CODES = ('070.22', '070.23', '070.32', '070.33', '070.44', '070.54', '070.6', '070.9', '573.3', '573.4', '573.8', '573.9', '570', '571', 'V42.7')
DIABETES_WITHOUT_COMPLICATIONS_CODES = ('250.8', '250.9')
HEMIPLEGIA_OR_PARAPLEGIA_CODES = ('342', '343', '334.1')
MALIGNANCY_CODES = ('238.6')
RENAL_DISEASE_CODES = ('403.01', '403.11', '403.11', '403.91', '404.02', '404.03', '404.12', '404.13', '404.92', '404.93', '582', '585', '586', '588.0', 'V42.0', 'V45.1', 'V56')
AIDS_HIV_CODES = ('042', '043', '044')
METASTATIC_SOLID_TUMOR_CODES = ('196', '197', '198', '199')

# Intervals
CONGESTIVE_HEART_FAILURE_INTERVALS = ('425.4', '425.9')
PERIPHERAL_VASCULAR_DISEASE_INTERVALS = ('443.1', '443.9')
DIABETES_WITHOUT_CHRONIC_COMPLICATIONS_INTERVAL = ('250.0', '250.3')
DIABETES_WITH_CHRONIC_COMPLICATIONS_INTERVAL = ('250.4', '250.7')
RHEUMATIC_DISEASE_INTERVALS = ('714.0', '714.2')
HEMIPLEGIA_OR_PARAPLEGIA_INTERVALS = ('344.0', '344.6')
RENAL_DISEASE_INTERVALS = ('583.0', '583.7')
MALIGNANCY_INTERVALS = [('140', '172'), ('174', '195.8'), ('200', '208')]
SEVERE_LIVER_DISEASE_INTERVALS = [('456.0', '456.2'), ('572.2', '572.8')]

def extract_code(code: str) -> Tuple[str, float]:
    try: 
        return None, float(code)
    except ValueError:
        if code is None:
            raise ValueError('Given code was None type')

        if code[0] == 'E' or code[0] == 'V':
            return code[0], float(code[1:])
        else:
            return None, -1.
        

def diagnosis_within_closed_interval(df: pd.DataFrame, interval: Tuple[str]) -> pd.Series:
    lower, upper = interval

    p1, lower = extract_code(lower)
    p2, upper = extract_code(upper)

    if (p1 is not None or p2 is not None) and p1 != p2: 
        raise ValueError(f'Interval upper and lower bound must have same category prefix. Upper bound: {upper}, lower bound: {lower}')

    prefix = p1

    within_closed_interval = lambda x: (extract_code(x)[1] >= lower) and (extract_code(x)[1] <= upper)

    if prefix is not None:
        return ((df.diag_1.str.startswith(prefix) & df.diag_1.apply(within_closed_interval)) | (df.diag_2.str.startswith(prefix) & df.diag_2.apply(within_closed_interval)) | (df.diag_3.str.startswith(prefix) & df.diag_3.apply(within_closed_interval)))
    
    return (df.diag_1.apply(within_closed_interval) | df.diag_2.apply(within_closed_interval) | df.diag_3.apply(within_closed_interval))

def diagnosis_startswith(df: pd.DataFrame, codes: Tuple[str]) -> pd.Series:
    return ((df.diag_1.str.startswith(codes)) | (df.diag_2.str.startswith(codes)) | (df.diag_3.str.startswith(codes)))

def generate_comorbidity_index_predicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates comorbidity-related features based on the Charlson Comorbidity Index (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6684052/)
    """
    df_copy = df.copy()
    df_copy['has_myocardial_infarction'] = diagnosis_startswith(df_copy, MYOCARDIAL_INFARCTION_CODES).astype(int)
    df_copy['has_congestive_heart_failure'] = (diagnosis_startswith(df_copy, CONGESTIVE_HEART_FAILURE_CODES) | diagnosis_within_closed_interval(df_copy, interval=CONGESTIVE_HEART_FAILURE_INTERVALS)).astype(int)
    df_copy['has_peripheral_vascular_disease'] = (diagnosis_startswith(df_copy, PERIPHERAL_VASCULAR_DISEASE_CODES) | diagnosis_within_closed_interval(df_copy, interval=PERIPHERAL_VASCULAR_DISEASE_INTERVALS)).astype(int)
    df_copy['has_cerebrovascular_disease'] = diagnosis_startswith(df_copy, CEREBROVASCULAR_DISEASE_CODES).astype(int)
    df_copy['has_dementia'] = diagnosis_startswith(df_copy, DEMENTIA_CODES).astype(int)
    df_copy['has_chronic_pulmonary_disease'] = diagnosis_startswith(df_copy, CHRONIC_PULMONARY_DISEASE_CODES).astype(int)
    df_copy['has_rheumatic_disease'] = (diagnosis_startswith(df_copy, RHEUMATIC_DISEASE_CODES) | diagnosis_within_closed_interval(df_copy, interval=RHEUMATIC_DISEASE_INTERVALS)).astype(int)
    df_copy['has_peptic_ulcer_disease'] = diagnosis_startswith(df_copy, PEPTIC_ULCER_DISEASE_CODES).astype(int)
    df_copy['has_mild_liver_disease'] = diagnosis_startswith(df_copy, MILD_LIVER_DISEASE_CODES).astype(int)
    df_copy['has_diabetes_without_complications'] = (diagnosis_startswith(df_copy, DIABETES_WITHOUT_COMPLICATIONS_CODES) | diagnosis_within_closed_interval(df_copy, interval=DIABETES_WITHOUT_CHRONIC_COMPLICATIONS_INTERVAL)).astype(int)
    df_copy['has_diabetes_with_chronic_complications'] = diagnosis_within_closed_interval(df_copy, interval=DIABETES_WITH_CHRONIC_COMPLICATIONS_INTERVAL).astype(int)
    df_copy['has_hemiplegia_or_paraplegia'] = (diagnosis_startswith(df_copy, HEMIPLEGIA_OR_PARAPLEGIA_CODES) | diagnosis_within_closed_interval(df_copy, interval=HEMIPLEGIA_OR_PARAPLEGIA_INTERVALS)).astype(int)
    df_copy['has_malignancy'] = (diagnosis_startswith(df_copy, MALIGNANCY_CODES) | diagnosis_within_closed_interval(df_copy, interval=MALIGNANCY_INTERVALS[0]) | diagnosis_within_closed_interval(df_copy, interval=MALIGNANCY_INTERVALS[1]) | diagnosis_within_closed_interval(df_copy, interval=MALIGNANCY_INTERVALS[2])).astype(int)
    df_copy['has_severe_liver_disease'] = (diagnosis_within_closed_interval(df_copy, interval=SEVERE_LIVER_DISEASE_INTERVALS[0]) | diagnosis_within_closed_interval(df_copy, interval=SEVERE_LIVER_DISEASE_INTERVALS[1])).astype(int)
    df_copy['has_renal_disease'] = (diagnosis_startswith(df_copy, RENAL_DISEASE_CODES) | diagnosis_within_closed_interval(df_copy, interval=RENAL_DISEASE_INTERVALS)).astype(int)
    df_copy['has_metastatic_solid_tumor'] = diagnosis_startswith(df_copy, METASTATIC_SOLID_TUMOR_CODES).astype(int)
    df_copy['has_aids_hiv'] = diagnosis_startswith(df_copy, AIDS_HIV_CODES).astype(int)

    return df_copy