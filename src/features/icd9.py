from math import isnan
import numpy as np

def icd9_to_classification(code: str) -> str:
    if code is np.nan or code is None or (isinstance(code, float) and isnan(code)):
        return 'Not Available'

    try:
        icd9_code = float(code)
    except ValueError:
        if code[0] == 'E':
            return 'Supplementary classification of external causes of injury and poisoning'
        elif code[0] == 'V':
            return 'Supplementary classification of factors influencing health status and contact with health services'
        else:
            return 'Other'
    icd9_mapping = {
        (1., 139.): 'Infectious and parasitic diseases',
        (140., 239.): 'Neoplasms',
        (240., 279.): 'Endocrine, nutritional, and metabolic diseases and immunity disorders',
        (280., 289.): 'Diseases of blood and blood-forming organs',
        (290., 319.): 'Mental disorders',
        (320., 389.): 'Diseases of the nervous system and sense organs',
        (390., 459.): 'Diseases of the circulatory system',
        (460., 519.): 'Diseases of the respiratory system',
        (520., 579.): 'Diseases of the digestive system',
        (580., 629.): 'Diseases of the genitourinary system',
        (630., 676.): 'Complications of pregnancy, childbirth and the puerperium',
        (680., 709.): 'Diseases of the skin and subcutaneous tissue',
        (710., 739.): 'Diseases of the musculoskeletal system and connective tissue',
        (740., 759.): 'Congenital anomalies',
        (760., 779.): 'Certain conditions originating in the perinatal period',
        (780., 799.): 'Symptoms, signs and ill-defined conditions',
        (800., 999.): 'Injury and poisoning'
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
