from math import isnan
import numpy as np

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
