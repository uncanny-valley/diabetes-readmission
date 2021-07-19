import math
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels


def lace_length_of_stay_score(length_of_stay_in_days: np.uint16) -> np.uint8:
    if length_of_stay_in_days < 4:
        return length_of_stay_in_days
    elif length_of_stay_in_days < 7:
        return 4
    elif length_of_stay_in_days < 14:
        return 5
    else:
        return 7

def lace_comorbidity_score(has_myocardial_infarction:bool=False,
    has_congestive_heart_failure:bool=False,
    has_peripheral_vascular_disease:bool=False,
    has_cerebrovascular_disease:bool=False, 
    has_dementia:bool=False,
    has_chronic_pulmonary_disease:bool=False,
    has_rheumatic_disease:bool=False,
    has_peptic_ulcer_disease:bool=False,
    has_mild_liver_disease:bool=False,
    has_diabetes_without_complications:bool=False,
    has_mild_renal_disease:bool=False,
    has_diabetes_with_chronic_complications:bool=False,
    has_hemiplegia_or_paraplegia:bool=False,
    has_malignancy:bool=False,
    has_severe_liver_disease:bool=False,
    has_severe_renal_disease:bool=False,
    has_hiv:bool=False,
    has_metastatic_solid_tumor:bool=False,
    has_aids:bool=False
) -> np.uint8:
    score = 0
    tier_one_factors = np.array([has_myocardial_infarction, has_congestive_heart_failure, has_peripheral_vascular_disease,
        has_cerebrovascular_disease, has_dementia, has_chronic_pulmonary_disease, has_rheumatic_disease,
        has_peptic_ulcer_disease, has_mild_liver_disease, has_diabetes_without_complications, has_mild_renal_disease], dtype=np.uint8)
    tier_two_factors = np.array([has_diabetes_with_chronic_complications, has_hemiplegia_or_paraplegia, has_malignancy], dtype=np.uint8)
    tier_three_factors = np.array([has_severe_liver_disease, has_severe_renal_disease, has_hiv], dtype=np.uint8)
    tier_six_factors = np.array([has_metastatic_solid_tumor, has_aids], dtype=np.uint8)

    return tier_one_factors.sum() + 2 * tier_one_factors.sum() + 3 * tier_three_factors.sum()

def lace_comorbidity_score_by_row(row: pd.Series) -> pd.Series:
    return lace_comorbidity_score(
        has_myocardial_infarction=row.has_myocardial_infarction if 'has_myocardial_infarction' in row else False,
        has_congestive_heart_failure=row.has_congestive_heart_failure if 'has_congestive_heart_failure' in row else False,
        has_peripheral_vascular_disease=row.has_peripheral_vascular_disease if 'has_peripheral_vascular_disease' in row else False,
        has_cerebrovascular_disease=row.has_cerebrovascular_disease if 'has_cerebrovascular_disease' in row else False,
        has_dementia=row.has_dementia if 'has_dementia' in row else False,
        has_chronic_pulmonary_disease=row.has_chronic_pulmonary_disease if 'has_chronic_pulmonary_disease' in row else False,
        has_rheumatic_disease=row.has_rheumatic_disease if 'has_rheumatic_disease' in row else False,
        has_peptic_ulcer_disease=row.has_peptic_ulcer_disease if 'has_peptic_ulcer_disease' in row else False,
        has_mild_liver_disease=row.has_mild_liver_disease if 'has_mild_liver_disease' in row else False,
        has_diabetes_without_complications=row.has_diabetes_without_complications if 'has_diabetes_without_complications' in row else False,
        has_mild_renal_disease=row.has_mild_renal_disease if 'has_mild_renal_disease' in row else False,
        has_diabetes_with_chronic_complications=row.has_diabetes_with_chronic_complications if 'has_diabetes_with_chronic_complications' else False,
        has_hemiplegia_or_paraplegia=row.has_hemiplegia_or_paraplegia if 'has_hemiplegia_or_paraplegia' in row else False,
        has_malignancy=row.has_malignancy if 'has_malignancy' in row else False,
        has_severe_liver_disease=row.has_severe_liver_disease if 'has_severe_liver_disease' in row else False,
        has_severe_renal_disease=row.has_severe_renal_disease if 'has_severe_renal_disease' in row else False,
        has_hiv=row.has_hiv if 'has_hiv' in row else False,
        has_metastatic_solid_tumor=row.has_metastatic_solid_tumor if 'has_metastatic_solid_tumor' in row else False,
        has_aids=row.has_aids if 'has_aids' in row else False
    )

class LACEIndexClassifier(BaseEstimator, ClassifierMixin):
    """
    This classifies based on the LACE index, which is a de facto clinical method for identifying
    high likelihood to readmit early. We use a threshold of 11 to distinguish between early and
    non-early readmission patients, based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5726103/
    """

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        return self

    def predict(self, X):
        df_copy = X.copy()
        lace_scores = df_copy.days_in_hospital.apply(lace_length_of_stay_score) \
            + (df_copy.admission_type_id.apply(lambda x: 3 if x == 1 else 0)) \
            + (df_copy.number_emergency.apply(lambda x: min(x, 4))) \
            + (df_copy.apply(lace_comorbidity_score_by_row, axis=1))

        return (lace_scores >= 11).astype(int)

    def predict_proba(self, X):
        predictions = self.predict(X)
        return pd.concat([1-predictions, predictions], axis=1).to_numpy()