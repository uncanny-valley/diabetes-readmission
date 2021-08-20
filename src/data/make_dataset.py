# -*- coding: utf-8 -*-
import click
import logging
import numpy as np
import pandas as pd
from os import path
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from typing import List

from load_dataset import load_raw_data, find_interim_path


MEDICATION_FEATURES = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
    'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide',
    'citoglipton', 'insulin', 'glyburide-metformin' , 'glipizide-metformin', 'glimepiride-pioglitazone', 
    'metformin-rosiglitazone', 'metformin-pioglitazone'
]

def categorize_medication_features(df: pd.DataFrame, medication_features: List[str]=MEDICATION_FEATURES) -> pd.DataFrame:
    out = df.copy()
    valid_categories = ['Up', 'Down', 'Steady', 'No']

    for f in medication_features:
        out[f] = pd.Categorical(out[f], categories=valid_categories)

    return out


def bucket_similar_categories_for_admission_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Missing data span three values: NaN (6), Not Available (5), and Not Mapped (8). To reduce noise and dimensionality, we assume that there is not a consequential distinction between the three and bucket them into "Not Available."
    """
    out = df.copy()
    out['admission_type_id'] = out.admission_type_id.replace(6, 5)
    out['admission_type_id'] = out.admission_type_id.replace(8, 5)
    out['admission_type_id'] = pd.Categorical(out.admission_type_id, categories=[
        1, 2, 3, 4, 5, 7
    ])
    return out


def bucket_similar_categories_for_admission_source(df: pd.DataFrame) -> pd.DataFrame:
    """
    Missing data map to NaN (17), Not Available (15), Not Mapped (20), Unknown/Invalid (21). We conflate these values by bucketing them in "Not Available."
    """
    out = df.copy()
    out['admission_source_id'] = out.admission_source_id.replace(17, 15)
    out['admission_source_id'] = out.admission_source_id.replace(20, 15)
    out['admission_source_id'] = out.admission_source_id.replace(21, 15)
    out['admission_source_id'] = pd.Categorical(out.admission_source_id, categories=[
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19,
        22, 23, 24, 25, 26
    ])
    return out


def bucket_similar_categories_for_discharge_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Missing data span three values: NaN (18), Not Mapped (25), and Unknown/Invalid (26). We assume no distinction between the three and bucket them into an existing value, e.g. "Not Mapped"
    """
    out = df.copy()
    out['discharge_disposition_id'] = out.discharge_disposition_id.replace(18, 25)
    out['discharge_disposition_id'] = out.discharge_disposition_id.replace(26, 25)
    out['discharge_disposition_id'] = pd.Categorical(out.discharge_disposition_id, categories=[
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20,
        21, 22, 23, 24, 25, 27, 28, 29
    ])

    return out


def binarize_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['is_readmitted_early'] = out.readmitted
    out['is_readmitted_early'] = out.is_readmitted_early.replace('>30', 0)
    out['is_readmitted_early'] = out.is_readmitted_early.replace('NO', 0)
    out['is_readmitted_early'] = out.is_readmitted_early.replace('<30', 1)
    out['is_readmitted_early'] = out.is_readmitted_early.astype('category')
    out = out.drop(columns=['readmitted'])
    return out


@click.command()
@click.option('--to-csv')
@click.option('--to-pickle')
def main(to_csv: bool, to_pickle: bool) -> None:
    """
    Cleans and imputes missing values in the raw data and exports it as intermediate (`interim`) data.

    Args:
        to_csv (bool): Whether or not to output the dataframe as a CSV (data/interim/00_diabetes.csv)
        to_pickle (bool): Whether or not to output the dataframe as a pickle (data/interim/00_diabetes.pkl)
    """

    logger = logging.getLogger(__name__)
    df = load_raw_data()

    # Only keep the first instance of a patient
    df = df.sort_values(by='encounter_id', axis='index').drop_duplicates(subset=['patient_nbr'], keep='first')
    df = df.reset_index()

    # Index by encounter and patient ID
    df = df.set_index(['encounter_id', 'patient_nbr'])

    # Standardize all null values in the datasetdata
    df[df == '?'] = np.nan
    df[df == 'None'] = np.nan

    # Classify null values in `race` as its own category
    df.loc[df.race.isna(), 'race'] = 'Not Available'

    # Classify null values in `medical_specialty` as its own category
    df.loc[df.medical_specialty.isna(), 'medical_specialty'] = 'Not Available'
    df['medical_specialty'] = df.medical_specialty.astype('category')

    # Classify null values in `payer_code` as its own category
    df.loc[df.payer_code.isnull(), 'payer_code'] = 'NA'
    df['payer_code'] = df.payer_code.astype('category')

    # # Remove all observations that are missing all diagnoses
    # df = df.loc[~(df.diag_1.isna() & df.diag_2.isna() & df.diag_3.isna())]

    # Classify null values in `A1Cresult` as its own category
    df.loc[df.A1Cresult.isna(), 'A1Cresult'] = 'Not Available'

    # Classify null values in `max_glu_serum` as its own category
    df.loc[df.max_glu_serum.isna(), 'max_glu_serum'] = 'Not Available'

    df = categorize_medication_features(df)

    # Categorize diabetes medication and change in medication
    df['diabetesMed'] = pd.Categorical(df.diabetesMed, categories=['Yes', 'No'])
    df['change'] = pd.Categorical(df.change, categories=['Ch', 'No'])

    # Merge similar null value mappings
    df = bucket_similar_categories_for_admission_type(df)
    df = bucket_similar_categories_for_admission_source(df)
    df = bucket_similar_categories_for_discharge_type(df)

    # Drop weight column due to high incidence of missing data
    df = df.drop(columns=['weight'])

    df = binarize_target_variable(df)

    df = df.rename(columns={'time_in_hospital': 'days_in_hospital'})

    logger.info('Finished cleaning and imputing the data')
    df.info()

    if to_csv:
        output_path = find_interim_path('00_diabetes.csv')
        logger.info(f'Exporting dataframe as CSV to {output_path}')
        df.to_csv(output_path)

    if to_pickle:
        output_path = find_interim_path('00_diabetes.pkl')
        logger.info(f'Exporting dataframe as pickle to {output_path}')       
        df.to_pickle(output_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
