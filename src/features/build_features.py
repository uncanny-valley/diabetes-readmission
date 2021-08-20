# -*- coding: utf-8 -*-
import click
import logging
import numpy as np
import pandas as pd
from os import path
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from typing import List, Callable
from mypy_extensions import TypedDict

from abc import ABC, abstractmethod

from category_encoders import OrdinalEncoder

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearndf.transformation import SimpleImputerDF, StandardScalerDF, FunctionTransformerDF, VarianceThresholdDF

from src.features.icd9 import icd9_to_classification, icd9_to_category
from src.features.transformer import PandasFeatureUnion, RowFilter, ColumnSelector, ColumnFilter, \
DiagnosisMapper, EncodeNaNCategoricalImputer, MostFrequentCategoricalImputer, CategoryCollapseThreshold, \
CategoricalHomogeneityThreshold, FeatureCombiner, DiagnosisComorbidityExtractor, CollinearityThreshold, OneHotEncoder

from src.data.load_dataset import load_data, split_dataset, find_preprocessed_path


class RowFilterDict(TypedDict):
    feature: str
    filter: Callable[[pd.Series], bool]


class AbstractPipelineCreator(ABC):
    @abstractmethod
    def create_nominal_preprocessing_pipeline(self) -> Pipeline:
        pass

    @abstractmethod
    def create_ordinal_preprocessing_pipeline(self) -> Pipeline:
        pass

    @abstractmethod
    def create_numerical_preprocessing_pipeline(self) -> Pipeline:
        pass

    @abstractmethod
    def create_preprocessing_pipeline(self) -> Pipeline:
        pass


class LaceIndexPipelineCreator(AbstractPipelineCreator):
    def __init__(self, row_filters: RowFilterDict, verbose: bool=False):
        self.verbose = verbose
        self.nominal_features = ['admission_type_id', 'diag_1', 'diag_2', 'diag_3']
        self.numerical_features = ['number_emergency', 'days_in_hospital']

        self.row_filters = row_filters
        self.column_filters = ['diag_1', 'diag_2', 'diag_3']

        self.categorical_features_to_encode_nan_as_category = ['diag_1', 'diag_2', 'diag_3']
        self.encode_nan_as_category_imputer_special_cases = {'admission_type_id': 5}


    def create_nominal_preprocessing_pipeline(self) -> Pipeline:
        return make_pipeline(
            ColumnSelector(columns=self.nominal_features),
            EncodeNaNCategoricalImputer(self.categorical_features_to_encode_nan_as_category, special_cases=self.encode_nan_as_category_imputer_special_cases),
            DiagnosisComorbidityExtractor(),
            verbose=self.verbose)

    def create_numerical_preprocessing_pipeline(self) -> Pipeline:
        return make_pipeline(
            ColumnSelector(columns=self.numerical_features),
            SimpleImputerDF(strategy='median'),
            verbose=self.verbose)
    
    def create_ordinal_preprocessing_pipeline(self) -> Pipeline:
        return make_pipeline(
            ColumnSelector(columns=[]),            
            verbose=self.verbose)

    def create_preprocessing_pipeline(self) -> Pipeline:
        return Pipeline([
            ('row_filter', RowFilter(self.row_filters)),
            ('column_selector', ColumnSelector(self.nominal_features + self.numerical_features)),
            ('features', PandasFeatureUnion([
                ('numerical', self.create_numerical_preprocessing_pipeline()),
                ('nominal', self.create_nominal_preprocessing_pipeline()),
            ])),
            ('column_filter', ColumnFilter(self.column_filters))
        ])




class PipelineCreator(AbstractPipelineCreator):
    def __init__(self, nominal_features: List[str], ordinal_features: List[str],
                 numerical_features: List[str], columns_to_drop: List[str], row_filters: RowFilterDict,
                 verbose: bool=True, **kwargs): 
        """
        Creates preprocessing pipelines

        Args:
            nominal_features (List[str]): Column features that are nominal categorical variables
            ordinal_features (List[str]): Column features that are ordinal categorical variables
            numerical_features (List[str]): Column features that are numerical variables
            columns_to_drop (List[str]): Column features to drop
            row_filters (RowFilterDict): A mapping between feature and conditional filters to filter observations in the dataframe
            verbose (bool): Whether to print helpful messages to standard output. Defaults to False.
        Keyword Args:
            categorical_features_to_encode_nan_as_category (List[str]): Features to encode missing values as its own category. Defaults to an empty list.
            encode_nan_as_category_imputer_special_cases (dict): A mapping from feature to the preferred label to refer to missing values. If feature not specified, it will use the encode_nan_as_category_imputer_default_label. Defaults to an empty dict.
            encode_nan_as_category_imputer_default_label (str): The new label to refer to NaN/missing values in all features. Defaults to "Not Available."

            most_frequent_imputer_features (List[str]): Features whose missing values to impute with the most frequent value. 

            nominal_features_to_combine (List[Tuple[Tuple, Callable, str, type]]): The input to a FeatureCombiner instance, where we will apply a function to a group of features to create an entirely new feature.
                                                                                 The list of tuples corresponding to (tuple of features to which we apply a function, the function, the feature name of the new combined feature, the type for the new feature).

            categorical_homogeneity_threshold_features (List[str]): The nominal features on which to apply the CategoricalHomogeneityThreshold. If the amount of entropy in a feature is below 
                                                                    `categorical_homogeneity_threshold`, then the feature will be dropped. Defaults to all nominal features.
            categorical_homogeneity_threshold (float): The threshold at which to tolerate feature entropy. Features whose entropy fall below this threshold will be dropped. Defaults to 0.05.

            category_collapse_threshold_features (List[str]): The nominal features on which to apply the CategoryCollapseThreshold. If the proportion of occurrence of values within a given feature falls below
                                                              `category_collapse_threshold`, the value will be bucketed into the `category_collapse_threshold_default_label` category. Defaults to all nominal column features.
            category_collapse_threshold (float): The threshold at which to tolerate proportional occurrences of small feature values. If the proportion of occurrence of values within a given feature falls below
                                                 this threshold, the value will be bucketed into the `category_collapse_threshold_default_label` category. Defaults to 0.05.
            category_collapse_threshold_default_label (str): The default label with which to bucket small categories whose occurrence falls below the `category_collapse_threshold`. Defaults to 'Other'.
            category_collapse_threshold_special_cases (dict): A mapping from feature to the label with which to bucket small categories whose occurrence falls below the `category_collapse_threshold`. Defaults to an empty dict.

            one_hot_encoder_features_to_ignore (List[str]): Categorical to skip when one-hot encoding. Defaults to None.

            numerical_features_to_combine (List[Tuple[Tuple, Callable, str, type]]): The numerical input to a FeatureCombiner instance, where we will apply a function to a group of features to create an entirely new numerical feature.
                                           The list of tuples corresponding to (tuple of features to which we apply a function, the function, the feature name of the new combined feature, the type for the new feature).
    
            variance_threshold (float): All numerical features with a variance below this threshold will be removed. Defaults to 0.
            
            collinearity_threshold (float): This threshold is the greatest Pearson correlation coefficients between any two features that we will tolerate. If two features, exceed this threshold, the first will be dropped. Defaults to 0.9.
        """

        self.columns_to_drop = columns_to_drop
        self.row_filters = row_filters
        self.verbose = verbose

        # Categorical features
        self.nominal_features = nominal_features
        self.ordinal_features = ordinal_features

        self.categorical_features_to_encode_nan_as_category = kwargs.get('categorical_features_to_encode_nan_as_category', [])
        self.encode_nan_as_category_imputer_special_cases = kwargs.get('encode_nan_as_category_imputer_special_cases', {})
        self.encode_nan_as_category_imputer_default_label = kwargs.get('encode_nan_as_category_imputer_default_label', 'Not Available')

        self.most_frequent_imputer_features = kwargs.get('most_frequent_imputer_features', [])

        self.nominal_features_to_combine = kwargs.get('nominal_features_to_combine', [])

        self.categorical_homogeneity_threshold_features = kwargs.get('categorical_homogeneity_threshold_features')
        self.categorical_homogeneity_threshold = kwargs.get('categorical_homogeneity_threshold', 0.05)

        self.category_collapse_threshold_features = kwargs.get('category_collapse_threshold_features')
        self.category_collapse_threshold = kwargs.get('category_collapse_threshold', 0.05)
        self.category_collapse_threshold_default_label = kwargs.get('category_collapse_threshold_default_label', 'Other')
        self.category_collapse_threshold_special_cases = kwargs.get('category_collapse_threshold_special_cases', {})

        self.one_hot_encoder_features_to_ignore = kwargs.get('one_hot_encoder_features_to_ignore', [])


        # Numerical features
        self.numerical_features = numerical_features
        self.numerical_features_to_combine = kwargs.get('numerical_features_to_combine', [])
        self.variance_threshold = kwargs.get('variance_threshold', 0.1)

        self.collinearity_threshold = kwargs.get('collinearity_threshold', 0.9)



    def create_nominal_preprocessing_pipeline(self) -> Pipeline:
        return make_pipeline(
            ColumnSelector(columns=self.nominal_features),
            EncodeNaNCategoricalImputer(
                columns=self.categorical_features_to_encode_nan_as_category,
                special_cases=self.encode_nan_as_category_imputer_special_cases,
                nan_label=self.encode_nan_as_category_imputer_default_label),
            MostFrequentCategoricalImputer(columns=self.most_frequent_imputer_features),
            FeatureCombiner(combined_features=self.nominal_features_to_combine),
            DiagnosisComorbidityExtractor(),
            DiagnosisMapper(),
            CategoricalHomogeneityThreshold(
                columns=self.categorical_homogeneity_threshold_features,
                threshold=self.categorical_homogeneity_threshold,
                verbose=self.verbose),
            CategoryCollapseThreshold(
                columns=self.category_collapse_threshold_features,
                threshold=self.category_collapse_threshold,
                default_category=self.category_collapse_threshold_default_label,
                special_cases=self.category_collapse_threshold_special_cases,
                verbose=self.verbose),
            OneHotEncoder(drop_first=True, columns_to_ignore=self.one_hot_encoder_features_to_ignore),
            verbose=self.verbose)

    def create_ordinal_preprocessing_pipeline(self) -> Pipeline:
        return make_pipeline(
            ColumnSelector(columns=self.ordinal_features),
            SimpleImputerDF(strategy='most_frequent'),
            OrdinalEncoder(return_df=True),
            verbose=self.verbose)

    def create_numerical_preprocessing_pipeline(self, log_transform: bool=True) -> Pipeline:
        transformers = [
            ('columnselector', ColumnSelector(columns=self.numerical_features)),
            ('simpleimputerdf', SimpleImputerDF(strategy='median')),
            ('logtransformer', FunctionTransformerDF(np.log1p)),
            ('standardscaler', StandardScalerDF()),
            ('featurecombiner', FeatureCombiner(combined_features=self.numerical_features_to_combine)),
            ('variancethreshold', VarianceThresholdDF(threshold=self.variance_threshold))
        ]

        if not log_transform:
            del transformers[2]

        return Pipeline(
            steps=transformers,
            verbose=self.verbose)


    def create_preprocessing_pipeline(self, log_transform: bool=True) -> Pipeline:
        return Pipeline([
            ('column_filter', ColumnFilter(columns=self.columns_to_drop)),
            ('row_filter', RowFilter(self.row_filters)),
            ('features', PandasFeatureUnion([
                ('numerical', self.create_numerical_preprocessing_pipeline(log_transform=log_transform)),
                ('ordinal', self.create_ordinal_preprocessing_pipeline()),
                ('nominal', self.create_nominal_preprocessing_pipeline()),
            ])),
            ('collinearity_threshold', CollinearityThreshold(threshold=self.collinearity_threshold, verbose=self.verbose))
        ])



def compute_entropy(d1: str, d2: str, d3: str) -> float:
    diagnoses = [d1, d2, d3]
    num_diagnoses = len(diagnoses)
    prob_d1 = diagnoses.count(d1) / num_diagnoses
    prob_d2 = diagnoses.count(d2) / num_diagnoses
    prob_d3 = diagnoses.count(d3) / num_diagnoses
    return -(prob_d1 * np.log(prob_d1) + prob_d2 * np.log(prob_d2) + prob_d3 * np.log(prob_d3)) 


def diagnosis_diversity(r: pd.Series) -> pd.Series:
    d1 = icd9_to_category(r.diag_1)
    d2 = icd9_to_category(r.diag_2)
    d3 = icd9_to_category(r.diag_3)
    return compute_entropy(d1, d2, d3)


def transform_and_export_preprocessed_data(preprocessor: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, train_export_path: str, test_export_path: str) -> None:
    preprocessor.fit(X_train, y_train)
    X_train_enc = preprocessor.transform(X_train)
    X_test_enc = preprocessor.transform(X_test)
    train_enc = pd.concat([X_train_enc, y_train[X_train_enc.index]], axis=1)
    test_enc = pd.concat([X_test_enc, y_test[X_test_enc.index]], axis=1)
    logging.info(f'Finished transforming training set {train_enc.shape} and test set {test_enc.shape}')
    logging.info(f'Exporting training set to {train_export_path}. Exporting test set to {test_export_path}')
    train_enc.to_pickle(train_export_path)
    test_enc.to_pickle(test_export_path)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
def main(input_filepath: Path) -> None:
    """
    Preprocesses intermediate data.
    Args:
        input_filepath (Path): Path to intermediate data.
    """
    data = load_data(input_filepath)
    logging.info(f'Loaded input data: {data.shape}')

    target_feature = 'is_readmitted_early'
    X, y = split_dataset(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y, shuffle=True)

    columns_to_drop = ['payer_code', 'examide', 'citoglipton', 'glimepiride-pioglitazone']
    categorical_features = X.select_dtypes(exclude=[np.number]).drop(columns=columns_to_drop)
    ordinal_features = ['age']
    nominal_features = categorical_features.drop(columns=ordinal_features).columns.tolist()
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()


    row_filters = {
        # Expired or Hospice-related discharges are not likely to be readmitted. Remove neonatal observations
        'discharge_disposition_id': lambda s: ~s.isin([11, 13, 14, 19, 20, 21, 10]),
        # We remove observations related to birth or infancy
        'admission_source_id': lambda s: ~s.isin([11, 12, 13, 14])
    }

    categorical_features_to_encode_nan_as_category = ['diag_1', 'diag_2', 'diag_3', 'gender', 'max_glu_serum', 'medical_specialty', 'A1Cresult']
    keyword_arguments = {
        'categorical_features_to_encode_nan_as_category': categorical_features_to_encode_nan_as_category,
        'encode_nan_as_category_imputer_special_cases': {
            'admission_type_id': 5,
            'discharge_disposition_id': 25,
            'admission_source_id': 15,
            'race': 'Unknown/Invalid'
        },
        'most_frequent_imputer_features': list(set(nominal_features) - set(categorical_features_to_encode_nan_as_category)),
        'nominal_features_to_combine': [(('diag_1', 'diag_2', 'diag_3'), diagnosis_diversity, 'diagnosis_diversity', float)],
        'categorical_homogeneity_threshold': 0.05,
        'category_collapse_threshold': 0.05,
        'category_collapse_threshold_special_cases': {
            'discharge_disposition_id': 30,
            'admission_type_id': 9,
            'admission_source_id': 27  
        },
        'one_hot_encoder_features_to_ignore': ['diagnosis_diversity'],
        'numerical_features_to_combine': [
            (('number_inpatient', 'number_outpatient', 'number_emergency'), np.sum, 'service_utilization', int)
        ],
        'variance_threshold': 0.1,
        'collinearity_threshold': 0.9
    }


    creator = PipelineCreator(
        nominal_features=nominal_features,
        ordinal_features=ordinal_features,
        numerical_features=numerical_features,
        columns_to_drop=columns_to_drop,
        row_filters=row_filters,
        verbose=True,
        **keyword_arguments)

    preprocessor = creator.create_preprocessing_pipeline(log_transform=True)
    train_output_path = find_preprocessed_path('train.pkl')
    test_output_path = find_preprocessed_path('test.pkl')
    transform_and_export_preprocessed_data(preprocessor, X_train, y_train, X_test, y_test, train_output_path, test_output_path)

    lace_creator = LaceIndexPipelineCreator(row_filters=row_filters, verbose=True)
    lace_preprocessor = lace_creator.create_preprocessing_pipeline()
    train_output_path = find_preprocessed_path('train_lace.pkl')
    test_output_path = find_preprocessed_path('test_lace.pkl')
    transform_and_export_preprocessed_data(lace_preprocessor, X_train, y_train, X_test, y_test, train_output_path, test_output_path)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
