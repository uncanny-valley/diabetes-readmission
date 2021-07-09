import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one
from scipy import sparse

from src.features.icd9 import icd9_to_classification, icd9_to_category
from src.data.load_dataset import convert_id_to_discharge_disposition


class PandasFeatureUnion(FeatureUnion):
    """
    Sci-kit learn feature union with Pandas properties
    https://zablo.net/blog/post/pandas-dataframe-in-scikit-learn-feature-union/index.html
    """
    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(
                transformer=trans,
                X=X,
                y=y,
                weight=weight,
                **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

    def merge_dataframes_by_column(self, Xs):
        return pd.concat(Xs, axis='columns', copy=False)

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(
                transformer=trans,
                X=X,
                y=None,
                weight=weight)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

class RowFilter(TransformerMixin):
    """
    Filters rows based on a set of 
    """
    def __init__(self, filters={}):
        self.filters = filters

    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X):
        df = X.copy()
        for column, condition in self.filters.items():
            df = df[condition(df[column])]

            # Reset categories if column is categorical data type
            if df[column].dtype == 'category':
                df[column] = df[column].cat.remove_unused_categories()

        return df


class ColumnSelector(TransformerMixin):
    """
    Selects only a specified subset of features
    """
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X):
        return X[self.columns]


class ColumnFilter(TransformerMixin):
    """
    Filters a specified subset of features
    """
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X):
        return X.drop(columns=self.columns, errors='ignore')

class DischargeMapper(TransformerMixin):
    """
    Translates discharge disposition IDs to classification name
    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        for column in self.columns:
            df[column] = convert_id_to_discharge_disposition(df[column])
        return df


class DiagnosisMapper(TransformerMixin):
    """
    Translates ICD9 codes to semantic classifications
    """
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        for column in self.columns:
            t1 = f'{column}_t1'
            t2 = f'{column}_t2'
            df[t1] = df[column].apply(lambda code: icd9_to_classification(code, concise=True))
            df[t2] = df[column].apply(icd9_to_category)

        return df.drop(columns=self.columns)


class EncodeNaNCategoricalImputer(TransformerMixin):
    def __init__(self, columns, special_cases={}, nan_label='Not Available'):
        self.columns = columns
        self._nan_label = nan_label
        self._special_cases = special_cases

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        df = X.copy()
        for column in self.columns:
            if column in self._special_cases:
                df[column] = df[column].fillna(self._special_cases.get(column))
            else:
                df[column] = df[column].fillna(self._nan_label)
        return df

class MostFrequentCategoricalImputer(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        df = X.copy()
        for column in self.columns:
            df[column] = df[column].fillna(df[column].mode()[0])
        return df

class FeatureCombiner(TransformerMixin):
    def __init__(self, combined_features):
        self.combined_features = combined_features

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        df = X.copy()
        for (features, operation, combined_feature_name) in self.combined_features:
            df[combined_feature_name] = df.loc[:, features].apply(operation, axis=1)

        return df


class CategoryCollapseThreshold(TransformerMixin):
    def __init__(self, columns:list=None, threshold:float=0.05, default_category:str='Other', special_cases:dict={}, verbose:bool=False):
        self.columns = columns
        self.threshold = threshold
        self._column_value_map = {}
        self._column_categories = {}
        self._default_category = default_category
        self._special_cases = special_cases
        self._verbose = verbose
        
    def fit(self, X, y=None, **fit_params):
        self.columns = self.columns if self.columns is not None else X.columns
        self._column_value_map = {}
        self._column_categories = {}
    
        for column in self.columns:
            if column not in self._column_categories:
                self._column_categories[column] = X[column].cat.categories if X[column].dtype.name == 'category' else X[column].astype('category').cat.categories

            value_counts = X[column].value_counts(normalize=True)
            for value, proportion in value_counts.items():
                if proportion < self.threshold:
                    if column in self._special_cases:
                        category = self._special_cases[column]
                    else:
                        category = self._default_category

                    if self._verbose:
                        print(f'[{self.__class__.__name__}={self.threshold}] Column: {column}, Value: ({value}, {round(proportion, 5)}). Replacing value with {category}.')
                    self._column_value_map[(column, value)] = category

        return self

    def transform(self, X):
        df = X.copy()
        for column in self.columns:
            df[column] = df[column].apply(lambda v: self._get_column_value(column, v))
            df[column] = df[column].astype('category')
        return df

    def _get_column_value(self, column, value):
        # If `value` is an unseen category, 
        if value not in self._column_categories[column]:
            # Return a custom catch-all category for the column, if specified
            if column in self._special_cases:
                return self._special_cases[column]
            # Otherwise, return the default catch-all category
            else:
                return self._default_category


        if (column, value) in self._column_value_map:
            return self._column_value_map[(column, value)]
        else:
            return value

class CategoricalHomogeneityThreshold(TransformerMixin):
    def __init__(self, columns=None, threshold=0.05, verbose=False):
        self.columns = columns
        self.threshold = threshold
        self._columns_to_drop = []
        self._verbose = verbose

    def fit(self, X, y=None, **fit_params):
        self._columns_to_drop = []
        self.columns = self.columns if self.columns is not None else X.columns
        for column in self.columns:
            probabilities = X[column].value_counts(normalize=True)
            entropy = -(probabilities * np.log(probabilities)).sum()
            if entropy < self.threshold:
                self._columns_to_drop.append(column)
                if self._verbose:
                    print(f'[{self.__class__.__name__}] Column: {column}, Entropy: {entropy}. Dropping column')

        return self

    def transform(self, X):
        return X.drop(columns=self._columns_to_drop)
