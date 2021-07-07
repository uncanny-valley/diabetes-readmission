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
        return pd.concat(Xs, axis="columns", copy=False)

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

class ColumnSelector(TransformerMixin):
    """
    Selects only a specified subset of features
    """
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.columns]

class ColumnSelector(TransformerMixin):
    """
    Selects only a specified subset of features
    """
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.columns]


class ColumnFilter(TransformerMixin):
    """
    Filters a specified subset of features
    """
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(columns=self.columns)

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

class DenseTransformer(TransformerMixin):
    def fit(self, X):
        return self

    def transform(self, X):
        return X.todense()