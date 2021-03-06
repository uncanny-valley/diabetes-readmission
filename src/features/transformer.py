from boruta import BorutaPy
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one
from scipy import sparse

from src.data.load_dataset import convert_id_to_discharge_disposition
from src.features.icd9 import icd9_to_classification, icd9_to_category, generate_comorbidity_index_predicates


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

class DiagnosisComorbidityExtractor(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return generate_comorbidity_index_predicates(X)

class DiagnosisMapper(TransformerMixin):
    """
    Translates ICD9 codes to semantic classifications
    """
    def __init__(self):
        self.columns = ['diag_1', 'diag_2', 'diag_3']
        
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
    def __init__(self, columns=[], special_cases={}, nan_label='Not Available'):
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
    def __init__(self, columns=[]):
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
        for (features, operation, combined_feature_name, target_type) in self.combined_features:
            df[combined_feature_name] = df.loc[:, features].apply(operation, axis=1).astype(target_type)

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
                # Skip column if it's a binary or unary feature
                if len(self._column_categories[column]) < 3:
                    continue

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

            if df[column].dtype.name == 'object':
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


class OneHotEncoder(TransformerMixin):
    def __init__(self, columns=None, columns_to_ignore=None, drop_first=False):
        self.columns = columns
        self.columns_to_ignore = columns_to_ignore
        self._drop_first = drop_first

    def fit(self, X: pd.DataFrame, y=None, **fit_params):
        self.columns = self.columns if self.columns is not None else X.columns

        if self.columns_to_ignore:
            self.columns = list(set(self.columns) - set(self.columns_to_ignore))

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(X, columns=self.columns, drop_first=self._drop_first)

class CollinearityThreshold(TransformerMixin):
    def __init__(self, threshold:np.float64=0.9, correlation_method:str='spearman', verbose=False):
        """
        Removes features that are considered highly correlated.

        Args:
            threshold (np.float64, optional): The upper bound on the acceptable correlation coefficient between two features. Defaults to 0.9.
        """
        self._threshold = threshold
        self._correlated_pairs = []
        self._verbose = verbose
    
    def fit(self, X: pd.DataFrame, y=None, **fit_params):
        df = X.copy()

        corr = pd.DataFrame(np.abs(np.corrcoef(df.values, rowvar=False)), columns=df.columns, index=df.columns)

        # Select upper triangle of matrix
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))

        for column, rows in upper.iteritems():
            self._correlated_pairs.extend([(column, i) if column < i else (i, column) for (i, coef) in rows.iteritems() if abs(coef) > self._threshold])
    
        if self._verbose:
            print(f'[{self.__class__.__name__}] Correlated features: {self._correlated_pairs}')
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Drop first feature in each correlated pair
        return X.drop(columns=[f1 for (f1, f2) in self._correlated_pairs])


class BorutaFeatureSelector(TransformerMixin):
    def __init__(self, columns=None, max_iter=100, random_state:int=0):
        self._forest = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
        self._boruta = BorutaPy(self._forest, n_estimators='auto', max_iter=max_iter, random_state=random_state)

    def fit(self, X: pd.DataFrame, y:pd.Series=None, **fit_params):
        self._boruta.fit(X.values, y[X.index].values)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._boruta.transform(X)