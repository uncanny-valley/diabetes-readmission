import os
import pandas as pd
import pathlib
from typing import Tuple
from imblearn.under_sampling import RandomUnderSampler


# The path to the central directory of the project
PROJECT_ROOT = os.path.abspath(os.curdir)

# The project root is erroneously set to one level below in the `notebooks` directory if invoked in a Jupyter notebook
if PROJECT_ROOT.split('/')[-1] == 'notebooks':
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

RAW_FOLDER = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_FOLDER = os.path.join(PROJECT_ROOT, 'data', 'processed')
INTERIM_FOLDER = os.path.join(PROJECT_ROOT, 'data', 'interim')


def find_raw_path(filename: str) -> str:
    return os.path.join(RAW_FOLDER, filename)

def find_interim_path(filename: str) -> str:
    return os.path.join(INTERIM_FOLDER, filename)

def find_preprocessed_path(filename: str) -> str:
    return os.path.join(PROCESSED_FOLDER, filename)

def load_raw_data(filepath: str=None) -> pd.DataFrame: 
    if filepath is None:
        return pd.read_csv(find_raw_path('diabetes.csv'), index_col=0)
    else:
        return pd.read_csv(filepath, index_col=0)

def load_data(filename, index_col=None) -> pd.DataFrame:
    file_ext = pathlib.Path(filename).suffix
    if file_ext == '.pkl':
        return pd.read_pickle(filename)
    elif file_ext == '.csv':
        return pd.read_csv(filename, index_col=index_col)
    else:
        raise ValueError(f'Unsupported file extension: {file_ext}')

def undersample_preprocessed_data(X_train: pd.DataFrame, y_train: pd.DataFrame, random_seed: int=0) -> Tuple[pd.DataFrame, pd.Series]:
    random_undersampler = RandomUnderSampler(random_state=random_seed)
    return random_undersampler.fit_resample(X_train, y_train)

def split_dataset(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    dataset = data.reset_index(drop=True)
    X = dataset.drop(columns=['is_readmitted_early'])
    y = dataset.is_readmitted_early
    return X, y

def load_preprocessed_csv(filename, index_col=None):
    return pd.read_csv(find_preprocessed_path(filename), index_col=index_col)

def load_preprocessed_pickle(filename) -> pd.DataFrame:
    return pd.read_pickle(find_preprocessed_path(filename))

def load_preprocessed_data(filename, index_col=None) -> pd.DataFrame:
    file_ext = pathlib.Path(filename).suffix
    if file_ext == '.pkl':
        return load_preprocessed_pickle(filename)
    elif file_ext == '.csv':
        return load_preprocessed_csv(filename, index_col=index_col)
    else:
        raise ValueError(f'Unsupported file extension: {file_ext}')

def load_interim_csv(filename, index_col=None):
    return pd.read_csv(find_interim_path(filename), index_col=index_col)

def load_interim_pickle(filename) -> pd.DataFrame:
    return pd.read_pickle(find_interim_path(filename))

def load_mapping(feature_name: str) -> pd.DataFrame:
    return pd.read_csv(find_raw_path(f'{feature_name}_mappings.csv'))

def convert_id_to_admission_source(series: pd.Series) -> pd.Series:
    f = 'admission_source_id'
    m = load_mapping(f).set_index(f)
    return series.replace(m.description)

def convert_id_to_admission_type(series: pd.Series) -> pd.Series:
    f = 'admission_type_id'
    m = load_mapping(f).set_index(f)
    return series.replace(m.description)

def convert_id_to_discharge_disposition(series: pd.Series) -> pd.Series:
    f = 'discharge_disposition_id'
    m = load_mapping(f).set_index(f)
    return series.replace(m.description)