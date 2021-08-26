import click
import logging
from dotenv import find_dotenv, load_dotenv
from imblearn.base import BaseSampler
from imblearn.over_sampling import SMOTE, SMOTENC, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from typing import List, Tuple

from src.data.load_dataset import find_interim_path, load_data, split_dataset


SAMPLINGS_METHODS: List[Tuple[str, BaseSampler]] = [
    ('initial_data', None),
    ('smote', SMOTE(random_state=0, n_jobs=-1)),
    ('over_sampled', RandomOverSampler(sampling_strategy='minority', random_state=0)),
    ('under_sampled', RandomUnderSampler(random_state=0))
]

@click.command()
@click.argument('input_filepath', type=Path)
@click.argument('n_splits', type=int)
def main(input_filepath: Path, n_splits: int) -> None:
    training_data = load_data(input_filepath)
    skf = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)

    X_train, y_train = split_dataset(training_data)

    for name, sampler in SAMPLINGS_METHODS:
        logging.info(f'Sampling with {name}')
        for i, (train_indices, validation_indices) in enumerate(skf.split(X_train, y_train)):
            logging.info(f'Holdout fold: {i}')
            xt, yt = X_train.iloc[train_indices], y_train.iloc[train_indices]
            xv, yv = X_train.iloc[validation_indices], y_train.iloc[validation_indices]

            if name is not 'initial_data':
                xt, yt = sampler.fit_resample(xt, yt)
                
            xt.to_pickle(find_interim_path(f'X-train-{name}-fold-{i}.pkl'))
            yt.to_pickle(find_interim_path(f'y-train-{name}-fold-{i}.pkl'))
            xv.to_pickle(find_interim_path(f'X-validation-{name}-fold-{i}.pkl'))
            yv.to_pickle(find_interim_path(f'y-validation-{name}-fold-{i}.pkl'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()