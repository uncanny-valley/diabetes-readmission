# -*- coding: utf-8 -*-
import click
import logging
import numpy as np
import pandas as pd
from os import path
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import optuna

from joblib import dump
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

from src.data.load_dataset import load_data, undersample_preprocessed_data, split_dataset



@click.command()
@click.argument('path_to_study', type=click.Path(exists=True))
@click.argument('study_name', type=str)
@click.argument('training_data_path', type=click.Path(exists=True))
@click.argument('model_type', type=str)
@click.argument('output_filepath', type=Path)
def main(path_to_study: Path, study_name: str, training_data_path: Path, model_type: str, output_filepath: Path) -> None:
    random_seed = 0
    training_data = load_data(training_data_path)

    logging.info(f'Loaded training data: {training_data.shape}')

    X_train, y_train = split_dataset(training_data)
    X_train, y_train = undersample_preprocessed_data(X_train, y_train, random_seed=random_seed)

    logging.info(f'Training data size after resampling: {X_train.shape}')

    study = optuna.load_study(study_name=study_name, storage=f'sqlite:///{path_to_study}')
    best_params = study.best_params

    if model_type == 'random_forest':
        model = RandomForestClassifier(random_state=random_seed, **best_params)
    elif model_type == 'lgbm':
        model = LGBMClassifier(random_state=random_seed, **best_params)
        aliases = {
            'lambda_l1': 'reg_alpha',
            'lambda_l2': 'reg_lambda',
            'bagging_fraction': 'subsample',
            'bagging_freq': 'subsample_freq',
            'feature_fraction': 'colsample_bytree'
        }
        
        # Reset parameters to their aliases to suppress LGBM warnings
        for param, alias in aliases.items(): 
            if param in best_params:
                best_params[alias] = best_params[param]
                del best_params[param]
    else:
        raise ValueError(f'Unsupported model type: {model_type}')

    model.fit(X_train, y_train)
    dump(model, output_filepath)
    logging.info(f'Exported model to {output_filepath}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
