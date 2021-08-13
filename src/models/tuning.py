import datetime
import numpy as np
import optuna
import pandas as pd

from argparse import ArgumentParser
from functools import partial

from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier

import operator

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import cross_val_score

from typing import Callable, Tuple


class EarlyStoppingCallback:
    """
    Early stopping functionality from https://github.com/optuna/optuna/issues/1001
    """
    def __init__(self, early_stopping_rounds: int, direction: str='maximize'):
        self.early_stopping_rounds = early_stopping_rounds
        self._iter = 0

        if direction == 'minimize':
            self._operator = operator.lt
            self._score = np.inf
        elif direction == 'maximize':
            self._operator = operator.gt
            self._score = -np.inf
        else:
            raise ValueError(f'Unsupported direction: {direction}')

    def __call__(self, study: optuna.Study, trial: optuna.Trial):
        if self._operator(study.best_value, self._score):
            self._iter = 0
            self._score = study.best_value
        else:
            self._iter += 1

        if self._iter >= self.early_stopping_rounds:
            study.stop()


def generate_objective_function(X_train: pd.DataFrame, y_train: pd.DataFrame, model: str, scoring: str, cv: int=5, random_state: int=0):
    if model == 'random_forest':
        return partial(_random_forest_objective, X_train=X_train, y_train=y_train, scoring=scoring, cv=cv, random_state=random_state)
    elif model == 'lgbm':
        return partial(_lgbm_objective, X_train=X_train, y_train=y_train, scoring=scoring, cv=cv, random_state=random_state)
    else:
        raise ValueError(f'Unsupported objective function for (model: {model})')


def _lgbm_objective(trial: optuna.trial.Trial, X_train: pd.DataFrame, y_train: pd.Series, scoring: str, cv: int, random_state: int):
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'is_unbalance': False,
        'n_jobs': -1,
        'random_state': random_state,
        'learning_rate': trial.suggest_categorical('learning_rate', [0.001, 0.01, 0.1]),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'n_estimators': trial.suggest_int('n_estimators', 120, 1200),
        'max_depth': trial.suggest_categorical('max_depth', [5, 8, 15, 25, 30, None]),
    }

    if scoring == 'f2':
        scoring = make_scorer(fbeta_score, beta=2, zero_division=0)

    clf = LGBMClassifier(**params)
    scores = cross_val_score(clf, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1)
    return scores.mean()

def _random_forest_objective(trial: optuna.trial.Trial, X_train: pd.DataFrame, y_train: pd.Series, scoring: str, cv: int, random_state: int):
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    n_estimators = trial.suggest_int('n_estimators', 120, 1200)
    max_depth = trial.suggest_categorical('max_depth', [5, 8, 15, 25, 30, None])
    min_samples_split = trial.suggest_categorical('min_samples_split', [2, 5, 10, 15, 100])
    min_samples_leaf = trial.suggest_categorical('min_samples_leaf', [2, 5, 10])
    max_features = trial.suggest_categorical('max_features', ['log2', 'sqrt', None])

    if scoring == 'f2':
        scoring = make_scorer(fbeta_score, beta=2, zero_division=0)

    clf = RandomForestClassifier(
        criterion=criterion,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state)

    scores = cross_val_score(clf, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1)
    return scores.mean()

def run_study(objective: Callable, name: str=None, direction: str='maximize', n_trials:int=100, early_stopping_rounds:int=25, storage:str=None) -> optuna.study.Study:
    study = optuna.create_study(study_name=name, direction=direction, storage=storage)
    study.optimize(objective, callbacks=[EarlyStoppingCallback(early_stopping_rounds, direction)], n_trials=n_trials)
    return study

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cv', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', choices=['random_forest', 'lgbm'], default='random_forest')
    parser.add_argument('--dataset', choices=['original', 'rfecv', 'pruned'], default='original')
    parser.add_argument('--scoring', type=str, default='f2')
    parser.add_argument('--num_trials', type=int, default=100)
    parser.add_argument('--study-name', type=str)
    parser.add_argument('--early-stopping-rounds', type=int, default=25)
    args = parser.parse_args()

    if args.dataset == 'original':
        filename = 'data/processed/train.pkl'
    elif args.dataset == 'rfecv':
        filename = 'data/processed/train_rfecv.pkl'
    elif args.dataset == 'pruned':
        filename = 'data/processed/train_pruned.pkl'

    training_data = pd.read_pickle(filename)
    X_train = training_data.drop(columns=['is_readmitted_early'])
    y_train = training_data.is_readmitted_early

    random_undersampler = RandomUnderSampler(random_state=args.seed)
    X_train_resampled, y_train_resampled = random_undersampler.fit_resample(X_train, y_train)
    
    objective = generate_objective_function(X_train_resampled, y_train_resampled, model=args.model, scoring=args.scoring, cv=args.cv, random_state=args.seed)

    if args.study_name is not None:
        name = args.study_name
    else:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        name = f'{args.model}-{timestamp}'

    study = run_study(objective, name=name, n_trials=args.num_trials, early_stopping_rounds=args.early_stopping_rounds, storage=f'sqlite:///studies/{name}.db')
    print(study.best_value, study.best_params)