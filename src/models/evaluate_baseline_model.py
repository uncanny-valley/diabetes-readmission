import click
import logging
import numpy as np
from dotenv import find_dotenv, load_dotenv
from lightgbm import LGBMClassifier
import pandas as pd
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from string import Template
from typing import List, Tuple
from xgboost import XGBClassifier

import matplotlib.pyplot as plt

from src.data.load_dataset import load_interim_pickle, load_data
from src.models.evaluate import generate_classification_report_from_model, plot_roc_with_trained_model, plot_mean_roc, plot_feature_importance


RANDOM_SEED = 0
SAMPLERS = ['initial_data', 'smote', 'over_sampled', 'under_sampled']
BASELINE_MODELS = [
    DummyClassifier(strategy='most_frequent'),
    LogisticRegression(solver='lbfgs', random_state=RANDOM_SEED, max_iter=1000, n_jobs=-1),
    GaussianNB(),
    DecisionTreeClassifier(random_state=RANDOM_SEED),
    RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
    GradientBoostingClassifier(random_state=RANDOM_SEED),
    LGBMClassifier(random_state=RANDOM_SEED),
    XGBClassifier(random_state=RANDOM_SEED, objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')
]

@click.command()
@click.argument('cross_validation_file_template', type=str)
@click.argument('n_splits', type=int)
def main(cross_validation_file_template: str, n_splits: int) -> None:
    t = Template(cross_validation_file_template)
    total_results = pd.DataFrame([])

    for model in BASELINE_MODELS:
        model_results = pd.DataFrame([])

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        mean_fpr = np.linspace(0, 1, 100)

        for i, sampler in enumerate(SAMPLERS):
            logging.info(f'Evaluating {model.__class__.__name__} and {sampler} over {n_splits} folds')
            reports = []
            tprs = []
            aucs = []
            ax = axes[i]

            for fold in range(n_splits):
                logging.info(f'Fold {fold}')
                X_train = load_interim_pickle(t.substitute(data='X', type='train', sampler=sampler, fold=fold))
                y_train = load_interim_pickle(t.substitute(data='y', type='train', sampler=sampler, fold=fold))
                X_val = load_interim_pickle(t.substitute(data='X', type='validation', sampler=sampler, fold=fold))
                y_val = load_interim_pickle(t.substitute(data='y', type='validation', sampler=sampler, fold=fold))

                model.fit(X_train, y_train)
                reports.append(generate_classification_report_from_model(model, X_train, y_train, X_val, y_val, index=sampler))

                tpr, roc_auc = plot_roc_with_trained_model(model, X_val, y_val, mean_fpr, fold=fold, ax=ax)
                tprs.append(tpr)
                aucs.append(roc_auc)

            plot_mean_roc(model, tprs, aucs, mean_fpr, sampling_method=sampler, ax=ax)
            
            sampler_results = pd.concat(reports).groupby(level=0).mean()
            sampler_results.columns = [f'Average {col}' for col in sampler_results.columns]
            sampler_results['model'] = model.__class__.__name__
            sampler_results = sampler_results.set_index(['model', sampler_results.index])

            model_results = pd.concat([model_results, sampler_results], axis=0)
            model_results.to_pickle(f'metrics/baseline/baseline_results_{model.__class__.__name__}.pkl')
        
        fig.tight_layout()
        fig.savefig(rf'metrics/baseline/roc_auc_plot_{model.__class__.__name__}.png')
        total_results = pd.concat([total_results, model_results], axis=0)

    mean_scores_by_sampler = total_results.groupby(level=1).mean()
    mean_scores_by_sampler.to_pickle('metrics/baseline/mean_scores_by_sampler.pkl')
    total_results.to_pickle('metrics/baseline/baseline_model_total_results.pkl')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()