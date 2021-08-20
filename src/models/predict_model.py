# -*- coding: utf-8 -*-
import click
import logging
import numpy as np
import pandas as pd
from os import path
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import optuna

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from joblib import load
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, plot_confusion_matrix, accuracy_score, precision_recall_curve

from src.data.load_dataset import load_data, split_dataset
from src.models.evaluate import plot_roc_with_optimum


def plot_cost_curve(y_pred_proba, targets, thresholds, cost_function, ax=None, title=None):
    total_costs = []
    
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype('int')
        total_cost = cost_function(targets, y_pred)
        total_costs.append(total_cost)

    optimal_threshold_index = np.argmin(total_costs)

    ax.plot(thresholds, total_costs)
    ax.plot(thresholds[optimal_threshold_index], total_costs[optimal_threshold_index], color='black', marker='o', label=f'Optimal threshold: {round(thresholds[optimal_threshold_index], 3)}, cost: ${round(total_costs[optimal_threshold_index], 3)}')
    ax.set_title(title if title is not None else 'Estimated Total Operational Cost by Varying Decision Threshold')
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Estimated Total Operational Cost')
    ax.legend()

def accuracy_by_threshold(y_test, probs, threshold):
    y_pred = probs.copy()
    y_pred = (y_pred >= threshold).astype('int')
    return accuracy_score(y_test, y_pred)


@click.command()
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('test_data_path', type=click.Path(exists=True))
@click.argument('metrics_directory', type=click.Path(exists=True))
def main(model_filepath: Path, test_data_path: Path, metrics_directory: Path) -> None:
    log_fmt = '%(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt, handlers=[
        logging.FileHandler(path.join(metrics_directory, 'results.txt'), mode='w'),
        logging.StreamHandler()
    ])

    test_data = load_data(test_data_path)
    X_test, y_test = split_dataset(test_data)

    logging.info(f'Loaded test set {test_data.shape}')

    model = load(model_filepath)
    logging.info(f'Loaded model {model.__class__.__name__}')
    logging.info(f'Hyperparameters: {model.get_params()}')

    y_pred = model.predict(X_test)

    logging.info(classification_report(y_test, y_pred, target_names=['No early readmission', 'Early readmission'], zero_division=0))
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    logging.info(f'ROC AUC Score: {roc_auc}')

    fig, ax = plt.subplots()
    best_threshold_roc = plot_roc_with_optimum(model, X_test, y_test, ax=ax)
    fig.savefig(path.join(metrics_directory, 'roc.png'))

    fig, ax = plt.subplots()
    plot_confusion_matrix(model, X_test, y_test, display_labels=['Non-early', 'Early'], normalize='true', cmap=plt.cm.Blues, colorbar=False, ax=ax)
    ax.set_title(f'Confusion Matrix for {model.__class__.__name__}')
    ax.grid(False)
    fig.savefig(path.join(metrics_directory, 'confusion_matrix.png'))

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1 =  2 * (precision * recall) / (precision + recall + 1e-7)
    f2 = (5 * precision * recall) / (4 * precision + recall + 1e-7)

    best_threshold_index_f1 = np.argmax(f1)
    best_threshold_f1 = thresholds[best_threshold_index_f1]

    best_threshold_index_f2 = np.argmax(f2)
    best_threshold_f2 = thresholds[best_threshold_index_f2]

    fig, ax = plt.subplots(figsize=(20, 5))

    accuracies = [accuracy_by_threshold(y_test, y_pred_proba, t) for t in thresholds]

    ax.axvline(x=best_threshold_f1, linestyle='--', color='orange', alpha=0.5, label='Optimal threshold (f1)')
    ax.axvline(x=best_threshold_f2, linestyle='--', color='k', alpha=0.5, label='Optimal threshold (f2)')
    ax.axvline(x=best_threshold_roc, linestyle='--', color='purple', alpha=0.5, label='Optimal threshold (ROC)')
    ax.plot(thresholds, precision[:-1], label='Precision')
    ax.plot(thresholds, recall[:-1], label='Recall')
    ax.plot(thresholds, accuracies, label='Accuracy')

    ax.set_title('Precision, Recall, and Accuracy Curves by Decision Threshold')
    ax.set_ylabel('Metric Value')
    ax.set_xlabel('Decision Threshold')
    fig.legend(bbox_to_anchor=(1.01, 1))
    fig.savefig(path.join(metrics_directory, 'precision_recall_curve.png'))


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
