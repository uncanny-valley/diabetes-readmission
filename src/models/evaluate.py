import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from itertools import product
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, f1_score, classification_report,
    roc_curve, auc, precision_recall_curve)
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import plot_confusion_matrix


def plot_roc_auc_curve(y_pred_proba, y_true, ax=None, title=None):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = auc(fpr, tpr)

    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(x=fpr, y=tpr, color='navy')
    ax.plot(fpr, tpr, c='g', lw=2, label=f'AUC = {round(auc_score, 3)}')
    ax.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--', label='TPR = FPR')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_title(title)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')

# def plot_precision_recall_curve(y_pred_proba, y_true, ax=None, title=None):
#     precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
#     auc_score = auc(recall, precision)
#     print(precision, recall)

#     dummy = len(y_true[y_true == 1.]) / len(y_true)
#     if ax is None:
#         fig, ax = plt.subplots()
#     ax.scatter(x=recall, y=precision, color='navy')
#     ax.plot(recall, precision, c='g', lw=2, label=f'AUC = {round(auc_score, 3)}')
#     ax.plot([0, 1], [dummy, dummy], linestyle='--', label='Dummy classifier')
#     ax.set_xlim([0.0, 1.0])
#     ax.set_ylim([0.0, 1.05])
#     ax.set_title(title)
#     ax.set_xlabel('Recall')
#     ax.set_ylabel('Precision')
#     ax.legend(loc='upper right')


def compare_classification_reports(models, X_test, y_true):
    classes = np.unique(y_true)
    columns = ['name', 'class_label', 'precision', 'recall', 'f1-score', 'support']
    per_target_metrics = pd.DataFrame(np.empty((len(models) * len(classes), len(columns))), columns=columns)
    accuracy_table = pd.DataFrame(columns=['name', 'accuracy']).set_index('name')
    averages_table = pd.DataFrame(columns=['name', 'average_type', 'precision', 'recall', 'f1-score', 'support']).set_index(['name', 'average_type'])

    for (name, model) in models:
        y_pred = model.predict(X_test)
        report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        accuracy_table.loc[name, 'accuracy'] = report['accuracy']

        for m in ['precision', 'recall', 'f1-score', 'support']:
            averages_table.loc[(name, 'macro'), m] = report['macro avg'][m]
            averages_table.loc[(name, 'weighted'), m] = report['weighted avg'][m]


    for i, ((name, model), c) in enumerate(list(product(models, classes))):
        class_label = str(c)
        y_pred = model.predict(X_test)
        report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        per_target_metrics.iloc[i, per_target_metrics.columns.get_loc('name')] = name
        per_target_metrics.iloc[i, per_target_metrics.columns.get_loc('class_label')] = c
        per_target_metrics.iloc[i, per_target_metrics.columns.get_loc('precision')] = report[class_label]['precision']
        per_target_metrics.iloc[i, per_target_metrics.columns.get_loc('recall')] = report[class_label]['recall']
        per_target_metrics.iloc[i, per_target_metrics.columns.get_loc('f1-score')] = report[class_label]['f1-score']
        per_target_metrics.iloc[i, per_target_metrics.columns.get_loc('support')] = report[class_label]['support']

    return per_target_metrics.set_index(['name', 'class_label']), accuracy_table, averages_table

def plot_evaluation_metrics(model, X, y_true,
    axes=None, title=None, figsize=(10, 4), confusion_matrix_cmap=plt.cm.Blues):
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    class_labels = ['Early', 'Not Early']

    if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=3,
            figsize=figsize, constrained_layout=True)
    
    plot_roc_auc_curve(y_pred_proba, y_true, ax=axes[0])
    plot_precision_recall_curve(model, X, y=y_true, ax=axes[1])

    plot_confusion_matrix(
        model, X, y_true, display_labels=class_labels,
        cmap=confusion_matrix_cmap, normalize='true', ax=axes[2])
    plt.suptitle(title, fontweight='bold')