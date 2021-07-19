import matplotlib.pyplot as plt
from matplotlib.axes import SubplotBase
import numpy as np
import pandas as pd

from imblearn.base import BaseSampler
from itertools import product

from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, f1_score, classification_report, precision_recall_fscore_support,
    roc_curve, auc, precision_recall_curve)
from sklearn.metrics import plot_precision_recall_curve, plot_confusion_matrix, plot_roc_curve
from sklearn.model_selection import StratifiedKFold

from time import time
from typing import List, Tuple, Generator


# Alias for k-fold model-sample combination (estimator, X_train, y_train, X_val, y_val)
ModelSamplePair = Tuple[BaseEstimator, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]


def kfold_samples(skf: StratifiedKFold, X: pd.DataFrame, y: pd.Series, sampling_methods: List[Tuple[str, BaseSampler]]) -> List:
    samples = []
    for (name, sampler) in sampling_methods:
        for train_indices, validation_indices in skf.split(X, y):
            xt, yt = X.iloc[train_indices], y.iloc[train_indices]
            xv, yv = X.iloc[validation_indices], y.iloc[validation_indices]
            
            if sampler is not None:
                xt, yt = sampler.fit_resample(xt, yt)
                
            samples.append((name, xt, yt, xv, yv))

    return samples


def train_model_on_sample(estimator: BaseEstimator, name: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> ModelSamplePair: 
    model = clone(estimator).fit(X_train, y_train)
    model.__sample__ = name
    return (model, X_train, y_train, X_val, y_val)

def train_models_over_samples(estimator: BaseEstimator, samples: List[Tuple[str, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]) -> List:
    return Parallel(n_jobs=-1, prefer='threads')(delayed(train_model_on_sample)(estimator, name, xt, yt, xv, yv) for (name, xt, yt, xv, yv) in samples)

def classification_report_from_models(model_sample_pairs: List[ModelSamplePair]) -> pd.DataFrame:
    reports = Parallel(n_jobs=-1, prefer='threads')(delayed(generate_classification_report_from_model)(estimator, xt, yt, xv, yv, estimator.__sample__) for estimator, xt, yt, xv, yv in model_sample_pairs)
    return pd.concat(reports).groupby(level=0).mean()

def classification_report_over_samples(estimator: BaseEstimator, samples: List[Tuple[str, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]) -> pd.DataFrame:
    reports = [generate_classification_report(xt, yt, xv, yv, estimator, name) for (name, xt, yt, xv, yv) in samples]
    return pd.concat(reports).groupby(level=0).mean()

def generate_classification_report_from_model(estimator: BaseEstimator, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, name:str, pos_label=1, neg_label=0) -> pd.DataFrame:
    y_pred = estimator.predict(X_val)
    precisions, recalls, f2s, _ = precision_recall_fscore_support(y_val, y_pred, beta=2.0, zero_division=0)
    precisions = [precisions[pos_label]]
    recalls = [recalls[pos_label]]
    f2s = [f2s[pos_label]]
    counts = y_train.value_counts()
    probs = estimator.predict_proba(X_val)
    auc = roc_auc_score(y_val, probs[:, pos_label])
    return pd.DataFrame({'Average Precision': precisions, 'Average Recall': recalls, 'Average f2-score': f2s,
                         'Average ROC AUC': auc, 'Average # of Positive Obs.': counts[pos_label],
                         'Average # of Negative Obs.': counts[neg_label], 'Average Proportion of Positive Obs.': counts[pos_label]/(counts[pos_label] + counts[neg_label])}) \
                         .rename(index={0:name})

def generate_classification_report(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, estimator: BaseEstimator, name: str, pos_label=1, neg_label=0) -> pd.DataFrame:
    estimator.fit(X_train, y_train)
    generate_classification_report_from_model(estimator, X_train, y_train, X_val, y_val, name, pos_label=pos_label, neg_label=neg_label)


def plot_roc_over_models(model_sample_pairs: List[ModelSamplePair], ax: SubplotBase=None):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    if not ax:
        _, ax = plt.subplots()

    for i, (estimator, _, _, xv, yv) in enumerate(model_sample_pairs):
        viz = plot_roc_curve(estimator, xv, yv,
                name='ROC fold {}'.format(i),
                alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title=f'Mean ROC for {estimator.__class__.__name__} (folds={len(model_sample_pairs)}, sampling={estimator.__sample__})')
    ax.legend(loc="lower right")   


def plot_roc_over_folds(X: pd.DataFrame, y: pd.Series, skf: StratifiedKFold, estimator: BaseEstimator, sampler: BaseSampler, ax: SubplotBase=None):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    if not ax:
        _, ax = plt.subplots()

    for i, (train_indices, validation_indices) in enumerate(skf.split(X, y)):
        xt, yt = X.iloc[train_indices], y.iloc[train_indices]
        xv, yv = X.iloc[validation_indices], y.iloc[validation_indices]

        if sampler is not None:
            xt, yt = sampler.fit_resample(xt, yt)

        estimator.fit(xt, yt)
        viz = plot_roc_curve(estimator, xv, yv,
                            name='ROC fold {}'.format(i),
                            alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title=f'Mean ROC for {estimator.__class__.__name__} (folds={skf.n_splits})')
    ax.legend(loc="lower right")

def plot_feature_importance(importance: List[np.float64], feature_names: List[str], max_num:int, title:str=None, ax: SubplotBase=None):
    if not ax:
        _, ax = plt.subplots()

    indexes = np.argsort(importance)
    names = []
    feature_importance = []

    for i in indexes:
        names.append(feature_names[i])
        feature_importance.append(importance[i])

    ax.set_title(title)
    ax.barh(names[-max_num::], feature_importance[-max_num::])
    ax.set_yticklabels = names
    
