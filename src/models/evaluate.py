import matplotlib.pyplot as plt
from matplotlib.axes import SubplotBase
import numpy as np
import pandas as pd
import seaborn as sns

from imblearn.base import BaseSampler
from itertools import product

from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import RFECV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, f1_score, classification_report, precision_recall_fscore_support,
    roc_curve, auc, precision_recall_curve)
from sklearn.metrics import plot_precision_recall_curve, plot_confusion_matrix, plot_roc_curve
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.utils.validation import check_is_fitted

from time import time
from typing import List, Tuple, Generator
import numpy.typing as npt


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
    df = pd.concat(reports).groupby(level=0).mean()
    df.columns = [f'Average {col}' for col in df.columns]
    return df

def classification_report_over_samples(estimator: BaseEstimator, samples: List[Tuple[str, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]) -> pd.DataFrame:
    reports = [generate_classification_report(xt, yt, xv, yv, estimator, index=name) for (name, xt, yt, xv, yv) in samples]
    df = pd.concat(reports).groupby(level=0).mean()
    df.columns = [f'Average {col}' for col in df.columns]
    return df


def generate_classification_report_from_model(estimator: BaseEstimator, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, index:str=None, pos_label=1, neg_label=0) -> pd.DataFrame:
    y_pred = estimator.predict(X_test)
    precisions, recalls, f2s, _ = precision_recall_fscore_support(y_test, y_pred, beta=2.0, zero_division=0)
    precisions = [precisions[pos_label]]
    recalls = [recalls[pos_label]]
    f2s = [f2s[pos_label]]
    counts = y_train.value_counts()
    probs = estimator.predict_proba(X_test)
    auc = roc_auc_score(y_test, probs[:, pos_label])
    df = pd.DataFrame({'Precision': precisions, 'Recall': recalls, 'f2-score': f2s,
                         'ROC AUC': auc, '# of Positive Obs.': counts[pos_label],
                         '# of Negative Obs.': counts[neg_label], 'Proportion of Positive Obs.': counts[pos_label]/(counts[pos_label] + counts[neg_label])})
    if index is not None:
        df = df.rename(index={0:index})
    
    return df

def generate_classification_report(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, estimator: BaseEstimator, name: str, pos_label=1, neg_label=0) -> pd.DataFrame:
    estimator.fit(X_train, y_train)
    generate_classification_report_from_model(estimator, X_train, y_train, X_val, y_val, name, pos_label=pos_label, neg_label=neg_label)


def plot_roc_with_trained_model(estimator: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series, fpr_space: npt.NDArray, fold: int=None, ax: SubplotBase=None) -> Tuple[npt.NDArray, npt.NDArray]:
    if ax is None:
        _, ax = plt.subplots()
    
    roc = plot_roc_curve(estimator, X_test, y_test, name=f'ROC fold {fold}' if fold is not None else 'ROC', alpha=0.3, lw=1, ax=ax)
    interp_tpr = np.interp(fpr_space, roc.fpr, roc.tpr)
    interp_tpr[0] = 0.0
    return interp_tpr, roc.roc_auc

def plot_mean_roc(estimator: BaseEstimator, tprs: List[float], aucs: List[float], fpr_space: npt.NDArray, sampling_method: str=None, ax: SubplotBase=None):
    if ax is None:
        _, ax = plt.subplots()

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(fpr_space, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(fpr_space, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(fpr_space, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    sampling_title = f' (sampling={sampling_method})' if sampling_method is not None else ''
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title=f'Mean ROC for {estimator.__class__.__name__}{sampling_title}')
    ax.legend(loc='lower right')  

def plot_roc_over_models(model_sample_pairs: List[ModelSamplePair], ax: SubplotBase=None):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    if not ax:
        _, ax = plt.subplots()

    for i, (estimator, _, _, xv, yv) in enumerate(model_sample_pairs):
        tpr, roc_auc = plot_roc_with_trained_model(estimator, xv, yv, fpr_space=mean_fpr, fold=i, ax=ax)
        tprs.append(tpr)
        aucs.append(roc_auc)

    plot_mean_roc(estimator, tprs, aucs, mean_fpr, sampling_method=estimator.__sample__, ax=ax)


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
    ax.legend(loc='lower right')

def get_top_features(rfecv: RFECV, X_train: pd.DataFrame, n:int) -> pd.DataFrame:
    sorted_rankings = sorted(enumerate(rfecv.ranking_), key=lambda x: x[1])
    top_n_indices = [i for i, ranking in sorted_rankings[:n]]
    return X_train.iloc[:, top_n_indices]

def plot_feature_importance(importance: List[np.float64], feature_names: List[str], max_num:int, title:str=None, ax: SubplotBase=None):
    if not ax:
        _, ax = plt.subplots()

    indexes = np.argsort(importance)
    names = []
    feature_importance = []

    for i in indexes:
        names.append(feature_names[i])
        feature_importance.append(importance[i])

    ax.set_title(title if title else f'Top {max_num} Features by Importance')
    ax.set_xlabel('Feature Importance')
    ax.barh(names[-max_num::], feature_importance[-max_num::])
    ax.set_yticklabels = names
    return np.array(feature_importance[::-1]), np.array(names[::-1])

def plot_rfecv(rfecv: RFECV, ax:SubplotBase=None):
    check_is_fitted(rfecv)

    if ax is None:
        _, ax = plt.subplots()

    sns.lineplot(x=np.arange(1, len(rfecv.grid_scores_) + 1), y=rfecv.grid_scores_, ax=ax)
    ax.set_xlabel('Number of selected features')
    ax.set_ylabel(f'CV Score ({rfecv.scoring})')
    ax.set_title(f'Recursive Feature Selection for {rfecv.estimator.__class__.__name__}')


def plot_learning_curve(estimator: BaseEstimator, X: pd.DataFrame, y: pd.Series, ax: SubplotBase=None, ylim=None, cv=5, scoring=str,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    if ax is None:
        _, ax = plt.subplots()

    ax.set_title(f'Learning curve for {estimator.__class__.__name__}')
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel('Training examples')
    ax.set_ylabel('Score')

    train_sizes, train_scores, test_scores, _, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, scoring=scoring,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ax.grid()
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color='r')
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color='g')
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    ax.legend(loc="best")


def plot_roc_with_optimum(estimator: BaseEstimator, X: pd.DataFrame, y: pd.Series, ax: SubplotBase=None):
    y_pred_proba = estimator.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)

    # Compute geometric mean of sensitivity and specificity
    g_means = np.sqrt(tpr * (1-fpr))

    best_threshold_roc_index = np.argmax(g_means)
    best_threshold_roc = thresholds[best_threshold_roc_index]
    
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
        
    plot_roc_curve(estimator, X, y, name='ROC', lw=1, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    ax.plot(fpr[best_threshold_roc_index], tpr[best_threshold_roc_index], color='black', marker='o', label=f'Optimal threshold: {round(best_threshold_roc, 3)}')
    ax.set_title(f'ROC AUC for {estimator.__class__.__name__}')
    
    ax.fill_between(fpr, tpr, hatch='.', alpha=0.25)
    
    plt.legend()
    return best_threshold_roc