import click
import logging
import numpy as np
import pandas as pd
from os import path
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import SubplotBase
import seaborn as sns
sns.set()

from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.data.load_dataset import load_data, load_preprocessed_pickle, split_dataset, undersample_preprocessed_data
from src.features.age import age_to_index
from src.features.icd9 import icd9_to_category
from src.models.evaluate import plot_roc_with_optimum, plot_rfecv, plot_feature_importance
from src.models.lace_index import LACEIndexClassifier
import src.visualization.visualize as viz


def generate_class_imbalance_figure(df: pd.DataFrame, export_path: str) -> None:
    _, ax = plt.subplots()
    viz.pie_plot(feature_name='is_readmitted_early', df=df, title='Dataset Class Composition')
    plt.savefig(export_path)


def generate_cumulative_variance_figure(pca: PCA, ax: SubplotBase=None):
    cumulative_variance = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)) * 100
    cumulative_variance = np.insert(cumulative_variance, 0, 0.0)

    if ax is None:
        _, ax = plt.subplots()

    sns.lineplot(x=np.arange(0, pca.n_components_ + 1), y=cumulative_variance, ax=ax)
    _ = ax.set_title('Cumulative percentage of explained variance\nby number of components')
    _ = ax.set_xlabel('Number of components')
    _ = ax.set_ylabel('Cumulative percentage of explained variance')


def generate_2d_pca_component_figure(X: pd.DataFrame, hue: pd.Series, alpha: float, ax: SubplotBase=None):
    if ax is None:
        _, ax = plt.subplots()

    _ = ax.scatter(X[:, 0], X[:, 1], c=hue, cmap='coolwarm', alpha=alpha)
    _ = ax.set_title('2-dimensional numerical data\nusing principal components')
    _ = ax.set_xlabel('Component 1')
    _ = ax.set_xticks([])
    _ = ax.set_yticks([])
    _ = ax.set_ylabel('Component 2')


def generate_pca_figures(df: pd.DataFrame, hue: pd.Series, export_path: str) -> None:
    n_components = df.shape[1]
    pca = PCA(n_components=n_components)
    pca.fit(df)
    transformed_X = pca.transform(df)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    generate_cumulative_variance_figure(pca, ax=axes[0])
    generate_2d_pca_component_figure(transformed_X, hue=hue, alpha=0.2, ax=axes[1])
    fig.tight_layout()
    fig.savefig(export_path)


def generate_grouped_box_plots(df: pd.DataFrame, hue: pd.Series, export_path: str) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes_iter = iter(axes.flatten())
    data = pd.concat([df, hue], axis=1)

    for f in df.columns:
        ax = next(axes_iter)
        viz.grouped_box_plot(feature_name=f, groupby='is_readmitted_early', df=data, ax=ax)
    fig.suptitle('Box plots of log-transformed, scaled numerical features')
    fig.tight_layout()
    fig.savefig(export_path)

def generate_grouped_discharge_bar_plot(df: pd.DataFrame, export_path: str) -> None:
    fig, ax = plt.subplots(figsize=(20, 5))

    nlargest_counts = df.discharge_disposition_id.value_counts(normalize=True).sort_values(ascending=False).nlargest(10)
    nlargest_index = nlargest_counts.index
    nlargest_labels = ['Home', 'SNF', 'Home Health Service', 'Not Mapped', 'Short-term hospital', 'Rehab', 'Inpatient care.', 'ICF', 'Left AMA', 'Long-term hospital']

    tmp = df[df.discharge_disposition_id.isin(nlargest_index)].copy()
    tmp.loc[:, 'discharge_disposition_id'] = tmp.discharge_disposition_id.cat.remove_unused_categories()
    viz.grouped_bar_plot(feature_name='discharge_disposition_id', df=tmp, ax=ax,
                        groupby='is_readmitted_early', order=nlargest_index.tolist(),
                        title='Proportion of discharge types, grouped by early readmission',
                        xlabels=nlargest_labels)
    fig.savefig(export_path)

def generate_grouped_age_bar_plot(df: pd.DataFrame, export_path: str) -> None:
    fig, ax = plt.subplots(figsize=(20, 5))
    viz.grouped_bar_plot(feature_name='age', df=df, ax=ax,
                     order=sorted(df.age.unique()),
                     groupby='is_readmitted_early',
                     title='Proportion of age, grouped by early readmission')
    fig.savefig(export_path)

def generate_grouped_primary_diagnosis_bar_plot(df: pd.DataFrame, export_path: str) -> None:
    diagnosis_categories = df.copy()
    diagnosis_categories['diag_1'] = diagnosis_categories.diag_1.apply(icd9_to_category)

    fig, ax = plt.subplots(figsize=(20, 5))
    viz.grouped_bar_plot(feature_name='diag_1', df=diagnosis_categories, ax=ax, title='Primary Diagnosis Categories, grouped by early readmission')
    fig.savefig(export_path)


def generate_lace_index_roc_plot(estimator: LACEIndexClassifier, X_test: pd.DataFrame, y_test: pd.Series, export_path: str) -> None:
    fig, ax = plt.subplots()
    plot_roc_with_optimum(estimator, X_test, y_test, ax=ax)
    fig.savefig(export_path)


def generate_lace_index_confusion_matrix(estimator: LACEIndexClassifier, X_test: pd.DataFrame, y_test: pd.Series, export_path: str) -> None:
    fig, ax = plt.subplots()
    plot_confusion_matrix(estimator, X_test, y_test, display_labels=['Non-early', 'Early'], normalize='true', cmap=plt.cm.Blues, colorbar=False, ax=ax)
    ax.grid(False)
    ax.set_title('LACE Index Confusion Matrix')
    fig.savefig(export_path)


def generate_rfecv_graphs(X_train: pd.DataFrame, y_train: pd.Series, export_path: str) -> None:
    model = RandomForestClassifier(random_state=0)
    model.fit(X_train, y_train)
    rfecv = RFECV(model, step=1, cv=StratifiedKFold(n_splits=5, random_state=0, shuffle=True), scoring='roc_auc', n_jobs=-1)
    rfecv.fit(X_train, y_train)

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    plot_rfecv(rfecv, ax=axes[0])
    plot_feature_importance(rfecv.estimator.feature_importances_, X_train.columns, max_num=15, ax=axes[1])
    fig.tight_layout()
    fig.savefig(export_path)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('figure_directory', type=click.Path(exists=True))
def main(input_filepath: Path, figure_directory: Path) -> None:
    data = load_data(input_filepath)

    training_data = load_preprocessed_pickle('train.pkl')
    X_train, y_train = split_dataset(training_data)
    X_train, y_train = undersample_preprocessed_data(X_train, y_train)

    lace_training_data = load_preprocessed_pickle('train_lace.pkl')
    lace_testing_data = load_preprocessed_pickle('test_lace.pkl')
    X_train_lace, y_train_lace = split_dataset(lace_training_data)
    X_test_lace, y_test_lace = split_dataset(lace_testing_data)
    X_train_lace, y_train_lace = undersample_preprocessed_data(X_train_lace, y_train_lace)

    lace_classifier = LACEIndexClassifier()
    lace_classifier.fit(X_train_lace, y_train_lace)

    logging.info(f'Loaded input data: {data.shape}')

    numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
    numerical_data = data[numerical_features].copy()
    numerical_data['age'] = age_to_index(data.age)
    numerical_data = numerical_data.apply(np.log1p)
    scaler = StandardScaler()
    numerical_data = pd.DataFrame(scaler.fit_transform(numerical_data), index=numerical_data.index, columns=numerical_data.columns)

    generate_class_imbalance_figure(data, path.join(figure_directory, 'class_imbalance.png'))
    generate_pca_figures(numerical_data, data.is_readmitted_early, path.join(figure_directory, 'pca.png'))
    generate_grouped_box_plots(numerical_data, data.is_readmitted_early, path.join(figure_directory, 'grouped_box_plots.png'))
    generate_grouped_discharge_bar_plot(data, path.join(figure_directory, 'discharge.png'))
    generate_grouped_age_bar_plot(data, path.join(figure_directory, 'age.png'))
    generate_grouped_primary_diagnosis_bar_plot(data, path.join(figure_directory, 'primary_diagnosis.png'))
    generate_lace_index_confusion_matrix(lace_classifier, X_test_lace, y_test_lace, path.join(figure_directory, 'lace_confusion_matrix.png'))
    generate_lace_index_roc_plot(lace_classifier, X_test_lace, y_test_lace, path.join(figure_directory, 'lace_roc.png'))
    generate_rfecv_graphs(X_train, y_train, path.join(figure_directory, 'rfecv.png'))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
