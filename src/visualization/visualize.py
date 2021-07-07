from matplotlib.axes import SubplotBase
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def pie_plot(feature_name: str, df: pd.DataFrame, ax: SubplotBase=None, title: str=None):
    title = title if title is not None else f'{feature_name} Pie Chart'
    if not ax:
        fig, ax = plt.subplots()

    _ = df[feature_name].value_counts(normalize=True).plot.pie(autopct="%.3f%%", labels=['No ER', 'ER'])
    ax.set_title(title)
    ax.set_ylabel(None)
    plt.tight_layout()

def density_bar_plot(feature_name: str, df: pd.DataFrame,  ax: SubplotBase=None, rot:int=0, title: str=None,):
    title = title if title is not None else f'{feature_name} Bar Plot'
    if not ax:
        fig, ax = plt.subplots()

    bar_plot = df[feature_name].value_counts(normalize=True).plot.bar(ax=ax, rot=rot)
    annotate_bar_plot_with_proportions(bar_plot, ax=ax, fontsize=11)

    ax.set_title(title)
    ax.set_ylabel('density')

def stacked_bar_plot(feature_a: str, feature_b: str, df: pd.DataFrame, ax: SubplotBase=None, title: str=None):
    title = title if title is not None else f'{feature_a} vs {feature_b} Stacked Bar Plot'

    if not ax:
        fig, ax = plt.subplots()

    c = pd.crosstab(df[feature_a], df[feature_b]).apply(lambda r: r/r.sum(), axis=1)
    stacked = c.plot.bar(stacked=True, rot=0, ax=ax)

    for p in stacked.patches:
        height = p.get_height()
        stacked.text(p.get_x() + p.get_width() / 2, 
              p.get_y() + height / 2,
              "{:.3f}".format(height),
              ha='center',
              va='bottom')

    ax.set_title(title)
    ax.set_ylabel('density')

def grouped_bar_plot(feature_name: str, df: pd.DataFrame, ax: SubplotBase=None, groupby: str='is_readmitted_early', order: list=None, title: str=None, xlabels: list=[], rot:int=0):
    title = title if title is not None else f'Proportion of {feature_name}, grouped by {groupby}'
    y_colname = f'density in {groupby} group'
    grouped = df.groupby(groupby)[feature_name] \
        .value_counts(normalize=True) \
        .rename(y_colname).to_frame() \
        .reset_index()

    if not ax:
        fig, ax = plt.subplots()
        fig.set_figwidth(10)

    bar_plot = sns.barplot(x=feature_name, y=y_colname, hue=groupby, data=grouped, ax=ax, order=order)

    if xlabels:
        ax.set_xticklabels(xlabels, rotation=rot)
    else:
        labels = ax.get_xticklabels()
        ax.set_xticklabels(labels, rotation=rot)

    ax.set_title(title, pad=20)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    annotate_bar_plot_with_proportions(bar_plot, ax=ax)

def annotate_bar_plot_with_proportions(bar_plot, fontsize=9, vertical=False, ax=None):
    """
    Annotates values atop of each vertical in a bar plot.
    """
    for p in bar_plot.patches:
        if ax:
            ax.annotate(format(p.get_height(), '.3f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (2 if vertical else 0, 13 if vertical else 9),
                textcoords='offset points',
                fontsize=fontsize,
                rotation=90 if vertical else 0)
        else:
            plt.annotate(format(p.get_height(), '.3f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (2 if vertical else 0, 13 if vertical else 9),
                textcoords='offset points',
                fontsize=fontsize,
                rotation=90 if vertical else 0)


def truncate_xticklabels(plot, stop_index=30):
    """
    Truncate xtick labels with an ellipsis after `stop_index`
    """
    for label in plot.get_xticklabels():
        t = label.get_text()
        if len(t) < stop_index:
            continue
        else:
            label.set_text(t[:stop_index] + '...')
    
    return plot.get_xticklabels()

def grouped_box_plot(feature_name: str, df: pd.DataFrame, groupby: str, ax: SubplotBase=None):
    if not ax:
        fig, ax = plt.subplots()

    sns.boxplot(y=feature_name, x=groupby, data=df, ax=ax,
        showmeans=True, boxprops=dict(alpha=.7), width=.3,
        flierprops=dict(markersize=3,alpha=0.1,
        linestyle='none'))