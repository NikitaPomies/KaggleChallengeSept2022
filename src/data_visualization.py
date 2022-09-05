import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
pd.set_option("display.max_rows", 100, "display.max_columns", 100)


def check_data_distribution(dataf):

    plt.figure(figsize=(18, 18))
    for i, col in enumerate(dataf.select_dtypes(include=['float64']).columns):
        plt.rcParams['axes.facecolor'] = 'black'
        ax = plt.subplot(5, 5, i + 1)
        sns.histplot(data=dataf, x=col, ax=ax, color='red', kde=True)
    plt.suptitle('Data distribution of continuous variables')
    plt.tight_layout()


def heatmap_missing_values(df):

    # Heatmap of missing values
    plt.figure(figsize=(15, 8))
    sns.heatmap(df.isna().T, cmap='YlGnBu')
    plt.title('Heatmap of missing values')

def visualize_test_train_distrib(df_train, df_test):
    """
    Plots the distribution of train and test set  on top of each other.
    Taken from https://www.kaggle.com/code/ambrosm/tpsaug22-eda-which-makes-sense
    """
    float_cols = [col for col in df_test.columns if df_test[col].dtypes == 'float64']
    both = pd.concat([df_train[df_test.columns], df_test])
    _, axs = plt.subplots(4, 4, figsize=(16, 16))
    for f, ax in zip(float_cols, axs.ravel()):
        mi = min(df_train[f].min(), df_test[f].min())
        ma = max(df_train[f].max(), df_test[f].max())
        bins = np.linspace(mi, ma, 40)
        ax.hist(df_train[f], bins=bins, alpha=0.5, density=True, label='Train set')
        ax.hist(df_test[f], bins=bins, alpha=0.5, density=True, label='Test set')
        ax.set_xlabel(f)
        if ax == axs[0, 0]: ax.legend(loc='lower right')

        ax2 = ax.twinx()
        total, _ = np.histogram(df_train[f], bins=bins)
        failures, _ = np.histogram(df_train[f][df_train.failure == 1], bins=bins)
        with warnings.catch_warnings():  # ignore divide by zero for empty bins
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            ax2.scatter((bins[1:] + bins[:-1]) / 2, failures / total,
                        color='m', s=10, label='failure probability')
        ax2.set_ylim(0, 0.5)
        ax2.tick_params(axis='y', colors='m')
        if ax == axs[0, 0]: ax2.legend(loc='upper right')
    plt.tight_layout(w_pad=1)
    plt.suptitle('Train set  and Test set distributions of the continuous features', fontsize=26, y=1.02)
    plt.show()