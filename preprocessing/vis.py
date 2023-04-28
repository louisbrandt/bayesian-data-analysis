import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_corr(df):
    fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(15,10))

    sns.regplot(x='tempmax', y='revenue', data=df, ax=axs[0][0],scatter_kws={"color": "blue"}, line_kws={"color": "red"})
    sns.regplot(x='humidity', y='revenue', data=df, ax=axs[0][1],scatter_kws={"color": "blue"}, line_kws={"color": "red"})
    sns.boxplot(x='precip', y='revenue', data=df, ax=axs[0][2])
    sns.boxplot(x='n_stores', y='revenue', data=df, ax=axs[0][3])
    sns.boxplot(x='dow', y='revenue', data=df, ax=axs[1][0])
    sns.boxplot(x='day', y='revenue', data=df, ax=axs[1][1])
    sns.boxplot(x='month', y='revenue', data=df, ax=axs[1][2])
    sns.boxplot(x='year', y='revenue', data=df, ax=axs[1][3])

    plt.tight_layout()
    plt.show()

def plot_cat(df,var):
    fig = plt.figure(figsize=(12,6))
    sns.boxplot(x=var, y='revenue', data=df)
    return fig

def plot_num(df,var):
    fig = plt.figure(figsize=(12,6))
    sns.regplot(x=var, y='revenue', data=df,scatter_kws={"color": "blue"}, line_kws={"color": "red"})
    return fig

if __name__ == '__main__':
    df = pd.read_csv('/Users/louisbrandt/itu/6/bachelor/data/deploy/processed_data.csv')

    # make a subplot for each variable
    for var in ['tempmax','humidity','cloudcover','windspeed']:
        fig = plot_num(df,var)
        fig.savefig('plots/'+var+'.png')

    for var in ['precip','n_stores','dow','day','month','year']:
        fig = plot_cat(df,var)
        fig.savefig('plots/'+var+'.png')

