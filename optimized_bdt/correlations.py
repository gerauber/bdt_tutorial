#############################
## author: G. Räuber, 2024 ##
#############################

import argparse
import pandas as pd
import numpy as np
import json
import pickle
import os.path
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', default='../dataset/dataset_2classes',
                    type=str, help='Name of input file')
parser.add_argument('-j', '--json', default='inputs', type=str,
                    help='Json file containing the inputs')
parser.add_argument('-o', '--outname', default='corr_mat', type=str,
                    help='Name of onput files')
args = parser.parse_args()

# Read the Dataframe:
df = pd.read_pickle(f'{args.filename}.pkl')

# Create two different classes, based on the datasets
df_cat1 = df.query('dataset==1').reset_index(drop=True)
df_cat2 = df.query('dataset==2').reset_index(drop=True)

# Delete the original dataframe to release memory
del df

# Collect the inputs
with open(f'{args.json}.json') as json_file:
    inputs = json.load(json_file)


# Compute correlation matrix
def mean_weight(x, w):
    return np.sum(x*w)/np.sum(w)


def cov_weight(x, y, w):
    return np.sum(w*(x-mean_weight(x, w))*(y-mean_weight(y, w)))/np.sum(w)


def corr_weight(x, y, w):
    return cov_weight(x, y, w)/np.sqrt(cov_weight(x, x, w)*cov_weight(y, y, w))


def compute_corrmat_weighted(df, weights=[0]):
    """
    Parameters:
    - df: pandas DataFrame
    - weights: pandas Series or DataFrame containing weights for each row
    """

    if len(weights) == 0:
        output = df.corr(method='pearson')
    else:
        n_columns = df.shape[1]
        corr_matrix = np.zeros((n_columns, n_columns))

        for i in range(n_columns):
            for j in range(i, n_columns):
                x = df.iloc[:, i]
                y = df.iloc[:, j]
                w = weights

                corr_matrix[i, j] = corr_weight(x, y, w)
                corr_matrix[j, i] = corr_matrix[i, j]  # Symmetric matrix
        output = pd.DataFrame(corr_matrix, columns=df.columns,
                              index=df.columns)
    return output


# Set the weights
df_cat1_weights = df_cat1['weight']
df_cat2_weights = df_cat2['weight']

# Set the features
df_cat1 = df_cat1[inputs['features']]
df_cat2 = df_cat2[inputs['features']]

# Sort the columns for the plot
df_cat1 = df_cat1.reindex(sorted(df_cat1.columns), axis=1)
df_cat2 = df_cat2.reindex(sorted(df_cat2.columns), axis=1)

# Compute the correlation matrices
mat_cat1 = compute_corrmat_weighted(df_cat1, df_cat1_weights)
mat_cat2 = compute_corrmat_weighted(df_cat2, df_cat2_weights)

# Determine highly correlated features
# Set a limit
lim = 0.95

# Put the result in a dictionary
dict_values = {}
cats = ['cat1', 'cat2']
for ca in cats:
    dict_values[ca] = {}

for mat in [mat_cat1, mat_cat2]:
    set_features = 0
    for num, i in enumerate(mat.columns):
        for j in mat.columns[:num]:
            if abs(mat[i][j]) > lim:
                dict_values['cat1'][f'{set_features}'] = {}
                dict_values['cat1'][f'{set_features}']['F1'] = i
                dict_values['cat1'][f'{set_features}']['F2'] = j
                dict_values['cat1'][f'{set_features}']['val'] = mat[i][j]

# Save the dictionary
with open(f"{args.outname}.json", 'w') as json_file:
    json.dump(dict_values, json_file, indent=4)

# Call the color map
if os.path.exists('colours.pkl'):
    with open('colours.pkl', 'rb') as f:
        custom_cmap = pickle.load(f)
else:
    custom_cmap = 'coolwarm'

# Plot the correlation matrices
def correlation_matrix(corrmat, filename, plot_title='', cbaryesno=True,
                       all_fontsize=30, in_fontsize=20,
                       limit_val=0.01):

    for el in corrmat:
        corrmat = corrmat.rename(columns={el: el}, index={el: el})

    list_var = list(corrmat.columns)

    fig, ax = plt.subplots(figsize=(len(list_var)*1.3, len(list_var)*0.7))

    if limit_val == 0:
        fig = sns.heatmap(corrmat,
                          annot=True, fmt='.2f', vmin=-1, vmax=1, center=0,
                          linewidths=1, cmap=custom_cmap, ax=ax,
                          annot_kws={'size': in_fontsize},
                          cbar_kws={"orientation": "vertical", "pad": 0.01},
                          cbar=cbaryesno)
    else:
        fig = sns.heatmap(corrmat, mask=abs(corrmat) < limit_val,
                          annot=True, fmt='.2f', vmin=-1, vmax=1, center=0,
                          linewidths=1, cmap=custom_cmap, ax=ax,
                          annot_kws={'size': in_fontsize},
                          cbar_kws={"orientation": "vertical", "pad": 0.01},
                          cbar=cbaryesno)

        sns.heatmap(corrmat, mask=abs(corrmat) >= limit_val, annot=False,
                    fmt='.2f', vmin=-1, vmax=1, center=0,
                    linewidths=1, cmap=custom_cmap, cbar=False)

    if cbaryesno:
        cbar = fig.collections[0].colorbar
        cbar.ax.tick_params(labelsize=all_fontsize)

    ytick = [0.5+k for k in range(len(list_var))]

    fig.set_yticks(ticks=ytick)
    fig.set_xticklabels(list_var, fontsize=all_fontsize)
    fig.set_yticklabels(list_var, fontsize=all_fontsize)

    ax.set_xticklabels(list_var, rotation=30, fontsize=all_fontsize,
                       rotation_mode='anchor', ha='right')

    ax.set_title(plot_title, loc='center', y=1.02, fontsize=all_fontsize)

    if filename:
        plt.savefig(f"{filename}.pdf", bbox_inches='tight')


# Plots
correlation_matrix(mat_cat1, f'{args.outname}_cat1', 'Cat 1',
                   cbaryesno=True)
correlation_matrix(mat_cat2, f'{args.outname}_cat2', 'Cat 2',
                   cbaryesno=True)
correlation_matrix(mat_cat1, f'{args.outname}_cat1_no_cbar', 'Cat 1',
                   in_fontsize=23, cbaryesno=False)
correlation_matrix(mat_cat2, f'{args.outname}_cat2_no_cbar', 'Cat 2',
                   in_fontsize=23, cbaryesno=False)
