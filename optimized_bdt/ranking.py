#############################
## author: G. Räuber, 2024 ##
#############################

import argparse
import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from matplotlib.ticker import MultipleLocator


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', default='../dataset/dataset_2classes',
                    type=str, help='Name of input file')
parser.add_argument('-j', '--json', default='inputs', type=str,
                    help='Json file containing the inputs')
parser.add_argument('-m', '--method', default='rfe', type=str,
                    choices=['basic_rank', 'advanced_rank', 'rfe'],
                    help='Method employed to rank the features')
parser.add_argument('-o', '--outname', default='rank', type=str,
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

# Get features' positions from selection
cols = list(df_cat1.columns)
positions = []
for i, fe in enumerate(inputs['features']):
    if fe in cols:
        positions.append(i)

# Set the weight column
wei = 'weight'

# Set the test set size to 30%
test_size = 0.3

# Put together your samples
X = np.concatenate((df_cat1, df_cat2))

# Generate a column of 0 and 1, to distinguish the classes
y = np.concatenate((np.ones(df_cat1.shape[0]), np.zeros(df_cat2.shape[0])))

# Definition of the training / test samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                    random_state=42)

# Select weights for the bdt
X_train_we = X_train[:, cols.index(wei)]
X_test_we = X_test[:, cols.index(wei)]

# Select the features relevant for the BDT
# (not before because the events are shuffled)
X_train = X_train[:, positions]
X_test = X_test[:, positions]

# Choose the method
method = LGBMClassifier()

if args.method == 'basic_rank':

    # Fit the dataset
    method.fit(X_train, y_train, sample_weight=X_train_we)

    # Get importance
    importance = method.feature_importances_
    sorted_importance = sorted(list(zip(importance, inputs['features'])),
                               reverse=True)

    # Put the result in a dictionary
    dict_values = {}
    for i, v in sorted_importance:
        dict_values[v] = float(i)

    # Save the dictionary
    with open(f"{args.outname}_basic.json", 'w') as json_file:
        json.dump(dict_values, json_file, indent=4)

    # Plot the result
    def barplot_feat(val_notsort, feat_notsort, col,
                     filename, xlab='Feature importance',
                     plots_size=(10, 8), all_fontsize=30):

        fig, ax = plt.subplots(figsize=plots_size)

        val, feat = (list(t) for t in zip(*sorted(zip(val_notsort,
                                                      feat_notsort))))

        feat_lat = [f for f in feat]

        plot = plt.barh(feat_lat, val, color=col)

        ax.tick_params(axis='y', labelsize=all_fontsize)
        ax.tick_params(axis='x', labelsize=all_fontsize)

        ax.set_xlabel(xlab, fontsize=all_fontsize)

        ax.grid(which='major', axis='both')
        ax.set_axisbelow(True)

        ax.margins(y=0)

        if filename:
            plt.savefig(f"{filename}.pdf", bbox_inches='tight')

    barplot_feat(list(dict_values.values()), list(dict_values.keys()),
                 '#8abd66c2', f"{args.outname}_basic")

if args.method == 'advanced_rank':

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    output = cross_validate(method, X_train, y_train, cv=cv,
                            scoring='roc_auc', return_estimator=True)

    att = {}
    for idx, estimator in enumerate(output['estimator']):
        print("Features sorted by their score for estimator {}:".format(idx))
        feat_imp = pd.DataFrame(estimator.feature_importances_,
                                index=inputs['features'],
                                columns=['importance'])
        feat_imp = feat_imp.sort_values('importance', ascending=False)
        print(feat_imp)
        att['importance_'+str(idx)] = {}
        att['importance_'+str(idx)] = feat_imp.to_dict()['importance']

    # Save the dictionary
    with open(f"{args.outname}_advanced.json", 'w') as json_file:
        json.dump(att, json_file, indent=4)

    results = {}
    for i in ['sum', 'min', 'max']:
        results[i] = {}

    for i, im in enumerate(att):
        list_k = list(att[im].keys())
        for k in list_k:
            if i == 0:
                results['sum'][k] = 0
                results['min'][k] = att[im][k]
                results['max'][k] = att[im][k]
            results['sum'][k] += att[im][k]/len(att)
            if att[im][k] < results['min'][k]:
                results['min'][k] = att[im][k]
            if att[im][k] > results['max'][k]:
                results['max'][k] = att[im][k]

    err_min = [list(results['sum'].values())[i] -
               list(results['min'].values())[i]
               for i in range(len(list(results['sum'].values())))]
    err_max = [list(results['max'].values())[i] -
               list(results['sum'].values())[i]
               for i in range(len(list(results['sum'].values())))]

    def barplot_feat_advanced(val_notsort, feat_notsort, col, filename,
                              xlab='Feature importance',
                              plots_size=(10, 8), all_fontsize=30,
                              err_min=None, err_plu=None):

        fig, ax = plt.subplots(figsize=plots_size)

        # Sort the values and features
        val, feat = (list(t) for t in zip(*sorted(zip(val_notsort,
                                                      feat_notsort))))

        feat_lat = [f for f in feat]

        # If error bars are provided, zip them similarly
        if err_min is not None and err_plu is not None:
            _, err_min_sort, err_plu_sort = zip(*sorted(zip(val_notsort,
                                                            err_min, err_plu)))
            error_bars = [err_min_sort, err_plu_sort]
        else:
            error_bars = None

        # Plot with error bars if they are provided
        error_kw = {'capsize': 10,
                    'color': 'k'}
        plot = plt.barh(feat_lat, val, color=col,
                        xerr=error_bars, error_kw=error_kw)

        # Setting tick parameters
        ax.tick_params(axis='y', labelsize=all_fontsize)
        ax.tick_params(axis='x', labelsize=all_fontsize)

        # Set x-label
        ax.set_xlabel(xlab, fontsize=all_fontsize)

        # Add grid
        ax.grid(which='major', axis='both')
        ax.set_axisbelow(True)

        # Adjust margins
        ax.margins(y=0)

        # Save the figure if filename is provided
        if filename:
            plt.savefig(f"{filename}.pdf", bbox_inches='tight')

    barplot_feat_advanced(list(results['sum'].values()),
                          list(results['sum'].keys()),
                          '#8abd66c2', f"{args.outname}_advanced",
                          err_min=err_min, err_plu=err_max)

if args.method == 'rfe':

    # Establish the optimal number of features
    # RFE method
    rfecv = RFECV(estimator=method)
    pipeline = Pipeline([('Feature Selection', rfecv), ('Model', method)])

    # Use of 5 splits and 3 repeats
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

    # Cross_val_score
    n_scores = cross_val_score(pipeline, X_train, y_train,
                               scoring='roc_auc', cv=cv, n_jobs=-1)
    np.mean(n_scores)

    # Fit
    pipeline.fit(X_train, y_train, **{'Model__sample_weight': X_train_we})

    print(f"Optimal number of features: {rfecv.n_features_} \
            out of {len(inputs['features'])}")

    print('AUC scores and nb of features:')
    dict_scores = {}
    dict_scores['nb_feat_selected'] = float(rfecv.n_features_)
    for k, aucval in enumerate(rfecv.grid_scores_):
        print(f'{k:<2} feat.: {aucval:<0.5f}')
        dict_scores[str(k)] = float(aucval)

    # Save values in a json file
    with open(f"{args.outname}_rfecv.json", 'w') as json_file:
        json.dump(dict_scores, json_file, indent=4)

    # Establish which features are selected
    print("establish which features are selected ...")
    numberfound = int(rfecv.n_features_)

    # RFE method
    rfe = RFE(estimator=method, n_features_to_select=numberfound)
    pipe = Pipeline([('Feature Selection', rfe), ('Model', method)])

    # Use of 5 splits and 3 repeats
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

    # Cross_val_score
    n_scores = cross_val_score(pipe, X_train, y_train,
                               scoring='roc_auc', cv=cv, n_jobs=-1)
    np.mean(n_scores)

    # Fit
    pipe.fit(X_train, y_train, **{'Model__sample_weight': X_train_we})

    # Rank of features, not sorted
    selectedfeat = {}
    for i in range(len(X_train[0])):
        print(f'{inputs["features"][i]+",":<40} Selected: \
                {str(rfe.support_[i])+",":<15} Rank: {rfe.ranking_[i]}')
        selectedfeat[inputs['features'][i]] = int(rfe.ranking_[i])

    # Save values in a json file
    with open(f"{args.outname}_rfe.json", 'w') as json_file:
        json.dump(selectedfeat, json_file, indent=4)

    # Plot the result
    def plot_rfe_result(val, maxauc, legloc,
                        filename, xlab="Number of features",
                        ylab="Cross validation score", plots_size=(10, 6),
                        all_fontsize=25, ylim=[0, 1], yloc=0.2):

        fig, ax = plt.subplots(figsize=plots_size)

        plt.xlabel(xlab, fontsize=all_fontsize)
        plt.ylabel(ylab, fontsize=all_fontsize)
        plt.plot(range(1, len(val)+1), val, label="AUC score", marker='o',
                 linestyle='-', linewidth=2, markersize=4, color='k')
        plt.vlines(maxauc, 0, 1, color='teal', linewidth=2,
                   linestyles='dashed', label='RFE choice')

        plt.legend(loc=legloc, fontsize=all_fontsize)

        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(yloc))

        ax.grid(which='major', axis='both')

        plt.xticks(fontsize=all_fontsize)
        plt.yticks(fontsize=all_fontsize)
        ax.set_ylim(ylim)
        ax.set_xlim([0, len(val)+1])

        if filename:
            plt.savefig(f"{filename}.pdf", bbox_inches='tight')

    plot_rfe_result(rfecv.grid_scores_, rfecv.n_features_, "lower left",
                    f"{args.outname}_rfe",
                    plots_size=(10, 6))

    plot_rfe_result(rfecv.grid_scores_, rfecv.n_features_, "upper left",
                    f"{args.outname}_rfe_zoom",
                    plots_size=(10, 6), ylim=[0.6, 0.9], yloc=0.05)
