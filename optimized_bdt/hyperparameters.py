#############################
## author: G. Räuber, 2024 ##
#############################

import argparse
import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from lightgbm import LGBMClassifier
from bayes_opt import BayesianOptimization
import math
import warnings
from sklearn.model_selection import RandomizedSearchCV
import optuna
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', default='../dataset/dataset_2classes',
                    type=str, help='Name of input file')
parser.add_argument('-j', '--json', default='inputs', type=str,
                    help='Json file containing the inputs')
parser.add_argument('-m', '--method', default='bayesian', type=str,
                    choices=['bayesian', 'random', 'optuna'],
                    help='Method employed to optimize the hyperparameters')
parser.add_argument('-o', '--outname', default='hyperparams', type=str,
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
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=test_size,
                                                    random_state=42)

# Select weights for the bdt
X_train_we = X_train[:, cols.index(wei)]
X_test_we = X_test[:, cols.index(wei)]

# Select the features relevant for the BDT
# (not before because the events are shuffled)
X_train = X_train[:, positions]
X_test = X_test[:, positions]


if args.method == 'bayesian':
    # Bayesian optimization

    nb_fold = 5
    nb_points = 20
    nb_iter = 2

    def score_lgbdt(num_leaves, max_depth, learning_rate,
                    n_estimators, subsample_for_bin,
                    min_child_weight, min_child_samples,
                    subsample, reg_alpha, reg_lambda):
        param_space = {}
        param_space['num_leaves'] = int(num_leaves)
        param_space['max_depth'] = int(max_depth)
        param_space['learning_rate'] = learning_rate
        param_space['n_estimators'] = int(n_estimators)
        param_space['subsample_for_bin'] = int(subsample_for_bin)
        param_space['min_child_weight'] = min_child_weight
        param_space['min_child_samples'] = int(min_child_samples)
        param_space['subsample'] = subsample
        param_space['reg_alpha'] = reg_alpha
        param_space['reg_lambda'] = reg_lambda

        scores = cross_val_score(LGBMClassifier(boosting_type='gbdt',
                                                importance_type='split',
                                                num_iterations=100,
                                                **param_space),
                                 X_train, y_train,
                                 scoring='roc_auc', cv=nb_fold).mean()
        score = scores.mean()
        return score

    param_lgbdt = {
        'num_leaves': (2, 1e2),
        'max_depth': (0, 3e1),
        'learning_rate': (1e-8, 1),
        'n_estimators': (1, 5e1),
        'subsample_for_bin': (1e1, 1e4),
        'min_child_weight': (0, 1),
        'min_child_samples': (0, 1e2),
        'subsample': (1e-8, 1),
        'reg_alpha': (0, 5e1),
        'reg_lambda': (0, 5e1)
    }

    # Set the bayesian optimisation
    rd_st = np.random.randint(0, 1e3)
    print('random state:', rd_st)
    met_opt = BayesianOptimization(score_lgbdt, param_lgbdt,
                                   random_state=rd_st)

    # Run Bayesian Optimization
    met_opt.maximize(init_points=nb_points, n_iter=nb_iter)

    # Print the results
    params_fin = met_opt.max['params']
    max_value = float('%.5f' % (met_opt.max["target"]))
    results = met_opt.res

    for cs in results:
        for cat in cs:
            if cat == 'target':
                cs[cat] = float('%.5f' % (cs[cat]))
            if cat == 'params':
                for subcat in cs[cat]:
                    cs[cat][subcat] = float('%.5f' % (cs[cat][subcat]))
                    if subcat in ['num_leaves', 'max_depth', 'n_estimators',
                                  'subsample_for_bin', 'min_child_samples']:
                        cs[cat][subcat] = round(cs[cat][subcat])

    for cat in params_fin:
        params_fin[cat] = float('%.5f' % (params_fin[cat]))
        if cat in ['num_leaves', 'max_depth', 'n_estimators',
                   'subsample_for_bin', 'min_child_samples']:
            params_fin[cat] = round(params_fin[cat])

    ma = 0
    best_params = {}
    for cs in results:
        if not math.isnan(cs['target']):
            if cs['target'] > ma:
                ma = cs['target']
                best_params = cs['params']

    max_value = ma
    params_fin = best_params

    # Add an entry to copy and paste the hyperparameters more easily
    pars = "boosting_type='gbdt', importance_type='split'"
    for elem in params_fin:
        pars += ", "+str(elem)+"="+str(params_fin[elem])

    hpandscore = {}
    hpandscore['best_score'] = max_value
    hpandscore['best_estimate'] = params_fin
    hpandscore['copypaste'] = pars

    # Save the hyperparameters in a json file
    with open(f'{args.outname}_bayesian.json', 'w') as outfiles:
        json.dump(hpandscore, outfiles, indent=4)

if args.method == 'random':
    # Random search

    method = LGBMClassifier()
    param_space = {'num_leaves': [2, 20, 100],
                   'max_depth': [1, 10, 30],
                   'learning_rate': [0.005, 0.1, 0.9],
                   'n_estimators': [5, 30],
                   'subsample_for_bin': [10, 1000],
                   'min_child_weight': [0.01, 0.2, 1],
                   'min_child_samples': [5, 100],
                   'subsample': [0.001, 0.5],
                   'reg_alpha': [0, 20],
                   'reg_lambda': [0, 20]
                   }

    nb_fold = 3
    n_iter = 10

    param_rand_search = RandomizedSearchCV(method, param_space,
                                           n_iter=n_iter,
                                           scoring="roc_auc", cv=nb_fold)

    param_rand_search.fit(X_train, y_train, sample_weight=X_train_we)

    dict_hyperparams = {}

    dict_hyperparams['best_score'] = float(param_rand_search.best_score_)
    dict_hyperparams['best_estimate'] = param_rand_search.best_params_

    pars = ""
    p = 0
    for elem in param_rand_search.best_params_:
        if p == 0:
            pars += str(elem)+"="+str(param_rand_search.best_params_[elem])
            p = 1
        pars += ", "+str(elem)+"="+str(param_rand_search.best_params_[elem])

    dict_hyperparams['copypaste'] = pars
    dict_hyperparams['top'] = {}

    params = param_rand_search.cv_results_['params']
    score = param_rand_search.cv_results_['mean_test_score']
    ranking_order = param_rand_search.cv_results_['rank_test_score']

    sorted_lists = sorted(zip(ranking_order, params, score))

    ranking_order_sort, params_sort, score_sort = zip(*sorted_lists)

    top = 5
    if top > n_iter:
        top = n_iter

    dict_hyperparams['top']['best_scores'] = score_sort[:top]
    dict_hyperparams['top']['best_estimates'] = params_sort[:top]

    # Save the hyperparameters in a json file
    with open(f'{args.outname}_random.json', 'w') as outfiles:
        json.dump(dict_hyperparams, outfiles, indent=4)


if args.method == 'optuna':
    def objective_lgbmc(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 2, 1e2),
            "max_depth": trial.suggest_int("max_depth", 0, 3e1),
            "learning_rate": trial.suggest_float("learning_rate", 1e-8,
                                                 1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 1, 5e1),
            "subsample_for_bin": trial.suggest_int("subsample_for_bin",
                                                   1e1, 1e4),
            "min_child_weight": trial.suggest_float("min_child_weight", 0, 1),
            "min_child_samples": trial.suggest_int("min_child_samples",
                                                   0, 1e2),
            "subsample": trial.suggest_float("subsample", 1e-8, 1),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5e1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5e1),
        }
        mod = LGBMClassifier(**params)
        mod.fit(X_train, y_train, sample_weight=X_train_we)
        predictions = mod.predict(X_test)
        rmse = mean_squared_error(y_test, predictions, squared=False)
        return rmse

    study = optuna.create_study(direction='maximize')
    study.optimize(objective_lgbmc, n_trials=50)

    dict_hyperparams = {}

    dict_hyperparams['best_score'] = float(study.best_value)
    dict_hyperparams['best_estimate'] = study.best_params

    pars = ""
    p = 0
    for elem in study.best_params:
        if p == 0:
            pars += str(elem)+"="+str(study.best_params[elem])
            p = 1
        pars += ", "+str(elem)+"="+str(study.best_params[elem])

    dict_hyperparams['copypaste'] = pars

    # Save the hyperparameters in a json file
    with open(f'{args.outname}_optuna.json', 'w') as outfiles:
        json.dump(dict_hyperparams, outfiles, indent=4)
