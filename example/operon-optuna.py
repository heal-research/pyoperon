import os
import numpy as np
import csv
from multiprocessing import Pool

from pyoperon.sklearn import SymbolicRegressor
import optuna
import pmlb

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# set up some configuration options
scale_x = True
scale_y = True
max_samples = int(1e4)

# instantiate a regressor

default_params = {
        'offspring_generator': 'basic',
        'initialization_method': 'btc',
        'n_threads': 4,
        'objectives':  ['r2', 'diversity'],
        'epsilon':  1e-6,
        'random_state': None,
        'reinserter': 'keep-best',
        'max_evaluations': int(1e5),
        'symbolic_mode': False,
        'tournament_size': 3,
        'pool_size': None,
        'initialization_max_length': 10,
        }


# define parameter distributions
param = {
            'optimizer_iterations' : optuna.distributions.IntDistribution(low=0, high=10, step=1),
            'allowed_symbols' : optuna.distributions.CategoricalDistribution(['add,sub,mul,aq,constant,variable', 'add,sub,mul,aq,sin,cos,exp,logabs,sqrtabs,tanh,constant,variable']),
            'population_size' : optuna.distributions.IntDistribution(low=100, high=1000, step=100),
            'max_length': optuna.distributions.IntDistribution(low=10, high=50, step=10)
        }

# perform a number of reps with the best parameters for each problem
reps = 20

def scale(X_train, X_test, y_train):
    # optionally scale the data
    sc_X = None
    sc_y = None
    if scale_x:
        sc_X = StandardScaler()
        X_train_scaled = sc_X.fit_transform(X_train)
        X_test_scaled = sc_X.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    if scale_y:
        sc_y = StandardScaler()
        y_train_scaled = sc_y.fit_transform(y_train.reshape(-1,1)).flatten()
    else:
        y_train_scaled = y_train

    return X_train_scaled, X_test_scaled, y_train_scaled, sc_X, sc_y


def optimize(name, X, y, scale_x, scale_y, reg):
    study = optuna.create_study(study_name=name, direction='maximize')

    # wrap regressor inside an optuna CV search object
    est = optuna.integration.OptunaSearchCV(reg, param, cv=5, study=study, refit=False, n_trials=50, timeout=3600)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, shuffle=True)

    global header

    if X_train.shape[0] > max_samples:
        sample_idx = np.random.choice(np.arange(len(X_train)), size=max_samples)
        X_train = X_train[sample_idx]
        y_train = y_train[sample_idx]

    X_train_scaled, X_test_scaled, y_train_scaled, sc_X, sc_y = scale(X_train, X_test, y_train)

    # perform search
    est.fit(X_train_scaled, y_train_scaled)

    n_trials = est.n_trials_

    if not os.path.exists('./results'):
        os.mkdir('./results')

    with open(f'./results/{name}.csv', 'w') as r:
        header = [ 'problem', 'rep', 'n_trials', 'r2_train', 'r2_test' ] + [ k for k in reg.get_params().keys() ] + [ 'model_length', 'model_complexity', 'generations', 'evaluation_count', 'residual_evaluations', 'jacobian_evaluations', 'random_state', 'model' ]
        for h in header:
            r.write(h)
            r.write('\n' if h == header[-1] else ';')

    for rep in range(reps):
        reg = SymbolicRegressor(**(est.best_params_ | default_params))

        # print(reg.get_params())

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, shuffle=True)

        if X_train.shape[0] > max_samples:
            sample_idx = np.random.choice(np.arange(len(X_train)), size=max_samples)
            X_train = X_train[sample_idx]
            y_train = y_train[sample_idx]

        X_train_scaled, X_test_scaled, y_train_scaled, sc_X, sc_y = scale(X_train, X_test, y_train)

        reg.fit(X_train_scaled, y_train_scaled)

        y_pred_train = reg.predict(X_train_scaled)
        y_pred_test = reg.predict(X_test_scaled)

        if scale_y:
            y_pred_train = sc_y.inverse_transform(y_pred_train)
            y_pred_test = sc_y.inverse_transform(y_pred_test)

        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)

        stats = { 'problem': name, 'rep': rep, 'n_trials': n_trials, 'r2_train': r2_train, 'r2_test': r2_test } | reg.get_params() | reg.stats_ | { 'model': reg.get_model_string(reg.model_, 4) }

        with open(f'./results/{name}.csv', 'a') as r:
            for h in header:
                r.write(f'{stats[h]}')
                r.write('\n' if h == header[-1] else ';')


def func(name):
    if os.path.isfile(f'./results/{name}.csv'):
        print(f'skip {name}')
        return

    X, y = pmlb.fetch_data(name, return_X_y=True, local_cache_dir='./datasets')

    reg = SymbolicRegressor(**default_params)
    optimize(name, X, y, scale_x, scale_y, reg)

if __name__ == '__main__':
    #names = pmlb.regression_dataset_names
    names = [ '1027_ESL' ]
    #names = [ '1028_SWD', '192_vineyard', '542_pollution', '1030_ERA', '1089_USCrime' ]


    with Pool(4) as pool:
        pool.map(func, names)
