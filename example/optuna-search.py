from pyoperon.sklearn import SymbolicRegressor
import optuna
import pandas as pd
import numpy as np
import copy

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import os
num_threads = int(os.environ['OPERON_THREADS']) if 'OPERON_THREADS' in os.environ else 1

rng = np.random.default_rng(1234)

class Objective:
    def __init__(self, X, y, n_folds, seed, params={}):
        self.X = X
        self.y = y
        self.n_folds = n_folds
        self.best_reg = None
        self.reg = None
        self.seed = seed
        self.params = params


    def __call__(self, trial):
        pop_size = trial.suggest_int('population_size', low=250, high=1000, step=250)

        suggested_params = {
                'population_size' : pop_size,
                'pool_size' : trial.suggest_int('pool_size', low=250, high=pop_size, step=250),
                'allowed_symbols' : trial.suggest_categorical('allowed_symbols', ['add,sub,mul,aq,constant,variable', 'add,sub,mul,aq,sin,cos,exp,log,sqrt,tanh,constant,variable']),
                'max_length' : trial.suggest_int('max_length', low=10, high=30, step=10),
                'optimizer_iterations' : trial.suggest_int('optimizer_iterations', low=0, high=5, step=1)
            }

        kf = KFold(n_splits=self.n_folds)
        score = 0

        X_, y_ = np.asarray(self.X), np.asarray(self.y)

        for train, test in kf.split(self.X, self.y):
            X_train, y_train = X_[train, :], y_[train]
            X_test, y_test   = X_[test, :], y_[test]

            params = copy.deepcopy(self.params)
            params.update(suggested_params)
            params.update({'random_state': self.seed})
            self.reg = SymbolicRegressor(**params)
            self.reg.fit(X_train, y_train)
            y_test_pred = np.nan_to_num(self.reg.predict(X_test))
            score += r2_score(y_test, y_test_pred)

        return score / self.n_folds


    def callback(self, study, trial):
        if self.best_reg is None or study.best_trial == trial:
            self.best_reg = self.reg

        if trial.value > 0.999999:
            study.stop()



class OperonOptuna(BaseEstimator, RegressorMixin):
    def __init__(self, n_folds=5, n_trials=50, timeout=3600, operon_params={}, random_state=None):
        self.n_folds = n_folds
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_state = random_state
        self.best_estimator = None
        self.operon_params=operon_params


    def fit(self, X, y):
        objective = Objective(X, y, self.n_folds, params=self.operon_params, seed=self.random_state)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, callbacks=[objective.callback], timeout=self.timeout)
        self.operon_params=objective.best_reg.get_params()
        self.best_estimator = SymbolicRegressor(**self.operon_params)
        self.best_estimator.fit(X, y)
        self.operon_params['completed trials'] = len(study.trials)


    def predict(self, X):
        return self.best_estimator.predict(X)



est = OperonOptuna(n_folds=5, n_trials=50, operon_params={
                'offspring_generator': 'basic',
                'initialization_method': 'btc',
                'n_threads': num_threads,
                'objectives':  ['r2', 'length'],
                'epsilon': 1e-4,
                'random_state': None,
                'reinserter': 'keep-best',
                'max_evaluations': int(1e6),
                'female_selector' : 'tournament',
                'male_selector': 'tournament',
                'brood_size': 5,
                'population_size': 1000,
                'model_selection_criterion': 'minimum_description_length',
                'pool_size': None,
                'time_limit': 900,
            })
