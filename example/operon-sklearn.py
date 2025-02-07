# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, make_scorer, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from pyoperon.sklearn import SymbolicRegressor
from pyoperon import R2, MSE, InfixFormatter, FitLeastSquares, Interpreter

D_train = pd.read_csv('../experiment/data/stage1/6589_noise_easy_data_train.csv', sep=',')
D_test = pd.read_csv('../experiment/data/stage1/6589_noise_easy_data_test.csv', sep=',')

X_train, y_train = D_train.iloc[:,:-1], D_train.iloc[:,-1]
X_test, y_test = D_test.iloc[:,:-1], D_train.iloc[:,-1]
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, shuffle=False)

y_pred = RandomForestRegressor(n_estimators=100).fit(X_train, y_train).predict(X_train)
sErr = np.sqrt(mean_squared_error(y_train,  y_pred)) # estimate of the uncertainty (empirical error)

from sympy import parse_expr
import matplotlib.pyplot as plt
from copy import deepcopy

reg = SymbolicRegressor(
        allowed_symbols= "add,sub,mul,aq,sin,constant,variable",
        brood_size= 10,
        comparison_factor= 0,
        crossover_internal_probability= 0.9,
        crossover_probability= 1.0,
        epsilon= 1e-05,
        female_selector= "tournament",
        generations= 1000,
        initialization_max_depth= 5,
        initialization_max_length= 10,
        initialization_method= "btc",
        irregularity_bias= 0.0,
        local_search_probability=1.0,
        lamarckian_probability=1.0,
        optimizer_iterations=1,
        optimizer='lm',
        male_selector= "tournament",
        max_depth= 10,
        max_evaluations= 1000000,
        max_length= 50,
        max_selection_pressure= 100,
        model_selection_criterion= "minimum_description_length",
        mutation_probability= 0.25,
        n_threads= 32,
        objectives= [ 'r2', 'length' ],
        offspring_generator= "os",
        pool_size= 1000,
        population_size= 1000,
        random_state= None,
        reinserter= "keep-best",
        max_time= 900,
        tournament_size=3,
        uncertainty= [sErr],
        add_model_intercept_term=True,
        add_model_scale_term=True
        )

reg.fit(X_train, y_train)
res = [(s['objective_values'], s['tree'], s['minimum_description_length']) for s in reg.pareto_front_]
for obj, expr, mdl in res:
    print(f'{obj}, {mdl:.2f}, {reg.get_model_string(expr, 12)}')
