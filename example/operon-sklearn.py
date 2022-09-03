# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, make_scorer
from scipy.stats import pearsonr

from pyoperon.sklearn import SymbolicRegressor
from pyoperon import R2, InfixFormatter, FitLeastSquares

from pmlb import fetch_data, dataset_names, classification_dataset_names, regression_dataset_names

X, y = fetch_data('1027_ESL', return_X_y=True, local_cache_dir='./datasets')

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, shuffle=True)

reg = SymbolicRegressor(
        allowed_symbols='add,sub,mul,aq,sin,constant,variable',
        offspring_generator='basic',
        local_iterations=10,
        max_length=50,
        initialization_method='btc',
        n_threads=32,
        objectives = ['r2', 'length'],
        epsilon = 1e-3,
        random_state=None,
        reinserter='keep-best',
        max_evaluations=int(1e5),
        symbolic_mode=False
        )

print(X_train.shape, y_train.shape)

reg.fit(X_train, y_train)
print(reg.get_model_string(reg.model_, 2))
print(reg.get_model_string(reg.model_, names=['A', 'B', 'C', 'D' ], precision=2))
print(reg.stats_)

for model, model_vars, model_obj, bic in reg.pareto_front_:
    y_pred_train = reg.evaluate_model(model, X_train)
    y_pred_test = reg.evaluate_model(model, X_test)

    scale, offset = FitLeastSquares(y_pred_train, y_train)
    y_pred_train = scale * y_pred_train + offset
    y_pred_test = scale * y_pred_test + offset

    variables = { v.Hash : v.Name for v in model_vars }
    print(f'{bic:.3f}', InfixFormatter.Format(model, variables, 3), model.Length, r2_score(y_train, y_pred_train), r2_score(y_train, y_pred_train))

r2 = R2()

y_pred_train = reg.predict(X_train)
print('r2 train (sklearn.r2_score): ', r2_score(y_train, y_pred_train))
print('r2 train (operon.r2): ', -r2(y_pred_train, y_train))

y_pred_test = reg.predict(X_test)
print('r2 test (sklearn.r2_score): ', r2_score(y_test, y_pred_test))

# crossvalidation
sc = make_scorer(r2_score, greater_is_better=True)
scores = cross_val_score(reg, X, y, cv=5, scoring=sc)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
