# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, make_scorer

from pyoperon.sklearn import SymbolicRegressor
from pyoperon import R2, MSE, InfixFormatter, FitLeastSquares, Interpreter

df_train = pd.read_csv('/home/bogdb/src/poetryenv/notebooks/sr-workshop/postprocessing/data/stage1/data/3946_extrapolation_easy_data_train.csv')
df_test = pd.read_csv('/home/bogdb/src/poetryenv/notebooks/sr-workshop/postprocessing/data/stage1/data/3946_extrapolation_easy_data_train.csv')

print(df_train.columns)

D_train = np.asarray(df_train)
D_test = np.asarray(df_test)

X_train, y_train = D_train[:,:-1], D_train[:,-1]
X_test, y_test = D_train[:,:-1], D_train[:,-1]

from sympy import parse_expr
import matplotlib.pyplot as plt
import seaborn as sns

reg = SymbolicRegressor(
        allowed_symbols= "add,sub,mul,div,constant,variable",
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
        local_iterations= 0,
        male_selector= "tournament",
        max_depth= 10,
        max_evaluations= 1000000,
        max_length= 50,
        max_selection_pressure= 100,
        model_selection_criterion= "minimum_description_length",
        mutation_probability= 0.25,
        n_threads= 32,
        objectives= [ 'r2', 'length' ],
        offspring_generator= "basic",
        pool_size= 1000,
        population_size= 1000,
        random_state= None,
        reinserter= "keep-best",
        time_limit= 900,
        tournament_size= 3,
        )

print(X_train.shape, y_train.shape)

reg.fit(X_train, y_train)
values = [s['objective_values'] for s in reg.pareto_front_]
for v in values:
    print(v)

m = reg.model_
s = reg.get_model_string(m, 3, ['a'])
print(s)


fig, ax = plt.subplots(figsize=(18,8))
ax.grid(True, linestyle='dotted')
ax.set(xlabel='Obj 1', ylabel='Obj 2')
sns.scatterplot(ax=ax, x=[x[1] for x in values], y=[x[0] for x in values])

from pyoperon import RankSorter 
rs = RankSorter()
fronts = rs.Sort(values)
print(fronts)
