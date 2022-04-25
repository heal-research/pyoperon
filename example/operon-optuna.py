from operon.sklearn import SymbolicRegressor
import optuna
import pmlb

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from operon import FitLeastSquares

# set up some configuration options
scale_x = True
scale_y = True

# instantiate a regressor
reg = SymbolicRegressor(
        offspring_generator='basic',
        local_iterations=0,
        initialization_method='btc',
        n_threads=32,
        objectives = ['r2', 'length'],
        epsilon = 1e-4,
        random_state=None,
        reinserter='keep-best',
        max_evaluations=int(1e6),
        symbolic_mode=False,
        tournament_size=3
        )

# define parameter distributions
param = {
            'local_iterations' : optuna.distributions.IntUniformDistribution(0, 10, 1),
            'allowed_symbols' : optuna.distributions.CategoricalDistribution(['add,mul,aq,constant,variable', 'add,mul,aq,sin,cos,exp,logabs,sqrtabs,tanh,constant,variable']),
            'max_length' : optuna.distributions.IntUniformDistribution(10, 50, 10)
        }

# wrap regressor inside an optuna CV search object
optuna_search = optuna.integration.OptunaSearchCV(reg, param, cv=5, n_trials=None, timeout=60)

# load a problem
X, y = pmlb.fetch_data('1089_USCrime', return_X_y=True, local_cache_dir='./datasets')

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, shuffle=True)

# optionally scale the data
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

# perform search
optuna_search.fit(X_train_scaled, y_train_scaled)
y_pred_train = optuna_search.predict(X_train_scaled)
y_pred_test = optuna_search.predict(X_test_scaled)

if scale_y:
    y_pred_train = sc_y.inverse_transform(y_pred_train)
    y_pred_test = sc_y.inverse_transform(y_pred_test)

print('train r2:', r2_score(y_train, y_pred_train), 'test r2:', r2_score(y_test, y_pred_test))
print('stats:', optuna_search.best_estimator_.stats_)
print('params:', optuna_search.best_estimator_.get_params())
