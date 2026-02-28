# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2024 Heal Research

"""Tests for the pyoperon.sklearn module.

Tests are organized into:
- Unit tests: parameter validation/resolution logic (no compiled extension)
- Integration tests: full fit/predict cycle (require compiled extension)
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from sklearn.base import clone
from sklearn.model_selection import cross_val_score

# Try importing the extension; skip integration tests if unavailable
try:
    import pyoperon as op
    from pyoperon.sklearn import SymbolicRegressor
    HAS_EXTENSION = True
except ImportError:
    HAS_EXTENSION = False

needs_extension = pytest.mark.skipif(
    not HAS_EXTENSION, reason='pyoperon C++ extension not available'
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_regression_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 3))
    y = X[:, 0] * 2.0 + X[:, 1] - 0.5 * X[:, 2] + rng.normal(0, 0.1, 100)
    return X, y


@pytest.fixture
def quick_regressor():
    return SymbolicRegressor(
        population_size=50,
        generations=5,
        max_evaluations=5000,
        n_threads=1,
        random_state=42,
    )


# ---------------------------------------------------------------------------
# Unit tests: parameter handling (no extension required)
# ---------------------------------------------------------------------------

@needs_extension
class TestParameterResolution:
    """Test that _resolve_params works correctly without mutating self."""

    def test_none_defaults_resolved(self):
        reg = SymbolicRegressor()
        resolved = reg._resolve_params()

        assert resolved['mutation'] is not None
        assert isinstance(resolved['mutation'], dict)
        assert resolved['objectives'] == ['r2']
        assert resolved['uncertainty'] == [1]
        assert resolved['pool_size'] == reg.population_size
        assert resolved['max_time'] != None
        assert isinstance(resolved['random_state'], int)

    def test_self_not_mutated(self):
        reg = SymbolicRegressor(
            random_state=None,
            pool_size=None,
            max_time=None,
            mutation=None,
            objectives=None,
            uncertainty=None,
        )

        reg._resolve_params()

        assert reg.random_state is None
        assert reg.pool_size is None
        assert reg.max_time is None
        assert reg.mutation is None
        assert reg.objectives is None
        assert reg.uncertainty is None

    def test_explicit_values_preserved(self):
        reg = SymbolicRegressor(
            pool_size=500,
            max_time=3600,
            random_state=123,
            mutation={'onepoint': 2.0},
            objectives=['r2', 'length'],
            uncertainty=[0.5],
        )
        resolved = reg._resolve_params()

        assert resolved['pool_size'] == 500
        assert resolved['max_time'] == 3600
        assert resolved['random_state'] == 123
        assert resolved['mutation'] == {'onepoint': 2.0}
        assert resolved['objectives'] == ['r2', 'length']
        assert resolved['uncertainty'] == [0.5]

    def test_scaling_objectives_force_scale_intercept(self):
        for obj in ['r2', 'nmse', 'rmse', 'mse', 'mae']:
            reg = SymbolicRegressor(
                objectives=[obj],
                add_model_scale_term=False,
                add_model_intercept_term=False,
            )
            with pytest.warns(UserWarning, match='overriding to True'):
                resolved = reg._resolve_params()

            assert resolved['add_model_scale_term'] is True
            assert resolved['add_model_intercept_term'] is True

    def test_non_scaling_objectives_respect_settings(self):
        reg = SymbolicRegressor(
            objectives=['c2'],
            add_model_scale_term=False,
            add_model_intercept_term=False,
        )
        resolved = reg._resolve_params()

        assert resolved['add_model_scale_term'] is False
        assert resolved['add_model_intercept_term'] is False

    def test_symbolic_mode_disables_optimizer(self):
        reg = SymbolicRegressor(
            symbolic_mode=True,
            optimizer_iterations=10,
        )
        resolved = reg._resolve_params()
        assert resolved['optimizer_iterations'] == 0

    def test_numpy_generator_converted_to_int(self):
        gen = np.random.default_rng(42)
        reg = SymbolicRegressor(random_state=gen)
        resolved = reg._resolve_params()
        assert isinstance(resolved['random_state'], (int, np.integer))

    def test_resolved_mutation_is_copy(self):
        """Resolved mutation dict should be a copy, not a reference."""
        reg = SymbolicRegressor(mutation={'onepoint': 1.0})
        resolved = reg._resolve_params()
        resolved['mutation']['onepoint'] = 999.0
        assert reg.mutation['onepoint'] == 1.0


@needs_extension
class TestParameterValidation:
    """Test that _validate_params raises errors for invalid inputs."""

    def test_invalid_symbol(self):
        reg = SymbolicRegressor(allowed_symbols='add,sub,INVALID')
        resolved = reg._resolve_params()
        with pytest.raises(ValueError, match='Unknown symbol'):
            reg._validate_params(resolved)

    def test_invalid_objective(self):
        reg = SymbolicRegressor(objectives=['r2', 'INVALID'])
        resolved = reg._resolve_params()
        with pytest.raises(ValueError, match='Unknown objective'):
            reg._validate_params(resolved)

    def test_invalid_mutation(self):
        reg = SymbolicRegressor(mutation={'INVALID': 1.0})
        resolved = reg._resolve_params()
        with pytest.raises(ValueError, match='Unknown mutation'):
            reg._validate_params(resolved)

    def test_invalid_offspring_generator(self):
        reg = SymbolicRegressor(offspring_generator='INVALID')
        resolved = reg._resolve_params()
        with pytest.raises(ValueError, match='offspring_generator'):
            reg._validate_params(resolved)

    def test_invalid_reinserter(self):
        reg = SymbolicRegressor(reinserter='INVALID')
        resolved = reg._resolve_params()
        with pytest.raises(ValueError, match='reinserter'):
            reg._validate_params(resolved)

    def test_invalid_selector(self):
        reg = SymbolicRegressor(female_selector='INVALID')
        resolved = reg._resolve_params()
        with pytest.raises(ValueError, match='female_selector'):
            reg._validate_params(resolved)

    def test_invalid_initialization_method(self):
        reg = SymbolicRegressor(initialization_method='INVALID')
        resolved = reg._resolve_params()
        with pytest.raises(ValueError, match='initialization_method'):
            reg._validate_params(resolved)

    def test_invalid_optimizer(self):
        reg = SymbolicRegressor(optimizer='INVALID')
        resolved = reg._resolve_params()
        with pytest.raises(ValueError, match='optimizer'):
            reg._validate_params(resolved)

    def test_invalid_sgd_update_rule(self):
        reg = SymbolicRegressor(sgd_update_rule='INVALID')
        resolved = reg._resolve_params()
        with pytest.raises(ValueError, match='sgd_update_rule'):
            reg._validate_params(resolved)

    def test_invalid_population_size(self):
        reg = SymbolicRegressor(population_size=0)
        resolved = reg._resolve_params()
        with pytest.raises(ValueError, match='population_size'):
            reg._validate_params(resolved)

    def test_empty_objectives(self):
        reg = SymbolicRegressor(objectives=[])
        resolved = reg._resolve_params()
        with pytest.raises(ValueError, match='objectives must not be empty'):
            reg._validate_params(resolved)

    def test_valid_params_pass(self):
        reg = SymbolicRegressor()
        resolved = reg._resolve_params()
        reg._validate_params(resolved)  # should not raise


# ---------------------------------------------------------------------------
# Sklearn interface tests
# ---------------------------------------------------------------------------

@needs_extension
class TestSklearnInterface:
    """Test sklearn estimator contract compliance."""

    def test_get_params_set_params_roundtrip(self):
        reg = SymbolicRegressor(population_size=500, generations=50)
        params = reg.get_params()
        assert params['population_size'] == 500
        assert params['generations'] == 50

        reg.set_params(population_size=200)
        assert reg.population_size == 200

    def test_clone_preserves_params(self):
        reg = SymbolicRegressor(
            population_size=500,
            random_state=42,
            mutation={'onepoint': 2.0},
        )
        reg2 = clone(reg)

        assert reg2.population_size == 500
        assert reg2.random_state == 42
        assert reg2.mutation == {'onepoint': 2.0}

    def test_clone_after_fit_preserves_original_params(
        self, small_regression_data, quick_regressor
    ):
        """Cloning after fit should use the original __init__ params."""
        X, y = small_regression_data
        quick_regressor.fit(X, y)

        cloned = clone(quick_regressor)
        assert cloned.random_state == 42
        assert not hasattr(cloned, 'is_fitted_')

    def test_none_defaults_clone_correctly(self):
        reg = SymbolicRegressor()
        cloned = clone(reg)

        assert cloned.mutation is None
        assert cloned.objectives is None
        assert cloned.uncertainty is None
        assert cloned.pool_size is None
        assert cloned.max_time is None
        assert cloned.random_state is None

    def test_repr(self):
        reg = SymbolicRegressor(population_size=100)
        r = repr(reg)
        assert 'SymbolicRegressor' in r
        assert 'population_size=100' in r


# ---------------------------------------------------------------------------
# Integration tests: full fit/predict (require extension)
# ---------------------------------------------------------------------------

@needs_extension
class TestFitPredict:
    """Integration tests for the full fit/predict pipeline."""

    def test_basic_fit_predict(self, small_regression_data, quick_regressor):
        X, y = small_regression_data
        quick_regressor.fit(X, y)

        assert hasattr(quick_regressor, 'is_fitted_')
        assert hasattr(quick_regressor, 'model_')
        assert hasattr(quick_regressor, 'pareto_front_')
        assert hasattr(quick_regressor, 'stats_')
        assert hasattr(quick_regressor, 'individuals_')

        y_pred = quick_regressor.predict(X)
        assert y_pred.shape == y.shape
        assert np.all(np.isfinite(y_pred))

    def test_n_features_in_set(self, small_regression_data, quick_regressor):
        X, y = small_regression_data
        quick_regressor.fit(X, y)
        assert quick_regressor.n_features_in_ == 3

    def test_feature_names_from_dataframe(self, small_regression_data):
        import pandas as pd

        X, y = small_regression_data
        df = pd.DataFrame(X, columns=['a', 'b', 'c'])
        reg = SymbolicRegressor(
            population_size=50, generations=5,
            max_evaluations=5000, random_state=42,
        )
        reg.fit(df, y)

        assert hasattr(reg, 'feature_names_in_')
        np.testing.assert_array_equal(
            reg.feature_names_in_, np.array(['a', 'b', 'c']),
        )

    def test_predict_before_fit_raises(self):
        reg = SymbolicRegressor()
        with pytest.raises(Exception):
            reg.predict(np.ones((10, 3)))

    def test_model_string(self, small_regression_data, quick_regressor):
        X, y = small_regression_data
        quick_regressor.fit(X, y)
        s = quick_regressor.get_model_string(quick_regressor.model_)
        assert isinstance(s, str)
        assert len(s) > 0

    def test_pareto_front_structure(
        self, small_regression_data, quick_regressor
    ):
        X, y = small_regression_data
        quick_regressor.fit(X, y)

        assert len(quick_regressor.pareto_front_) >= 1
        entry = quick_regressor.pareto_front_[0]
        expected_keys = {
            'model', 'variables', 'length', 'complexity', 'tree',
            'objective_values', 'mean_squared_error',
            'minimum_description_length',
            'bayesian_information_criterion',
            'akaike_information_criterion',
        }
        assert set(entry.keys()) == expected_keys

    def test_stats_structure(self, small_regression_data, quick_regressor):
        X, y = small_regression_data
        quick_regressor.fit(X, y)

        expected_keys = {
            'model_length', 'model_complexity', 'generations',
            'evaluation_count', 'residual_evaluations',
            'jacobian_evaluations', 'random_state',
        }
        assert set(quick_regressor.stats_.keys()) == expected_keys

    def test_multi_objective(self, small_regression_data):
        X, y = small_regression_data
        reg = SymbolicRegressor(
            population_size=50, generations=5,
            max_evaluations=5000, random_state=42,
            objectives=['r2', 'length'],
        )
        reg.fit(X, y)
        assert len(reg.pareto_front_) >= 1

        y_pred = reg.predict(X)
        assert y_pred.shape == y.shape

    def test_symbolic_mode(self, small_regression_data):
        X, y = small_regression_data
        reg = SymbolicRegressor(
            population_size=50, generations=5,
            max_evaluations=5000, random_state=42,
            symbolic_mode=True,
        )
        reg.fit(X, y)
        y_pred = reg.predict(X)
        assert y_pred.shape == y.shape

    def test_warm_start(self, small_regression_data):
        X, y = small_regression_data
        reg = SymbolicRegressor(
            population_size=50, generations=3,
            max_evaluations=5000, random_state=42,
            warm_start=True,
        )
        reg.fit(X, y)
        first_gen = reg.stats_['generations']

        reg.fit(X, y)
        assert reg.is_fitted_

    def test_evaluate_model(self, small_regression_data, quick_regressor):
        X, y = small_regression_data
        quick_regressor.fit(X, y)

        for entry in quick_regressor.pareto_front_:
            y_pred = quick_regressor.evaluate_model(entry['tree'], X)
            assert y_pred.shape == y.shape

    def test_cross_val_score_runs(self, small_regression_data):
        X, y = small_regression_data
        reg = SymbolicRegressor(
            population_size=50, generations=3,
            max_evaluations=3000, random_state=42,
        )
        scores = cross_val_score(reg, X, y, cv=2, scoring='r2')
        assert len(scores) == 2

    def test_different_optimizers(self, small_regression_data):
        X, y = small_regression_data
        for opt in ['lm', 'lbfgs']:
            reg = SymbolicRegressor(
                population_size=50, generations=3,
                max_evaluations=3000, random_state=42,
                optimizer=opt, optimizer_iterations=1,
            )
            reg.fit(X, y)
            assert reg.is_fitted_

    def test_different_initialization_methods(self, small_regression_data):
        X, y = small_regression_data
        for method in ['btc', 'ptc2', 'koza']:
            reg = SymbolicRegressor(
                population_size=50, generations=3,
                max_evaluations=3000, random_state=42,
                initialization_method=method,
            )
            reg.fit(X, y)
            assert reg.is_fitted_
