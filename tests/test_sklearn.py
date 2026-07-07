# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2024 Heal Research

"""Tests for the pyoperon.sklearn module.

Tests are organized into:
- Unit tests: parameter validation/resolution logic (no compiled extension)
- Integration tests: full fit/predict cycle (require compiled extension)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone
from sklearn.model_selection import cross_val_score

DATA_DIR = Path(__file__).parent / 'data'

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

    def test_callbacks_generator_rejected(self):
        """self.callbacks is re-read on every fit() call, so a one-shot
        iterator/generator would be silently exhausted after the first
        fit(). Reject anything that isn't a list/tuple/Callback/None
        outright instead of accepting it and breaking on the second fit()."""
        reg = SymbolicRegressor(callbacks=iter([op.EarlyStopping()]))
        with pytest.raises(ValueError, match='callbacks must be'):
            reg._resolve_params()


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


# ---------------------------------------------------------------------------
# Unit tests: Callback / CallbackList / EarlyStopping logic (no fit required)
# ---------------------------------------------------------------------------

@needs_extension
class TestCallbackLogic:
    """Test callback dispatch/early-stop logic against a fake model, without
    running an actual GP fit."""

    class _FakeBestModel:
        def __init__(self, fitness):
            self._fitness = fitness

        def GetFitness(self, idx):
            return self._fitness

    class _FakeModel:
        def __init__(self, fitness):
            self.BestModel = TestCallbackLogic._FakeBestModel(fitness)

    def test_callback_list_ors_stop_requests(self):
        class AlwaysStop(op.Callback):
            def on_generation_end(self, model):
                return True

        class NeverStop(op.Callback):
            def on_generation_end(self, model):
                return False

        cb_list = op.CallbackList([NeverStop(), AlwaysStop()])
        assert cb_list.on_generation_end(self._FakeModel(1.0)) is True

    def test_callback_list_does_not_short_circuit(self):
        calls = []

        class Recorder(op.Callback):
            def __init__(self, name):
                self.name = name

            def on_generation_end(self, model):
                calls.append(self.name)
                return True

        cb_list = op.CallbackList([Recorder('a'), Recorder('b')])
        cb_list.on_generation_end(self._FakeModel(1.0))
        assert calls == ['a', 'b']

    def test_early_stopping_stops_after_patience(self):
        es = op.EarlyStopping(patience=3)
        es.on_fit_begin(self._FakeModel(1.0))
        # Constant fitness never improves, so the wait counter climbs every
        # generation until it reaches patience.
        results = [es.on_generation_end(self._FakeModel(1.0)) for _ in range(5)]
        assert results == [False, False, False, True, True]

    def test_early_stopping_resets_between_fits(self):
        es = op.EarlyStopping(patience=2)
        es.on_fit_begin(self._FakeModel(1.0))
        for _ in range(3):
            es.on_generation_end(self._FakeModel(1.0))
        assert es._wait >= 2

        # fit() doesn't clone self.callbacks, so calling fit() twice on the
        # same regressor (e.g. warm_start) reuses this exact instance - a
        # fresh on_fit_begin must reset accumulated state rather than
        # carrying it over.
        es.on_fit_begin(self._FakeModel(1.0))
        assert es._wait == 0
        assert es._best is None

    def test_normalize_callbacks_accepts_single_callback(self):
        cb = op.EarlyStopping()
        reg = SymbolicRegressor(callbacks=cb)
        assert reg._normalize_callbacks(reg.callbacks) == [cb]

    def test_normalize_callbacks_accepts_tuple(self):
        cb = op.EarlyStopping()
        reg = SymbolicRegressor(callbacks=(cb,))
        assert reg._normalize_callbacks(reg.callbacks) == [cb]

    def test_early_stopping_rejects_negative_patience(self):
        with pytest.raises(ValueError, match='patience'):
            op.EarlyStopping(patience=-1)

    def test_early_stopping_rejects_negative_objective_index(self):
        with pytest.raises(ValueError, match='objective_index'):
            op.EarlyStopping(objective_index=-1)


# ---------------------------------------------------------------------------
# Integration tests: callbacks wired through SymbolicRegressor.fit()
# ---------------------------------------------------------------------------

@needs_extension
class TestCallbacksIntegration:

    def test_on_fit_end_skipped_if_on_fit_begin_raises(self, small_regression_data):
        X, y = small_regression_data

        class Bad(op.Callback):
            def __init__(self):
                self.end_calls = 0

            def on_fit_begin(self, model):
                raise RuntimeError('boom')

            def on_fit_end(self, model):
                self.end_calls += 1

        cb = Bad()
        reg = SymbolicRegressor(
            population_size=50, generations=5,
            max_evaluations=5000, random_state=42,
            callbacks=[cb],
        )
        with pytest.raises(RuntimeError, match='boom'):
            reg.fit(X, y)
        assert cb.end_calls == 0

    def test_custom_callback_hooks_invoked(self, small_regression_data):
        X, y = small_regression_data

        class Recorder(op.Callback):
            def __init__(self):
                self.begin_calls = 0
                self.end_calls = 0
                self.generations_seen = []

            def on_fit_begin(self, model):
                self.begin_calls += 1

            def on_generation_end(self, model):
                self.generations_seen.append(model.Generation)
                return False

            def on_fit_end(self, model):
                self.end_calls += 1

        rec = Recorder()
        reg = SymbolicRegressor(
            population_size=50, generations=5,
            max_evaluations=5000, random_state=42,
            callbacks=[rec],
        )
        reg.fit(X, y)

        assert rec.begin_calls == 1
        assert rec.end_calls == 1
        assert len(rec.generations_seen) > 0

    def test_early_stopping_stops_run_early(self, small_regression_data):
        X, y = small_regression_data
        # An enormous min_delta means no generation ever counts as an
        # improvement, so with patience=1 the run should stop after just a
        # couple of generations. max_evaluations is set far above what 1000
        # generations at this population size could ever consume, so it
        # can't be the reason the run stops early - only early-stopping can
        # (a tight bound here, rather than "< 1000", is what actually
        # catches a regression where the stop flag stops propagating).
        es = op.EarlyStopping(patience=1, min_delta=1e6)
        reg = SymbolicRegressor(
            population_size=50, generations=1000,
            max_evaluations=10_000_000, random_state=42,
            callbacks=[es],
        )
        reg.fit(X, y)
        assert reg.stats_['generations'] < 10

    def test_callback_survives_cross_val_score(self, small_regression_data):
        X, y = small_regression_data
        # cross_val_score clones reg per fold via sklearn's clone(), which
        # deep-copies self.callbacks (EarlyStopping isn't a BaseEstimator,
        # so it doesn't get cloned "deep=False" like nested estimators do) -
        # this just confirms that round-trips cleanly and doesn't crash.
        es = op.EarlyStopping(patience=2)
        reg = SymbolicRegressor(
            population_size=50, generations=5,
            max_evaluations=5000, random_state=42,
            callbacks=[es],
        )
        scores = cross_val_score(reg, X, y, cv=2, scoring='r2')
        assert len(scores) == 2

    def test_on_fit_begin_invoked_on_every_fit_call(self, small_regression_data):
        X, y = small_regression_data

        class Recorder(op.Callback):
            def __init__(self):
                self.begin_calls = 0

            def on_fit_begin(self, model):
                self.begin_calls += 1

        rec = Recorder()
        reg = SymbolicRegressor(
            population_size=50, generations=5,
            max_evaluations=5000, random_state=42,
            callbacks=[rec],
        )
        # Unlike clone(), fit() does *not* copy self.callbacks - calling
        # fit() twice on the same regressor (e.g. warm_start) reuses this
        # exact instance, so on_fit_begin must fire every time for
        # per-fit state (e.g. EarlyStopping's _wait/_best, see
        # TestCallbackLogic.test_early_stopping_resets_between_fits) to
        # actually reset instead of carrying over.
        reg.fit(X, y)
        assert rec.begin_calls == 1

        reg.fit(X, y)
        assert rec.begin_calls == 2


# ---------------------------------------------------------------------------
# Unit tests: Dataset.SetWeights/Weights (no fit required)
# ---------------------------------------------------------------------------

@needs_extension
class TestDatasetWeights:

    def test_weights_none_by_default(self):
        ds = op.Dataset(np.asfortranarray(np.zeros((5, 2), dtype=np.float32)))
        assert ds.Weights is None

    def test_set_weights_roundtrips(self):
        ds = op.Dataset(np.asfortranarray(np.zeros((5, 2), dtype=np.float32)))
        w = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        ds.SetWeights(w)
        np.testing.assert_array_equal(np.asarray(ds.Weights), w)


# ---------------------------------------------------------------------------
# Integration tests: sample_weight through SymbolicRegressor.fit()
# ---------------------------------------------------------------------------

@needs_extension
class TestSampleWeight:

    def test_mismatched_length_raises(self, small_regression_data):
        X, y = small_regression_data
        reg = SymbolicRegressor(population_size=50, generations=3, random_state=42)
        with pytest.raises(ValueError):
            reg.fit(X, y, sample_weight=np.ones(len(y) - 1))

    def test_negative_weight_raises(self, small_regression_data):
        X, y = small_regression_data
        reg = SymbolicRegressor(population_size=50, generations=3, random_state=42)
        sample_weight = np.ones(len(y))
        sample_weight[0] = -1.0
        with pytest.raises(ValueError):
            reg.fit(X, y, sample_weight=sample_weight)

    def test_all_zero_weight_raises(self, small_regression_data):
        X, y = small_regression_data
        reg = SymbolicRegressor(population_size=50, generations=3, random_state=42)
        with pytest.raises(ValueError):
            reg.fit(X, y, sample_weight=np.zeros(len(y)))

    def test_optimizer_iterations_warns_with_sample_weight(self, small_regression_data):
        X, y = small_regression_data
        reg = SymbolicRegressor(
            population_size=50, generations=3, random_state=42,
            optimizer_iterations=5,
        )
        with pytest.warns(UserWarning, match='optimizer_iterations'):
            reg.fit(X, y, sample_weight=np.ones(len(y)))

    @pytest.mark.filterwarnings('error')
    def test_no_warning_with_default_optimizer_iterations(self, small_regression_data):
        """optimizer_iterations defaults to 0, so the default sample_weight
        path must stay silent - pytest.mark.filterwarnings('error') turns
        any stray warning (this one or otherwise) into a test failure."""
        X, y = small_regression_data
        reg = SymbolicRegressor(population_size=50, generations=3, random_state=42)
        reg.fit(X, y, sample_weight=np.ones(len(y)))

    def test_uniform_weights_match_unweighted_metric(self):
        """Uniform sample_weight must reduce to the unweighted formula.
        Checked directly on fixed predictions/targets rather than through
        a full (stochastic) GP run: even a 1-ULP difference in reduction
        order between the weighted and unweighted vstat paths could flip
        tournament selections and diverge two GP populations over many
        generations, so comparing full-run stats would be fragile."""
        rng = np.random.default_rng(0)
        y_true = rng.standard_normal(50).astype(np.float32)
        y_pred = (y_true + rng.normal(0, 0.1, 50)).astype(np.float32)
        w = np.ones(50, dtype=np.float32)

        scale_u, offset_u = op.FitLeastSquares(y_pred, y_true)
        scale_w, offset_w = op.FitLeastSquares(y_pred, y_true, w)
        assert scale_w == pytest.approx(scale_u)
        assert offset_w == pytest.approx(offset_u)

    def test_evaluator_uniform_weight_matches_unweighted(self):
        """Same equivalence property as above, but exercised directly
        through operon's weighted Evaluator path (error_(buf, target,
        weights) in evaluator.cpp) rather than just FitLeastSquares - on a
        manually-built tree/individual, again avoiding a full GP run."""
        rng_np = np.random.default_rng(0)
        X = rng_np.standard_normal((30, 1)).astype(np.float32)
        y = (2 * X[:, 0]).astype(np.float32)
        D = np.asfortranarray(np.column_stack((X, y)))
        ds = op.Dataset(D)
        target = max(ds.Variables, key=lambda v: v.Index)
        inputs = [v.Hash for v in ds.Variables if v.Hash != target.Hash]

        problem = op.Problem(ds)
        problem.TrainingRange = op.Range(0, ds.Rows)
        problem.Target = target
        problem.InputHashes = inputs

        var_node = op.Node.Variable(1.0)
        var_node.HashValue = inputs[0]
        tree = op.Tree([var_node]).UpdateNodes()
        ind = op.Individual()
        ind.Genotype = tree

        dtable = op.DispatchTable()
        evaluator = op.Evaluator(problem, dtable, op.MSE(), False)
        rng = op.RandomGenerator(np.uint64(0))

        fitness_unweighted = evaluator(rng, ind)[0]
        ds.SetWeights(np.ones(ds.Rows, dtype=np.float32))
        fitness_weighted = evaluator(rng, ind)[0]

        assert fitness_weighted == pytest.approx(fitness_unweighted)

    def test_downweighting_outliers_improves_fit_on_clean_data(self):
        """A real, visibly-biased scenario (not just a formula check):
        mostly-clean linear data contaminated with a handful of outliers
        whose y values are unrelated to X. Without sample_weight, the GP
        loss is dragged around by the outliers; heavily down-weighting
        them should recover a much better fit to the clean majority.
        """
        rng = np.random.default_rng(0)
        n_clean, n_outliers = 90, 10
        X_clean = rng.uniform(0, 10, n_clean).reshape(-1, 1)
        y_clean = 2 * X_clean[:, 0] + 1 + rng.normal(0, 0.1, n_clean)
        X_outliers = rng.uniform(0, 10, n_outliers).reshape(-1, 1)
        y_outliers = rng.uniform(-40, 40, n_outliers)

        X = np.vstack([X_clean, X_outliers])
        y = np.concatenate([y_clean, y_outliers])

        sample_weight = np.ones(n_clean + n_outliers)
        sample_weight[n_clean:] = 0.001

        common_kwargs = dict(
            population_size=100, generations=20, max_evaluations=50_000,
            random_state=0, allowed_symbols='add,sub,mul,constant,variable',
        )
        reg_unweighted = SymbolicRegressor(**common_kwargs)
        reg_unweighted.fit(X, y)
        reg_weighted = SymbolicRegressor(**common_kwargs)
        reg_weighted.fit(X, y, sample_weight=sample_weight)

        err_unweighted = np.mean((reg_unweighted.predict(X_clean) - y_clean) ** 2)
        err_weighted = np.mean((reg_weighted.predict(X_clean) - y_clean) ** 2)

        # Outliers are down-weighted 1000x, so the weighted fit's error on
        # the clean majority should be dramatically lower, not barely so -
        # a tight margin makes this robust across operon versions/builds.
        assert err_weighted < 0.5 * err_unweighted

    @staticmethod
    def _weighted_mse(pred, y, w):
        return np.sum(w * (pred - y) ** 2) / np.sum(w)

    def test_star98_weighted_fit_wins_on_weighted_metric(self):
        """Real dataset, not synthetic: 1998 California STAR testing results
        for 303 school districts (see tests/data/star98.json) - a classic
        textbook example of weighted least squares. MATHTOT (students
        tested per district) is the natural precision weight: a district's
        pass-rate estimate is more reliable the more students it tested.
        Verified independently with plain WLS (see star98.json) that this
        dataset shows a genuine, non-trivial weighting effect before
        writing this GP-based test.

        Only asserts the direction that held across every random_state we
        tried (see PR discussion): the model trained with sample_weight
        achieves a lower *weighted* MSE than the model trained without it.
        The converse (unweighted-trained model wins unweighted MSE) also
        held for this fixed seed/budget but was seed-sensitive in general,
        so it's not asserted here to avoid cross-platform GP-determinism
        flakiness.
        """
        df = pd.read_csv(DATA_DIR / 'star98.csv')
        y = (df['PR50M'] / df['MATHTOT'] * 100).to_numpy()
        w = df['MATHTOT'].to_numpy(dtype=float)
        cols = [
            'LOWINC', 'PERASIAN', 'PERBLACK', 'PERHISP', 'PERMINTE',
            'AVYRSEXP', 'AVSALK', 'PERSPENK', 'PTRATIO', 'PCTAF',
            'PCTCHRT', 'PCTYRRND',
        ]
        X = df[cols].to_numpy()

        common_kwargs = dict(
            population_size=500, generations=100, max_evaluations=500_000,
            random_state=0, allowed_symbols='add,sub,mul,constant,variable',
        )
        reg_unweighted = SymbolicRegressor(**common_kwargs)
        reg_unweighted.fit(X, y)
        reg_weighted = SymbolicRegressor(**common_kwargs)
        reg_weighted.fit(X, y, sample_weight=w)

        mse_w_unweighted_fit = self._weighted_mse(reg_unweighted.predict(X), y, w)
        mse_w_weighted_fit = self._weighted_mse(reg_weighted.predict(X), y, w)
        assert mse_w_weighted_fit < mse_w_unweighted_fit

    def test_acs_pums_weighted_fit_wins_on_weighted_metric(self):
        """Real dataset, not synthetic: a random subsample of the 2022 ACS
        1-Year PUMS person file for Wyoming (see tests/data/acs_pums_wy2022.json),
        public domain US Census data. PWGTP is a genuine survey sampling
        weight (how many people in the population each record represents) -
        the canonical real-world use case sample_weight was requested for
        (pyoperon issue #17), unlike star98's derived precision weight.

        Same asymmetric-assertion rationale as the star98 test above: only
        the direction robust across every random_state tried is asserted.
        """
        df = pd.read_csv(DATA_DIR / 'acs_pums_wy2022.csv')
        y = np.log(df['PINCP'].to_numpy())
        w = df['PWGTP'].to_numpy(dtype=float)
        cols = ['AGEP', 'SCHL', 'WKHP', 'SEX', 'MAR']
        X = df[cols].to_numpy()

        common_kwargs = dict(
            population_size=100, generations=20, max_evaluations=100_000,
            random_state=1, allowed_symbols='add,sub,mul,constant,variable',
        )
        reg_unweighted = SymbolicRegressor(**common_kwargs)
        reg_unweighted.fit(X, y)
        reg_weighted = SymbolicRegressor(**common_kwargs)
        reg_weighted.fit(X, y, sample_weight=w)

        mse_w_unweighted_fit = self._weighted_mse(reg_unweighted.predict(X), y, w)
        mse_w_weighted_fit = self._weighted_mse(reg_weighted.predict(X), y, w)
        assert mse_w_weighted_fit < mse_w_unweighted_fit
