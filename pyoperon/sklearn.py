# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2024 Heal Research

from __future__ import annotations

import sys
import random
import warnings
from typing import Any

import numpy as np
import pyoperon as op

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import mean_squared_error


_KNOWN_SYMBOLS: dict[str, Any] = {
    'add'      : op.NodeType.Add,
    'mul'      : op.NodeType.Mul,
    'sub'      : op.NodeType.Sub,
    'div'      : op.NodeType.Div,
    'fmin'     : op.NodeType.Fmin,
    'fmax'     : op.NodeType.Fmax,
    'aq'       : op.NodeType.Aq,
    'pow'      : op.NodeType.Pow,
    'powabs'   : op.NodeType.Powabs,
    'abs'      : op.NodeType.Abs,
    'acos'     : op.NodeType.Acos,
    'asin'     : op.NodeType.Asin,
    'atan'     : op.NodeType.Atan,
    'cbrt'     : op.NodeType.Cbrt,
    'ceil'     : op.NodeType.Ceil,
    'cos'      : op.NodeType.Cos,
    'cosh'     : op.NodeType.Cosh,
    'exp'      : op.NodeType.Exp,
    'floor'    : op.NodeType.Floor,
    'log'      : op.NodeType.Log,
    'logabs'   : op.NodeType.Logabs,
    'log1p'    : op.NodeType.Log1p,
    'sin'      : op.NodeType.Sin,
    'sinh'     : op.NodeType.Sinh,
    'sqrt'     : op.NodeType.Sqrt,
    'sqrtabs'  : op.NodeType.Sqrtabs,
    'tan'      : op.NodeType.Tan,
    'tanh'     : op.NodeType.Tanh,
    'square'   : op.NodeType.Square,
    'constant' : op.NodeType.Constant,
    'variable' : op.NodeType.Variable,
}

_SCALING_OBJECTIVES = frozenset({'r2', 'nmse', 'rmse', 'mse', 'mae'})

_VALID_OBJECTIVES = frozenset({
    'r2', 'c2', 'nmse', 'rmse', 'mse', 'mae',
    'length', 'shape', 'diversity', 'mdl', 'poisson', 'gauss',
})

_VALID_GENERATORS  = frozenset({'basic', 'os', 'brood', 'poly'})
_VALID_REINSERTERS = frozenset({'replace-worst', 'keep-best'})
_VALID_SELECTORS   = frozenset({'tournament', 'proportional', 'random'})
_VALID_INIT_METHODS = frozenset({'btc', 'ptc2', 'koza'})
_VALID_OPTIMIZERS  = frozenset({'lm', 'lbfgs', 'sgd'})

_VALID_SGD_RULES = frozenset({
    'constant', 'momentum', 'rmsprop', 'adadelta',
    'adamax', 'adam', 'yamadam', 'amsgrad', 'yogi',
})

_VALID_MUTATIONS = frozenset({
    'onepoint', 'multipoint', 'discretepoint', 'changevar',
    'changefunc', 'insertsubtree', 'replacesubtree', 'removesubtree',
})

_DEFAULT_MUTATION: dict[str, float] = {
    'onepoint': 1.0, 'multipoint': 1.0, 'discretepoint': 1.0,
    'changevar': 1.0, 'changefunc': 1.0, 'insertsubtree': 1.0,
    'replacesubtree': 1.0, 'removesubtree': 1.0,
}


class SymbolicRegressor(BaseEstimator, RegressorMixin):
    """Genetic programming-based symbolic regression estimator.

    Uses the Operon C++ library to evolve mathematical expressions that
    fit the training data. Supports single and multi-objective optimization,
    multiple tree initialization methods, and various coefficient optimizers.

    Parameters
    ----------
    allowed_symbols : str, default='add,sub,mul,div,constant,variable'
        Comma-separated list of allowed primitive functions. Available
        symbols: add, sub, mul, div, fmin, fmax, aq, pow, powabs, abs,
        acos, asin, atan, cbrt, ceil, cos, cosh, exp, floor, log, logabs,
        log1p, sin, sinh, sqrt, sqrtabs, tan, tanh, square, constant,
        variable.

    symbolic_mode : bool, default=False
        If True, use integer coefficients and disable coefficient
        optimization. Useful for discovering exact symbolic expressions.

    crossover_probability : float, default=1.0
        Probability of applying crossover to generate offspring.

    crossover_internal_probability : float, default=0.9
        Probability of selecting an internal (non-leaf) node as the
        crossover point.

    mutation : dict or None, default=None
        Dictionary mapping mutation operator names to their relative
        weights. None uses all operators with equal weight. Valid keys:
        onepoint, multipoint, discretepoint, changevar, changefunc,
        insertsubtree, replacesubtree, removesubtree.

    mutation_probability : float, default=0.25
        Probability of applying mutation to generate offspring.

    offspring_generator : str, default='basic'
        Strategy for generating offspring. Options: 'basic', 'os'
        (offspring selection), 'brood', 'poly' (polygenic).

    reinserter : str, default='keep-best'
        Strategy for reinserting offspring into the population. Options:
        'keep-best', 'replace-worst'.

    objectives : list of str or None, default=None
        Fitness objectives. None defaults to ['r2']. For multi-objective,
        NSGA-II is used automatically. Options: 'r2', 'c2', 'nmse',
        'rmse', 'mse', 'mae', 'length', 'shape', 'diversity', 'mdl',
        'poisson', 'gauss'.

    optimizer : str, default='lm'
        Coefficient optimizer. Options: 'lm' (Levenberg-Marquardt),
        'lbfgs', 'sgd'.

    optimizer_likelihood : str, default='gaussian'
        Likelihood function for LBFGS/SGD optimizers. Options:
        'gaussian', 'poisson', 'poisson_log'.

    optimizer_batch_size : int, default=0
        Batch size for the optimizer. 0 uses all training samples.

    optimizer_iterations : int, default=0
        Number of optimizer iterations per individual. 0 disables
        coefficient optimization.

    local_search_probability : float, default=1.0
        Probability of applying local search (coefficient optimization).

    lamarckian_probability : float, default=1.0
        Probability of keeping optimized coefficients (Lamarckian).

    sgd_update_rule : str, default='constant'
        SGD learning rate update rule. Options: 'constant', 'momentum',
        'rmsprop', 'adadelta', 'adamax', 'adam', 'yamadam', 'amsgrad',
        'yogi'.

    sgd_learning_rate : float, default=0.01
        Initial learning rate for SGD.

    sgd_beta : float, default=0.9
        First momentum parameter for SGD update rules.

    sgd_beta2 : float, default=0.999
        Second momentum parameter for SGD update rules.

    sgd_epsilon : float, default=1e-6
        Epsilon parameter for numerical stability in SGD.

    sgd_debias : bool, default=False
        Whether to use debiasing in the Yogi update rule.

    max_length : int, default=50
        Maximum allowed tree length (number of nodes).

    max_depth : int, default=10
        Maximum allowed tree depth.

    initialization_method : str, default='btc'
        Tree initialization method. Options: 'btc' (balanced tree
        creator), 'ptc2' (probabilistic tree creator), 'koza' (grow).

    initialization_max_length : int, default=10
        Maximum tree length during initialization.

    initialization_max_depth : int, default=5
        Maximum tree depth during initialization.

    female_selector : str, default='tournament'
        Parent selection method for the female parent. Options:
        'tournament', 'proportional', 'random'.

    male_selector : str, default='tournament'
        Parent selection method for the male parent. Options:
        'tournament', 'proportional', 'random'.

    population_size : int, default=1000
        Number of individuals in the population.

    pool_size : int or None, default=None
        Number of offspring generated per generation. None defaults
        to population_size.

    generations : int, default=1000
        Maximum number of generations.

    max_evaluations : int, default=1_000_000
        Maximum number of fitness evaluations.

    max_selection_pressure : int, default=100
        Maximum selection pressure for offspring selection generator.

    comparison_factor : float, default=0
        Comparison factor for offspring selection generator.

    brood_size : int, default=10
        Brood size for brood and polygenic generators.

    tournament_size : int, default=5
        Tournament size for tournament selection.

    irregularity_bias : float, default=0.0
        Bias towards irregular trees in tree creation.

    epsilon : float, default=1e-5
        Convergence threshold.

    model_selection_criterion : str, default='minimum_description_length'
        Criterion for selecting the best model from the Pareto front.
        Must match a key in the solution stats: 'mean_squared_error',
        'minimum_description_length', 'bayesian_information_criterion',
        'akaike_information_criterion'.

    add_model_scale_term : bool, default=True
        Whether to include a scale (slope) term via linear scaling.
        Automatically set to True when using r2, nmse, rmse, mse, or
        mae objectives.

    add_model_intercept_term : bool, default=True
        Whether to include an intercept (offset) term via linear scaling.
        Automatically set to True when using r2, nmse, rmse, mse, or
        mae objectives.

    uncertainty : list of float or None, default=None
        Noise standard deviation estimate. None defaults to [1].
        Required for MDL and Gaussian likelihood objectives; a warning
        is issued if left at the default.

    n_threads : int, default=1
        Number of threads for parallel evaluation.

    max_time : int or None, default=None
        Maximum wall-clock time in seconds. None means no time limit.

    random_state : int, np.random.Generator, or None, default=None
        Random seed for reproducibility. None uses a random seed.

    warm_start : bool, default=False
        If True, reuse individuals from a previous fit as the initial
        population.

    Attributes
    ----------
    model_ : operon.Tree
        The best model found during fitting.

    pareto_front_ : list of dict
        Statistics for each model on the Pareto front.

    stats_ : dict
        Run statistics including generation count, evaluation count, etc.

    individuals_ : list of operon.Individual
        The final population (used for warm_start).

    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : ndarray of str
        Feature names seen during fit (only set when input has column
        names, e.g. a pandas DataFrame).

    Examples
    --------
    >>> from pyoperon.sklearn import SymbolicRegressor
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> reg = SymbolicRegressor(population_size=100, generations=10)
    >>> reg.fit(X, y)
    SymbolicRegressor(generations=10, population_size=100)
    """

    def __init__(
        self,
        allowed_symbols: str = 'add,sub,mul,div,constant,variable',
        symbolic_mode: bool = False,
        crossover_probability: float = 1.0,
        crossover_internal_probability: float = 0.9,
        mutation: dict[str, float] | None = None,
        mutation_probability: float = 0.25,
        offspring_generator: str = 'basic',
        reinserter: str = 'keep-best',
        objectives: list[str] | None = None,
        optimizer: str = 'lm',
        optimizer_likelihood: str = 'gaussian',
        optimizer_batch_size: int = 0,
        optimizer_iterations: int = 0,
        local_search_probability: float = 1.0,
        lamarckian_probability: float = 1.0,
        sgd_update_rule: str = 'constant',
        sgd_learning_rate: float = 0.01,
        sgd_beta: float = 0.9,
        sgd_beta2: float = 0.999,
        sgd_epsilon: float = 1e-6,
        sgd_debias: bool = False,
        max_length: int = 50,
        max_depth: int = 10,
        initialization_method: str = 'btc',
        initialization_max_length: int = 10,
        initialization_max_depth: int = 5,
        female_selector: str = 'tournament',
        male_selector: str = 'tournament',
        population_size: int = 1000,
        pool_size: int | None = None,
        generations: int = 1000,
        max_evaluations: int = 1_000_000,
        max_selection_pressure: int = 100,
        comparison_factor: float = 0,
        brood_size: int = 10,
        tournament_size: int = 5,
        irregularity_bias: float = 0.0,
        epsilon: float = 1e-5,
        model_selection_criterion: str = 'minimum_description_length',
        add_model_scale_term: bool = True,
        add_model_intercept_term: bool = True,
        uncertainty: list[float] | None = None,
        n_threads: int = 1,
        max_time: int | None = None,
        random_state: int | np.random.Generator | None = None,
        warm_start: bool = False,
    ):
        self.allowed_symbols = allowed_symbols
        self.symbolic_mode = symbolic_mode
        self.crossover_probability = crossover_probability
        self.crossover_internal_probability = crossover_internal_probability
        self.mutation = mutation
        self.mutation_probability = mutation_probability
        self.offspring_generator = offspring_generator
        self.reinserter = reinserter
        self.objectives = objectives
        self.optimizer = optimizer
        self.optimizer_likelihood = optimizer_likelihood
        self.optimizer_batch_size = optimizer_batch_size
        self.optimizer_iterations = optimizer_iterations
        self.local_search_probability = local_search_probability
        self.lamarckian_probability = lamarckian_probability
        self.sgd_update_rule = sgd_update_rule
        self.sgd_learning_rate = sgd_learning_rate
        self.sgd_beta = sgd_beta
        self.sgd_beta2 = sgd_beta2
        self.sgd_epsilon = sgd_epsilon
        self.sgd_debias = sgd_debias
        self.max_length = max_length
        self.max_depth = max_depth
        self.initialization_method = initialization_method
        self.initialization_max_length = initialization_max_length
        self.initialization_max_depth = initialization_max_depth
        self.female_selector = female_selector
        self.male_selector = male_selector
        self.population_size = population_size
        self.pool_size = pool_size
        self.generations = generations
        self.max_evaluations = max_evaluations
        self.max_selection_pressure = max_selection_pressure
        self.comparison_factor = comparison_factor
        self.brood_size = brood_size
        self.tournament_size = tournament_size
        self.irregularity_bias = irregularity_bias
        self.epsilon = epsilon
        self.n_threads = n_threads
        self.model_selection_criterion = model_selection_criterion
        self.add_model_scale_term = add_model_scale_term
        self.add_model_intercept_term = add_model_intercept_term
        self.uncertainty = uncertainty
        self.max_time = max_time
        self.random_state = random_state
        self.warm_start = warm_start

    # --- Parameter resolution and validation (never mutates self) ---

    def _resolve_params(self) -> dict[str, Any]:
        """Resolve None-valued and dependent parameters.

        Returns a dict with resolved values for parameters that have
        None defaults or need runtime computation. The original
        attributes on ``self`` are never modified.
        """
        mutation = dict(self.mutation) if self.mutation is not None else dict(_DEFAULT_MUTATION)
        objectives = list(self.objectives) if self.objectives is not None else ['r2']
        uncertainty = list(self.uncertainty) if self.uncertainty is not None else [1]
        pool_size = self.pool_size if self.pool_size is not None else self.population_size
        max_time = self.max_time if self.max_time is not None else sys.maxsize

        random_state = self.random_state
        if random_state is None:
            random_state = random.getrandbits(64)
        elif isinstance(random_state, np.random.Generator):
            random_state = random_state.bit_generator.random_raw()

        optimizer_iterations = self.optimizer_iterations
        if self.symbolic_mode:
            optimizer_iterations = 0

        add_model_scale_term = self.add_model_scale_term
        add_model_intercept_term = self.add_model_intercept_term
        if any(obj in _SCALING_OBJECTIVES for obj in objectives):
            if not add_model_scale_term or not add_model_intercept_term:
                warnings.warn(
                    'Objectives requiring linear scaling (r2, nmse, rmse, '
                    'mse, mae) need model scaling and intercept terms; '
                    'overriding to True.'
                )
                add_model_scale_term = True
                add_model_intercept_term = True

        return {
            'mutation': mutation,
            'objectives': objectives,
            'uncertainty': uncertainty,
            'pool_size': pool_size,
            'max_time': max_time,
            'random_state': random_state,
            'optimizer_iterations': optimizer_iterations,
            'add_model_scale_term': add_model_scale_term,
            'add_model_intercept_term': add_model_intercept_term,
        }

    def _validate_params(self, resolved: dict[str, Any]) -> None:
        """Validate parameter values, raising ValueError for invalid ones."""
        symbols = [s.strip() for s in self.allowed_symbols.split(',')]
        for s in symbols:
            if s not in _KNOWN_SYMBOLS:
                raise ValueError(
                    f'Unknown symbol: {s!r}. '
                    f'Valid symbols: {", ".join(sorted(_KNOWN_SYMBOLS))}'
                )

        for obj in resolved['objectives']:
            if obj not in _VALID_OBJECTIVES:
                raise ValueError(
                    f'Unknown objective: {obj!r}. '
                    f'Valid objectives: {", ".join(sorted(_VALID_OBJECTIVES))}'
                )

        for mut_name in resolved['mutation']:
            if mut_name not in _VALID_MUTATIONS:
                raise ValueError(
                    f'Unknown mutation: {mut_name!r}. '
                    f'Valid mutations: {", ".join(sorted(_VALID_MUTATIONS))}'
                )

        _check_option = [
            (self.offspring_generator, _VALID_GENERATORS, 'offspring_generator'),
            (self.reinserter, _VALID_REINSERTERS, 'reinserter'),
            (self.female_selector, _VALID_SELECTORS, 'female_selector'),
            (self.male_selector, _VALID_SELECTORS, 'male_selector'),
            (self.initialization_method, _VALID_INIT_METHODS, 'initialization_method'),
            (self.optimizer, _VALID_OPTIMIZERS, 'optimizer'),
            (self.sgd_update_rule, _VALID_SGD_RULES, 'sgd_update_rule'),
        ]
        for value, valid, name in _check_option:
            if value not in valid:
                raise ValueError(
                    f'Invalid {name}: {value!r}. '
                    f'Valid options: {", ".join(sorted(valid))}'
                )

        if self.population_size < 1:
            raise ValueError(
                f'population_size must be >= 1, got {self.population_size}'
            )

        if len(resolved['objectives']) == 0:
            raise ValueError('objectives must not be empty')

    # --- Component construction helpers ---

    @staticmethod
    def _init_primitive_config(allowed_symbols: str) -> int:
        config = 0
        for s in allowed_symbols.split(','):
            s = s.strip()
            config |= int(_KNOWN_SYMBOLS[s])
        return config

    def _init_creator(self, initialization_method, pset, inputs):
        if initialization_method == 'btc':
            return op.BalancedTreeCreator(pset, inputs, self.irregularity_bias)
        elif initialization_method == 'ptc2':
            return op.ProbabilisticTreeCreator(pset, inputs, self.irregularity_bias)
        elif initialization_method == 'koza':
            return op.GrowTreeCreator(pset, inputs)
        raise ValueError(f'Unknown initialization method: {initialization_method!r}')

    def _init_selector(self, selection_method, comp):
        if selection_method == 'tournament':
            selector = op.TournamentSelector(comp)
            selector.TournamentSize = self.tournament_size
            return selector
        elif selection_method == 'proportional':
            return op.ProportionalSelector(comp)
        elif selection_method == 'random':
            return op.RandomSelector()
        raise ValueError(f'Unknown selection method: {selection_method!r}')

    @staticmethod
    def _init_evaluator(objective, problem, dtable, uncertainty):
        if objective == 'r2':
            return op.Evaluator(problem, dtable, op.R2(), True)
        elif objective == 'c2':
            return op.Evaluator(problem, dtable, op.C2(), False)
        elif objective == 'nmse':
            return op.Evaluator(problem, dtable, op.NMSE(), True)
        elif objective == 'rmse':
            return op.Evaluator(problem, dtable, op.RMSE(), True)
        elif objective == 'mse':
            return op.Evaluator(problem, dtable, op.MSE(), True)
        elif objective == 'mae':
            return op.Evaluator(problem, dtable, op.MAE(), True)
        elif objective == 'length':
            return op.LengthEvaluator(problem)
        elif objective == 'shape':
            return op.ShapeEvaluator(problem)
        elif objective == 'diversity':
            return op.DiversityEvaluator(problem)
        elif objective == 'mdl':
            if uncertainty == [1]:
                warnings.warn(
                    'MDL requires an estimate for the noise standard '
                    'deviation. Set the uncertainty parameter for '
                    'reliable results.'
                )
            evaluator = op.MinimumDescriptionLengthEvaluator(problem, dtable, 'gauss')
            evaluator.Sigma = uncertainty
            return evaluator
        elif objective == 'poisson':
            evaluator = op.PoissonLikelihoodEvaluator(problem, dtable)
            evaluator.Sigma = uncertainty
            return evaluator
        elif objective == 'gauss':
            if uncertainty == [1]:
                warnings.warn(
                    'Gaussian likelihood requires an estimate for the '
                    'noise standard deviation. Set the uncertainty '
                    'parameter for reliable results.'
                )
            evaluator = op.GaussianLikelihoodEvaluator(problem, dtable)
            evaluator.Sigma = uncertainty
            return evaluator
        raise ValueError(f'Unknown objective: {objective!r}')

    def _init_generator(self, generator_name, evaluator, crossover, mutator,
                        female_selector, male_selector, coeff_optimizer):
        if male_selector is None:
            male_selector = female_selector

        if generator_name == 'basic':
            return op.BasicOffspringGenerator(
                evaluator, crossover, mutator,
                female_selector, male_selector, coeff_optimizer,
            )
        elif generator_name == 'os':
            generator = op.OffspringSelectionGenerator(
                evaluator, crossover, mutator,
                female_selector, male_selector, coeff_optimizer,
            )
            generator.MaxSelectionPressure = self.max_selection_pressure
            generator.ComparisonFactor = self.comparison_factor
            return generator
        elif generator_name == 'brood':
            generator = op.BroodOffspringGenerator(
                evaluator, crossover, mutator,
                female_selector, male_selector, coeff_optimizer,
            )
            generator.BroodSize = self.brood_size
            return generator
        elif generator_name == 'poly':
            generator = op.PolygenicOffspringGenerator(
                evaluator, crossover, mutator,
                female_selector, male_selector, coeff_optimizer,
            )
            generator.BroodSize = self.brood_size
            return generator
        raise ValueError(f'Unknown generator method: {generator_name!r}')

    @staticmethod
    def _init_reinserter(reinserter_name, comp):
        if reinserter_name == 'replace-worst':
            return op.ReplaceWorstReinserter(comp)
        elif reinserter_name == 'keep-best':
            return op.KeepBestReinserter(comp)
        raise ValueError(f'Unknown reinsertion method: {reinserter_name!r}')

    def _init_mutation(self, mutation_name, inputs, pset, creator,
                       coeff_initializer):
        if mutation_name == 'onepoint':
            mut = (op.UniformIntOnePointMutation() if self.symbolic_mode
                   else op.NormalOnePointMutation())
            if self.symbolic_mode:
                mut.ParameterizeDistribution(-5, +5)
            else:
                mut.ParameterizeDistribution(0, 1)
            return mut
        elif mutation_name == 'multipoint':
            mut = (op.UniformIntMultiPointMutation() if self.symbolic_mode
                   else op.NormalMultiPointMutation())
            if self.symbolic_mode:
                mut.ParameterizeDistribution(-5, +5)
            else:
                mut.ParameterizeDistribution(0, 1)
            return mut
        elif mutation_name == 'changevar':
            return op.ChangeVariableMutation(inputs)
        elif mutation_name == 'changefunc':
            return op.ChangeFunctionMutation(pset)
        elif mutation_name == 'replacesubtree':
            return op.ReplaceSubtreeMutation(
                creator, coeff_initializer, self.max_depth, self.max_length,
            )
        elif mutation_name == 'insertsubtree':
            return op.InsertSubtreeMutation(
                creator, coeff_initializer, self.max_depth, self.max_length,
            )
        elif mutation_name == 'removesubtree':
            return op.RemoveSubtreeMutation(pset)
        elif mutation_name == 'discretepoint':
            mut = op.DiscretePointMutation()
            for c in op.Math.Constants:
                mut.Add(c, 1.0)
            return mut
        raise ValueError(f'Unknown mutation method: {mutation_name!r}')

    def _init_sgd_update_rule(self):
        rule = self.sgd_update_rule
        if rule == 'constant':
            return op.ConstantUpdateRule(0, self.sgd_learning_rate)
        elif rule == 'momentum':
            return op.MomentumUpdateRule(0, self.sgd_learning_rate, self.sgd_beta)
        elif rule == 'rmsprop':
            return op.RmsPropUpdateRule(
                0, self.sgd_learning_rate, self.sgd_beta, self.sgd_epsilon,
            )
        elif rule == 'adadelta':
            return op.AdaDeltaUpdateRule(0, self.sgd_beta, self.sgd_epsilon)
        elif rule == 'adamax':
            return op.AdaMaxUpdateRule(
                0, self.sgd_learning_rate, self.sgd_beta, self.sgd_beta2,
            )
        elif rule == 'adam':
            return op.AdamUpdateRule(
                0, self.sgd_learning_rate, self.sgd_epsilon,
                self.sgd_beta, self.sgd_beta2,
            )
        elif rule == 'yamadam':
            return op.YamAdamUpdateRule(0, self.sgd_epsilon)
        elif rule == 'amsgrad':
            return op.AmsGradUpdateRule(
                0, self.sgd_learning_rate, self.sgd_epsilon,
                self.sgd_beta, self.sgd_beta2,
            )
        elif rule == 'yogi':
            return op.YogiUpdateRule(
                0, self.sgd_learning_rate, self.sgd_epsilon,
                self.sgd_beta, self.sgd_beta2, self.sgd_debias,
            )
        raise ValueError(f'Unknown SGD update rule: {rule!r}')

    @staticmethod
    def _init_optimizer(dtable, problem, optimizer, likelihood, max_iter,
                        batch_size, update_rule=None):
        if optimizer == 'lm':
            return op.LMOptimizer(dtable, problem, max_iter, batch_size)
        elif optimizer == 'lbfgs':
            return op.LBFGSOptimizer(
                dtable, problem, likelihood, max_iter, batch_size,
            )
        elif optimizer == 'sgd':
            return op.SGDOptimizer(
                dtable, problem, update_rule, likelihood, max_iter, batch_size,
            )
        raise ValueError(f'Unknown optimizer: {optimizer!r}')

    # --- Public API ---

    def get_model_string(
        self,
        model,
        precision: int = 3,
        names: list[str] | None = None,
    ) -> str:
        """Return an infix string representation of an operon tree model.

        Parameters
        ----------
        model : operon.Tree
            The tree model to format.
        precision : int, default=3
            Number of decimal places for coefficients.
        names : list of str or None, default=None
            Custom variable names. Must match the number of input
            variables. None uses the default names from training.

        Returns
        -------
        str
            Infix representation of the model.
        """
        check_is_fitted(self)
        hashes = set(x.HashValue for x in model.Nodes if x.IsVariable)
        if len(hashes) == 0:
            warnings.warn('Model contains no variables')

        if names is None:
            return op.InfixFormatter.Format(model, self.variables_, precision)

        if len(names) != len(self.variables_):
            raise ValueError(
                f'Expected {len(self.variables_)} names, got {len(names)}'
            )
        names_map = {k: names[i] for i, k in enumerate(self.variables_)}
        return op.InfixFormatter.Format(model, names_map, precision)

    def fit(self, X, y):
        """Fit the symbolic regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self
            Fitted estimator.
        """
        # Resolve parameters without mutating self
        resolved = self._resolve_params()
        self._validate_params(resolved)

        mutation             = resolved['mutation']
        objectives           = resolved['objectives']
        uncertainty          = resolved['uncertainty']
        pool_size            = resolved['pool_size']
        max_time             = resolved['max_time']
        random_state         = resolved['random_state']
        optimizer_iterations = resolved['optimizer_iterations']
        add_scale            = resolved['add_model_scale_term']
        add_intercept        = resolved['add_model_intercept_term']

        # Record feature names before validation converts DataFrame to array
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns, dtype=object)

        X, y = check_X_y(X, y, accept_sparse=False)
        self.n_features_in_ = X.shape[1]

        # Build dataset and problem
        D                     = np.asfortranarray(np.column_stack((X, y)))
        ds                    = op.Dataset(D)
        target                = max(ds.Variables, key=lambda x: x.Index)
        self.variables_       = {
            v.Hash: v.Name
            for v in sorted(ds.Variables, key=lambda x: x.Index)
            if v.Hash != target.Hash
        }
        inputs                = list(self.variables_)
        training_range        = op.Range(0, ds.Rows)
        test_range            = op.Range(ds.Rows - 1, ds.Rows)
        problem               = op.Problem(ds)
        problem.TrainingRange = training_range
        problem.TestRange     = test_range
        problem.Target        = target
        problem.InputHashes   = inputs

        primitive_set_config  = self._init_primitive_config(self.allowed_symbols)
        problem.ConfigurePrimitiveSet(primitive_set_config)
        pset = problem.PrimitiveSet

        # Tree creator and coefficient initializer
        creator = self._init_creator(self.initialization_method, pset, inputs)
        if self.symbolic_mode:
            coeff_initializer = op.UniformIntCoefficientAnalyzer()
            coeff_initializer.ParameterizeDistribution(-5, +5)
        else:
            coeff_initializer = op.NormalCoefficientInitializer()
            coeff_initializer.ParameterizeDistribution(0, 1)

        single_objective = len(objectives) == 1

        # Dispatch table and optimizer
        dtable = op.DispatchTable()
        update_rule = self._init_sgd_update_rule()
        optimizer = self._init_optimizer(
            dtable, problem, self.optimizer, self.optimizer_likelihood,
            optimizer_iterations, self.optimizer_batch_size, update_rule,
        )

        # Information criteria evaluators (always created for model stats)
        mdl_eval = op.MinimumDescriptionLengthEvaluator(problem, dtable, 'gauss')
        mdl_eval.Sigma = uncertainty
        bic_eval = op.BayesianInformationCriterionEvaluator(problem, dtable)
        aik_eval = op.AkaikeInformationCriterionEvaluator(problem, dtable)

        # Fitness evaluator(s)
        # Keep references to extend lifetimes of C++ objects held by pointer
        evaluators = []
        for obj in objectives:
            eval_ = self._init_evaluator(obj, problem, dtable, uncertainty)
            eval_.Budget = self.max_evaluations
            evaluators.append(eval_)

        if single_objective:
            evaluator = evaluators[0]
        else:
            evaluator = op.MultiEvaluator(problem)
            for eval_ in evaluators:
                evaluator.Add(eval_)
            evaluator.Budget = self.max_evaluations

        # Selection, crossover, mutation, generation
        comparison = (op.SingleObjectiveComparison(0) if single_objective
                      else op.CrowdedComparison())

        female_selector = self._init_selector(self.female_selector, comparison)
        male_selector   = self._init_selector(self.male_selector, comparison)
        reinserter      = self._init_reinserter(self.reinserter, comparison)
        cx              = op.SubtreeCrossover(
            self.crossover_internal_probability,
            self.max_depth, self.max_length,
        )

        mut = op.MultiMutation()
        # Keep references to extend lifetimes (MultiMutation stores pointers)
        mut_list = []
        for k, v in mutation.items():
            m = self._init_mutation(k, inputs, pset, creator, coeff_initializer)
            mut.Add(m, v)
            mut_list.append(m)

        coeff_optimizer = op.CoefficientOptimizer(optimizer)
        generator = self._init_generator(
            self.offspring_generator, evaluator, cx, mut,
            female_selector, male_selector, coeff_optimizer,
        )

        # Tree initializer
        min_arity, _ = pset.FunctionArityLimits()
        tree_initializer = op.UniformLengthTreeInitializer(creator)
        tree_initializer.ParameterizeDistribution(
            min_arity + 1,
            min(self.initialization_max_length, self.max_length),
        )
        tree_initializer.MinDepth = 1
        tree_initializer.MaxDepth = (
            self.initialization_max_depth
            if self.initialization_method == 'koza'
            else 1000
        )

        # Algorithm configuration and instantiation
        config = op.GeneticAlgorithmConfig(
            generations      = self.generations,
            max_evaluations  = self.max_evaluations,
            local_iterations = optimizer_iterations,
            population_size  = self.population_size,
            pool_size        = pool_size,
            p_crossover      = self.crossover_probability,
            p_mutation       = self.mutation_probability,
            p_local          = self.local_search_probability,
            p_lamarck        = self.lamarckian_probability,
            epsilon          = self.epsilon,
            seed             = random_state,
            max_time         = max_time,
        )

        if single_objective:
            gp = op.GeneticProgrammingAlgorithm(
                config, problem, tree_initializer, coeff_initializer,
                generator, reinserter,
            )
        else:
            sorter = op.RankSorter()
            gp = op.NSGA2Algorithm(
                config, problem, tree_initializer, coeff_initializer,
                generator, reinserter, sorter,
            )

        if self.warm_start and hasattr(self, 'is_fitted_') and self.is_fitted_:
            gp.RestoreIndividuals(self.individuals_)
            gp.IsFitted = True

        rng = op.RandomGenerator(np.uint64(config.Seed))
        gp.Run(rng, None, self.n_threads, self.warm_start)

        # --- Extract results ---

        def get_solution_stats(solution):
            y_pred = op.Evaluate(dtable, solution.Genotype, ds, training_range)
            scale, offset = op.FitLeastSquares(y_pred, y)
            nodes = solution.Genotype.Nodes

            if not add_scale:
                scale = 1.0
            if not add_intercept:
                offset = 0.0
            if scale != 1:
                nodes += [op.Node.Constant(scale), op.Node.Mul()]
            if offset != 0:
                nodes += [op.Node.Constant(offset), op.Node.Add()]
            solution.Genotype = op.Tree(nodes).UpdateNodes()

            return {
                'model': op.InfixFormatter.Format(
                    solution.Genotype, self.variables_, 6,
                ),
                'variables': set(
                    self.variables_[x.HashValue]
                    for x in nodes if x.IsVariable
                ),
                'length': len(nodes),
                'complexity': solution.Genotype.AdjustedLength,
                'tree': solution.Genotype,
                'objective_values': evaluator(rng, solution),
                'mean_squared_error': mean_squared_error(
                    y, scale * y_pred + offset,
                ),
                'minimum_description_length': mdl_eval(rng, solution)[0],
                'bayesian_information_criterion': bic_eval(rng, solution)[0],
                'akaike_information_criterion': aik_eval(rng, solution)[0],
            }

        front = [gp.BestModel] if single_objective else gp.BestFront
        self.pareto_front_ = [get_solution_stats(m) for m in front]
        best = min(
            self.pareto_front_,
            key=lambda x: x[self.model_selection_criterion],
        )
        self.model_ = best['tree']

        self.stats_ = {
            'model_length': best['length'],
            'model_complexity': best['complexity'],
            'generations': gp.Generation,
            'evaluation_count': evaluator.CallCount,
            'residual_evaluations': evaluator.ResidualEvaluations,
            'jacobian_evaluations': evaluator.JacobianEvaluations,
            'random_state': random_state,
        }

        self.individuals_ = list(gp.Individuals)
        self.is_fitted_ = True
        return self

    def evaluate_model(self, model, X) -> np.ndarray:
        """Evaluate an arbitrary tree model on input data.

        Parameters
        ----------
        model : operon.Tree
            The tree model to evaluate.
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        X = check_array(X, accept_sparse=False)
        ds = op.Dataset(np.asfortranarray(X))
        rg = op.Range(0, ds.Rows)
        dtable = op.DispatchTable()
        return op.Evaluate(dtable, model, ds, rg).reshape(-1)

    def predict(self, X) -> np.ndarray:
        """Predict using the fitted symbolic regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self)
        return self.evaluate_model(self.model_, X)
