# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

import sys
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import random
import operon.pyoperon as op

class SymbolicRegressor(BaseEstimator, RegressorMixin):
    """ Builds a symbolic regression model.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------

    Examples
    --------
    >>> from operon import SymbolicRegressor
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = SymbolicRegressor()
    >>> estimator.fit(X, y)
    """
    def __init__(self,
        allowed_symbols                = 'add,sub,mul,div,constant,variable',
        symbolic_mode                  = None,
        crossover_probability          = 1.0,
        crossover_internal_probability = 0.9,
        mutation                       = { 'onepoint' : 1.0, 'discretepoint' : 1.0, 'changevar' : 1.0, 'changefunc' : 1.0, 'insertsubtree' : 1.0, 'replacesubtree' : 1.0, 'removesubtree' : 1.0 },
        mutation_probability           = 0.25,
        offspring_generator            = 'basic',
        reinserter                     = 'replace-worst',
        objectives                     = ['r2'],
        max_length                     = 50,
        max_depth                      = 10,
        initialization_method          = 'btc',
        female_selector                = 'tournament',
        male_selector                  = 'tournament',
        population_size                = 1000,
        pool_size                      = None,
        generations                    = 1000,
        max_evaluations                = int(1000 * 1000),
        local_iterations               = 0,
        max_selection_pressure         = 100,
        comparison_factor              = 0,
        brood_size                     = 10,
        tournament_size                = 5,
        irregularity_bias              = 0.0,
        epsilon                        = 1e-5,
        n_threads                      = 1,
        time_limit                     = None,
        random_state                   = None
        ):

        # validate parameters
        self.allowed_symbols           = 'add,sub,mul,div,constant,variable' if allowed_symbols is None else allowed_symbols
        self.symbolic_mode             = False if symbolic_mode is None else symbolic_mode
        self.crossover_probability     = 1.0 if crossover_probability is None else crossover_probability
        self.crossover_internal_probability = 0.9 if crossover_internal_probability is None else crossover_internal_probability
        self.mutation                  = { 'onepoint' : 1.0, 'discretepoint': 1.0, 'changevar' : 1.0, 'changefunc' : 1.0, 'insertsubtree' : 1.0, 'replacesubtree' : 1.0, 'removesubtree' : 1.0 } if mutation is None else mutation
        self.mutation_probability      = 0.25 if mutation_probability is None else mutation_probability
        self.offspring_generator       = 'basic' if offspring_generator is None else offspring_generator
        self.reinserter                = 'replace-worst' if reinserter is None else reinserter
        self.objectives                = ['r2'] if objectives is None else objectives
        self.max_length                = 50 if max_length is None else int(max_length)
        self.max_depth                 = 10 if max_depth is None else int(max_depth)
        self.initialization_method     = 'btc' if initialization_method is None else initialization_method
        self.female_selector           = 'tournament' if female_selector is None else female_selector
        self.male_selector             = 'tournament' if male_selector is None else male_selector
        self.population_size           = 1000 if population_size is None else int(population_size)
        self.pool_size                 = population_size if pool_size is None else int(pool_size)
        self.generations               = 1000 if generations is None else int(generations)
        self.max_evaluations           = 1000000 if max_evaluations is None else int(max_evaluations)
        self.local_iterations          = 0 if local_iterations is None else int(local_iterations)
        self.max_selection_pressure    = 100 if max_selection_pressure is None else int(max_selection_pressure)
        self.comparison_factor         = 0 if comparison_factor is None else comparison_factor
        self.brood_size                = 10 if brood_size is None else int(brood_size)
        self.tournament_size           = 5 if tournament_size is None else tournament_size # todo: set for both parent selectors
        self.irregularity_bias         = 0.0 if irregularity_bias is None else irregularity_bias
        self.epsilon                   = 1e-5 if epsilon is None else epsilon
        self.n_threads                 = 1 if n_threads is None else int(n_threads)
        self.time_limit                = sys.maxsize if time_limit is None else int(time_limit)
        self.random_state              = random_state
        self._model                    = None
        self._model_vars               = {}
        self._pareto_front             = [ ]
        self._interpreter              = op.Interpreter()


    def __init_primitive_config(self, allowed_symbols):
        symbols = allowed_symbols.split(',')

        known_symbols = {
            'add' : op.NodeType.Add,
            'mul' : op.NodeType.Mul,
            'sub' : op.NodeType.Sub,
            'div' : op.NodeType.Div,
            'fmin' : op.NodeType.Fmin,
            'fmax' : op.NodeType.Fmax,
            'aq' : op.NodeType.Aq,
            'pow' : op.NodeType.Pow,
            'abs' : op.NodeType.Abs,
            'acos' : op.NodeType.Acos,
            'asin' : op.NodeType.Asin,
            'atan' : op.NodeType.Atan,
            'cbrt' : op.NodeType.Cbrt,
            'ceil' : op.NodeType.Ceil,
            'cos' : op.NodeType.Cos,
            'cosh' : op.NodeType.Cosh,
            'exp' : op.NodeType.Exp,
            'floor' : op.NodeType.Floor,
            'log' : op.NodeType.Log,
            'logabs' : op.NodeType.Logabs,
            'log1p' : op.NodeType.Log1p,
            'sin' : op.NodeType.Sin,
            'sinh' : op.NodeType.Sinh,
            'sqrt' : op.NodeType.Sqrt,
            'sqrtabs' : op.NodeType.Sqrtabs,
            'tan' : op.NodeType.Tan,
            'tanh' : op.NodeType.Tanh,
            'square' : op.NodeType.Square,
            'constant' : op.NodeType.Constant,
            'variable' : op.NodeType.Variable,
        }

        config = op.NodeType(0)
        for s in symbols:
            if s in known_symbols:
                config |= known_symbols[s]
            else:
                raise ValueError('Unknown symbol type {}'.format(s))

        return config


    def __init_creator(self, initialization_method, pset, inputs):
        if initialization_method == 'btc':
            return op.BalancedTreeCreator(pset, inputs, self.irregularity_bias)

        elif initialization_method == 'ptc2':
            return op.ProbabilisticTreeCreator(pset, inputs, self.irregularity_bias)

        elif initialization_method == 'koza':
            return op.GrowTreeCreator(pset, inputs)

        raise ValueError('Unknown initialization method {}'.format(initialization_method))


    def __init_selector(self, selection_method, comp):
        if selection_method == 'tournament':
            selector = op.TournamentSelector(comp)
            selector.TournamentSize = self.tournament_size
            return selector

        elif selection_method == 'proportional':
            selector = op.ProportionalSelector(comp)
            return selector

        elif selection_method == 'random':
            selector = op.RandomSelector()
            return selector

        raise ValueError('Unknown selection method {}'.format(selection_method))


    def __init_evaluator(self, objective, problem, interpreter):
        if objective == 'r2':
            err = op.R2()
            return op.Evaluator(problem, interpreter, err, True), err

        elif objective == 'c2':
            err = op.C2()
            return op.Evaluator(problem, interpreter, err, False), err

        elif objective == 'nmse':
            err = op.NMSE()
            return op.Evaluator(problem, interpreter, err, True), err

        elif objective == 'rmse':
            err = op.RMSE()
            return op.Evaluator(problem, interpreter, err, True), err

        elif objective == 'mse':
            err = op.MSE()
            return op.Evaluator(problem, interpreter, err, True), err

        elif objective == 'mae':
            err = op.MAE()
            return op.Evaluator(problem, interpreter, err, True), err

        elif objective == 'length':
            return op.LengthEvaluator(problem), None

        elif objective == 'shape':
            return op.ShapeEvaluator(problem), None

        raise ValueError('Unknown objective {}'.format(objectives))


    def __init_generator(self, generator_name, evaluator, crossover, mutator, female_selector, male_selector):
        if male_selector is None:
            male_selector = female_selector

        if generator_name == 'basic':
            return op.BasicOffspringGenerator(evaluator, crossover, mutator, female_selector, male_selector)

        elif generator_name == 'os':
            generator = op.OffspringSelectionGenerator(evaluator, crossover, mutator, female_selector, male_selector)
            generator.MaxSelectionPressure = self.max_selection_pressure
            generator.ComparisonFactor = self.comparison_factor
            return generator

        elif generator_name == 'brood':
            generator = op.BroodOffspringGenerator(evaluator, crossover, mutator, female_selector, male_selector)
            generator.BroodSize = self.brood_size
            return generator

        elif generator_name == 'poly':
            generator = op.PolygenicOffspringGenerator(evaluator, crossover, mutator, female_selector, male_selector)
            generator.BroodSize = self.brood_size
            return generator

        raise ValueError('Unknown generator method {}'.format(generator_name))


    def __init_reinserter(self, reinserter_name, comp):
        if reinserter_name == 'replace-worst':
            return op.ReplaceWorstReinserter(comp)

        elif reinserter_name == 'keep-best':
            return op.KeepBestReinserter(comp)

        raise ValueError('Unknown reinsertion method {}'.format(reinserter_name))


    def __init_mutation(self, mutation_name, inputs, pset, creator, coeff_initializer):
        if mutation_name == 'onepoint':
            mut = op.UniformIntOnePointMutation() if self.symbolic_mode else op.NormalOnePointMutation()
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
            return op.ReplaceSubtreeMutation(creator, coeff_initializer, self.max_depth, self.max_length)

        elif mutation_name == 'insertsubtree':
            return op.InsertSubtreeMutation(creator, coeff_initializer, self.max_depth, self.max_length)

        elif mutation_name == 'removesubtree':
            return op.RemoveSubtreeMutation(pset)

        elif mutation_name == 'discretepoint':
            mut = op.DiscretePointMutation()
            for c in op.Math.Constants:
                mut.Add(c, 1.0)
            return mut

        raise ValueError('Unknown mutation method {}'.format(mutation_name))


    def get_model_string(self, precision):
        if len(self._model_vars) == 0:
            print('warning: model contains no variables', file=sys.stderr)
        return op.InfixFormatter.Format(self._model, self._model_vars, precision)


    def get_pareto_front(self, precision):
        front = []
        for (model, model_vars) in self._pareto_front:
            front.append(op.InfixFormatter.Format(model, model_vars, precision))

        return front


    def fit(self, X, y):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y                  = check_X_y(X, y, accept_sparse=False)
        D                     = np.column_stack((X, y))

        ds                    = op.Dataset(D)
        target                = max(ds.Variables, key=lambda x: x.Index) # last column is the target

        inputs                = op.VariableCollection(v for v in ds.Variables if v.Index != target.Index)
        training_range        = op.Range(0, ds.Rows)
        test_range            = op.Range(ds.Rows-1, ds.Rows) # hackish, because it can't be empty
        problem               = op.Problem(ds, inputs, target.Name, training_range, test_range)

        pset                  = op.PrimitiveSet()
        pcfg                  = self.__init_primitive_config(self.allowed_symbols)
        pset.SetConfig(pcfg)

        creator               = self.__init_creator(self.initialization_method, pset, inputs)
        coeff_initializer     = op.UniformIntCoefficientAnalyzer() if self.symbolic_mode else op.NormalCoefficientInitializer()

        if self.symbolic_mode:
            coeff_initializer.ParameterizeDistribution(-5, +5)
        else:
            coeff_initializer.ParameterizeDistribution(0, 1)

        single_objective      = True if len(self.objectives) == 1 else False

        error_metrics = [] # placeholder for the error metric
        evaluators = [] # placeholder for the evaluator(s)

        for obj in self.objectives:
            eval_, err_  = self.__init_evaluator(obj, problem, self._interpreter)
            eval_.Budget = self.max_evaluations
            eval_.LocalOptimizationIterations = self.local_iterations
            evaluators.append(eval_)
            error_metrics.append(err_)

        if single_objective:
            evaluator = evaluators[0]
        else:
            evaluator = op.MultiEvaluator(problem)
            for eval_ in evaluators:
                evaluator.Add(eval_)
            evaluator.LocalOptimizationIterations = self.local_iterations
            evaluator.Budget = self.max_evaluations

        comparison            = op.SingleObjectiveComparison(0) if single_objective else op.CrowdedComparison()

        female_selector       = self.__init_selector(self.female_selector, comparison)
        male_selector         = self.__init_selector(self.male_selector, comparison)
        reinserter            = self.__init_reinserter(self.reinserter, comparison)
        cx                    = op.SubtreeCrossover(self.crossover_internal_probability, self.max_depth, self.max_length)

        mut                   = op.MultiMutation()
        mut_list = [] # this list is needed as a placeholder to keep alive the mutation operators objects (since the multi-mutation only stores references)
        for k in self.mutation:
            v = self.mutation[k]
            m = self.__init_mutation(k, inputs, pset, creator, coeff_initializer)
            mut.Add(m, v)
            mut_list.append(m)

        generator             = self.__init_generator(self.offspring_generator, evaluator, cx, mut, female_selector, male_selector)

        min_arity, max_arity  = pset.FunctionArityLimits()
        tree_initializer      = op.UniformLengthTreeInitializer(creator)
        tree_initializer.ParameterizeDistribution(min_arity+1, self.max_length)
        tree_initializer.MinDepth = 1

        # btc and ptc2 do not need a depth restriction
        tree_initializer.MaxDepth = self.max_depth if self.initialization_method == 'koza' else 1000


        if self.random_state is None:
            self.random_state = random.getrandbits(64)

        config                = op.GeneticAlgorithmConfig(
                                    generations      = self.generations,
                                    max_evaluations  = self.max_evaluations,
                                    local_iterations = self.local_iterations,
                                    population_size  = self.population_size,
                                    pool_size        = self.pool_size,
                                    p_crossover      = self.crossover_probability,
                                    p_mutation       = self.mutation_probability,
                                    epsilon          = self.epsilon,
                                    seed             = self.random_state,
                                    time_limit       = self.time_limit
                                    )

        sorter                = None if single_objective else op.RankSorter()
        gp                    = op.GeneticProgrammingAlgorithm(problem, config, tree_initializer, coeff_initializer, generator, reinserter) if single_objective \
                                else op.NSGA2Algorithm(problem, config, tree_initializer, coeff_initializer, generator, reinserter, sorter)
        rng                   = op.RomuTrio(np.uint64(config.Seed))

        gp.Run(rng, None, self.n_threads)
        comp                  = op.SingleObjectiveComparison(0)
        best                  = gp.BestModel()
        nodes                 = best.Genotype.Nodes
        n_vars                = len([ node for node in nodes if node.IsVariable ])

        # add four nodes at the top of the tree for linear scaling
        y_pred                = op.Evaluate(self._interpreter, best.Genotype, ds, training_range)
        scale, offset         = op.FitLeastSquares(y_pred, y)
        nodes.extend([ op.Node.Constant(scale), op.Node.Mul(), op.Node.Constant(offset), op.Node.Add() ])

        self._model           = op.Tree(nodes).UpdateNodes()

        get_model_vars = lambda model: { node.HashValue : ds.GetVariable(node.HashValue).Name for node in model.Nodes if node.IsVariable }

        # update model vars dictionary
        self._model_vars = get_model_vars(self._model)

        self._pareto_front = [ (self._model, self._model_vars) ] if single_objective else [ (x.Genotype, get_model_vars(x.Genotype)) for x in gp.BestFront ]

        self._stats = {
            'model_length':        self._model.Length - 4, # do not count scaling nodes?
            'model_complexity':    self._model.Length - 4 + 2 * n_vars,
            'generations':         gp.Generation,
            'fitness_evaluations': evaluator.EvaluationCount,
            'residual_evaluations': evaluator.ResidualEvaluations,
            'jacobian_evaluations': evaluator.JacobianEvaluations,
            'random_state':        self.random_state
        }

        self.is_fitted_ = True
        # `fit` should always return `self`
        return self


    def evaluate_model(self, model, X):
        X = check_array(X, accept_sparse=False)
        ds = op.Dataset(X)
        rg = op.Range(0, ds.Rows)
        return op.Evaluate(self._interpreter, model, ds, rg)


    def predict(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        check_is_fitted(self)
        return self.evaluate_model(self._model, X).reshape(-1, 1)

