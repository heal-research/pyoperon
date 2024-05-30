# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

import sys
import random
import numpy as np
import pyoperon as op

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import mean_squared_error

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
        symbolic_mode                  = False,
        crossover_probability          = 1.0,
        crossover_internal_probability = 0.9,
        mutation                       = { 'onepoint' : 1.0, 'discretepoint' : 1.0, 'changevar' : 1.0, 'changefunc' : 1.0, 'insertsubtree' : 1.0, 'replacesubtree' : 1.0, 'removesubtree' : 1.0 },
        mutation_probability           = 0.25,
        offspring_generator            = 'basic',
        reinserter                     = 'keep-best',
        objectives                     = ['r2'],
        optimizer                      = 'lm',
        optimizer_likelihood           = 'gaussian',
        optimizer_batch_size           = 0,
        optimizer_iterations           = 0,
        local_search_probability       = 1.0,
        lamarckian_probability         = 1.0,
        sgd_update_rule                = 'constant',
        sgd_learning_rate              = 0.01,
        sgd_beta                       = 0.9,
        sgd_beta2                      = 0.999,
        sgd_epsilon                    = 1e-6,
        max_length                     = 50,
        max_depth                      = 10,
        initialization_method          = 'btc',
        initialization_max_length      = 10,
        initialization_max_depth       = 5,
        female_selector                = 'tournament',
        male_selector                  = 'tournament',
        population_size                = 1000,
        pool_size                      = None,
        generations                    = 1000,
        max_evaluations                = int(1e6),
        max_selection_pressure         = 100,
        comparison_factor              = 0,
        brood_size                     = 10,
        tournament_size                = 5,
        irregularity_bias              = 0.0,
        epsilon                        = 1e-5,
        model_selection_criterion      = 'minimum_description_length',
        add_model_scale_term           = True,
        add_model_intercept_term       = True,
        uncertainty                    = [1],
        n_threads                      = 1,
        time_limit                     = None,
        random_state                   = None
        ):

        self.allowed_symbols           = allowed_symbols
        self.symbolic_mode             = symbolic_mode
        self.crossover_probability     = crossover_probability
        self.crossover_internal_probability = crossover_internal_probability
        self.mutation                  = mutation
        self.mutation_probability      = mutation_probability
        self.offspring_generator       = offspring_generator
        self.reinserter                = reinserter
        self.objectives                = objectives
        self.optimizer                 = optimizer
        self.optimizer_likelihood      = optimizer_likelihood
        self.optimizer_batch_size      = optimizer_batch_size
        self.optimizer_iterations      = optimizer_iterations
        self.local_search_probability  = local_search_probability
        self.lamarckian_probability    = lamarckian_probability
        self.sgd_update_rule           = sgd_update_rule
        self.sgd_learning_rate         = sgd_learning_rate
        self.sgd_beta                  = sgd_beta
        self.sgd_beta2                 = sgd_beta2
        self.sgd_epsilon               = sgd_epsilon
        self.max_length                = max_length
        self.max_depth                 = max_depth
        self.initialization_method     = initialization_method
        self.initialization_max_length = initialization_max_length
        self.initialization_max_depth  = initialization_max_depth
        self.female_selector           = female_selector
        self.male_selector             = male_selector
        self.population_size           = population_size
        self.pool_size                 = pool_size
        self.generations               = generations
        self.max_evaluations           = max_evaluations
        self.max_selection_pressure    = max_selection_pressure
        self.comparison_factor         = comparison_factor
        self.brood_size                = brood_size
        self.tournament_size           = tournament_size # todo: set for both parent selectors
        self.irregularity_bias         = irregularity_bias
        self.epsilon                   = epsilon
        self.n_threads                 = n_threads
        self.model_selection_criterion = model_selection_criterion
        self.add_model_scale_term      = add_model_scale_term
        self.add_model_intercept_term  = add_model_intercept_term
        self.uncertainty               = uncertainty
        self.time_limit                = time_limit
        self.random_state              = random_state


    def __check_parameters(self):
        check = lambda x, y: y if x is None else x
        self.allowed_symbols                = check(self.allowed_symbols, 'add,sub,mul,div,constant,variable')
        self.symbolic_mode                  = check(self.symbolic_mode, False)
        self.crossover_probability          = check(self.crossover_probability, 1.0)
        self.crossover_internal_probability = check(self.crossover_internal_probability, 0.9)
        self.mutation                       = check(self.mutation, { 'onepoint': 1.0, 'discretepoint': 1.0, 'changevar': 1.0, 'changefunc': 1.0, 'insertsubtree': 1.0, 'removesubtree': 1.0 })
        self.mutation_probability           = check(self.mutation_probability, 0.25)
        self.offspring_generator            = check(self.offspring_generator, 'basic')
        self.reinserter                     = check(self.reinserter, 'keep-best')
        self.objectives                     = check(self.objectives, [ 'r2' ])
        self.optimizer                      = check(self.optimizer, 'lbfgs')
        self.optimizer_likelihood           = check(self.optimizer_likelihood, 'gaussian')
        self.optimizer_batch_size           = check(self.optimizer_batch_size, 0)
        self.optimizer_iterations           = check(self.optimizer_iterations, 0)
        self.local_search_probability       = check(self.local_search_probability, 1.0)
        self.lamarckian_probability         = check(self.lamarckian_probability, 1.0)
        self.sgd_update_rule                = check(self.sgd_update_rule, 'constant')
        self.sgd_learning_rate              = check(self.sgd_learning_rate, 0.01)
        self.sgd_beta                       = check(self.sgd_beta, 0.9)
        self.sgd_beta2                      = check(self.sgd_beta2, 0.999)
        self.sgd_epsilon                    = check(self.sgd_epsilon, 1e-6)
        self.max_length                     = check(self.max_length, 50)
        self.max_depth                      = check(self.max_depth, 10)
        self.initialization_method          = check(self.initialization_method, 'btc')
        self.initialization_max_length      = check(self.initialization_max_length, 10)
        self.initialization_max_depth       = check(self.initialization_max_depth, 5)
        self.female_selector                = check(self.female_selector, 'tournament')
        self.male_selector                  = check(self.male_selector, self.female_selector)
        self.population_size                = check(self.population_size, 1000)
        self.pool_size                      = check(self.pool_size, self.population_size)
        self.generations                    = check(self.generations, 1000)
        self.max_evaluations                = check(self.max_evaluations, int(1e6))
        self.max_selection_pressure         = check(self.max_selection_pressure, 100)
        self.comparison_factor              = check(self.comparison_factor, 0)
        self.brood_size                     = check(self.brood_size, 10)
        self.tournament_size                = check(self.tournament_size, 3)
        self.irregularity_bias              = check(self.irregularity_bias, 0.0)
        self.epsilon                        = check(self.epsilon, 1e-5)
        self.model_selection_criterion      = check(self.model_selection_criterion, 'minimum_description_length')
        self.add_model_scale_term           = check(self.add_model_scale_term, True)
        self.add_model_intercept_term       = check(self.add_model_intercept_term, True)
        self.uncertainty                    = check(self.uncertainty, [1])
        self.n_threads                      = check(self.n_threads, 1)
        self.time_limit                     = check(self.time_limit, sys.maxsize)
        self.random_state                   = check(self.random_state, random.getrandbits(64))


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


    def __init_evaluator(self, objective, problem, dtable):
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
            evaluator = op.MinimumDescriptionLengthEvaluator(problem, dtable)
            evaluator.Sigma = self.uncertainty
            return evaluator

        elif objective == 'poisson':
            evaluator = op.PoissonLikelihoodEvaluator(problem, dtable)
            evaluator.Sigma = self.uncertainty
            return evaluator

        elif objective == 'gauss':
            evaluator = op.GaussianLikelihoodEvaluator(problem, dtable)
            evaluator.Sigma = self.uncertainty
            return evaluator

        raise ValueError('Unknown objective {}'.format(objective))


    def __init_generator(self, generator_name, evaluator, crossover, mutator, female_selector, male_selector, coeff_optimizer):
        if male_selector is None:
            male_selector = female_selector

        if generator_name == 'basic':
            return op.BasicOffspringGenerator(evaluator, crossover, mutator, female_selector, male_selector, coeff_optimizer)

        elif generator_name == 'os':
            generator = op.OffspringSelectionGenerator(evaluator, crossover, mutator, female_selector, male_selector, coeff_optimizer)
            generator.MaxSelectionPressure = self.max_selection_pressure
            generator.ComparisonFactor = self.comparison_factor
            return generator

        elif generator_name == 'brood':
            generator = op.BroodOffspringGenerator(evaluator, crossover, mutator, female_selector, male_selector, coeff_optimizer)
            generator.BroodSize = self.brood_size
            return generator

        elif generator_name == 'poly':
            generator = op.PolygenicOffspringGenerator(evaluator, crossover, mutator, female_selector, male_selector, coeff_optimizer)
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


    def __init_sgd_update_rule(self):
        if self.sgd_update_rule == 'constant':
            return op.ConstantUpdateRule(0, self.sgd_learning_rate)
        elif self.sgd_update_rule == 'momentum':
            return op.MomentumUpdateRule(0, self.sgd_learning_rate, self.sgd_beta)
        elif self.sgd_update_rule == 'rmsprop':
            return op.RmsPropUpdateRule(0, self.sgd_learning_rate, self.sgd_beta, self.sgd_epsilon)
        elif self.sgd_update_rule == 'adamax':
            return op.AdaMaxUpdateRule(0, self.sgd_learning_rate, self.sgd_beta, self.sgd_beta2)
        elif self.sgd_update_rule == 'amsgrad':
            return op.AmsGradUpdateRule(0, self.sgd_learning_rate, self.sgd_epsilon, self.sgd_beta, self.sgd_beta2)

        raise ValueError('Unknown update rule {}'.format(self.sgd_update_rule))


    def __init_optimizer(self, dtable, problem, optimizer, likelihood, max_iter, batch_size, update_rule = None):
        if optimizer == 'lm':
            return op.LMOptimizer(dtable, problem, max_iter, batch_size)
        elif optimizer == 'lbfgs':
            return op.LBFGSOptimizer(dtable, problem, likelihood, max_iter, batch_size)
        elif optimizer == 'sgd':
            return op.SGDOptimizer(dtable, problem, update_rule, likelihood, max_iter, batch_size)

        raise ValueError('Unknown optimizer {}'.format(optimizer))


    def get_model_string(self, model, precision=3, names=None):
        """Returns an infix string representation of an operon tree model"""
        hashes = set(x.HashValue for x in model.Nodes if x.IsVariable)
        if len(hashes) == 0:
            print('warning: model contains no variables', file=sys.stderr)

        if names is None:
            return op.InfixFormatter.Format(model, self.variables_, precision)

        else:
            assert(len(names) == len(self.variables_))
            names_map = { k : names[i] for i, k in enumerate(self.variables_) }
            return op.InfixFormatter.Format(model, names_map, precision)


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

        # first make sure that the parameters are proper
        self.__check_parameters()

        X, y                  = check_X_y(X, y, accept_sparse=False)
        D                     = np.column_stack((X, y))

        ds                    = op.Dataset(D)
        target                = max(ds.Variables, key=lambda x: x.Index) # last column is the target
        self.variables_       = { v.Hash : v.Name for v in sorted(ds.Variables, key=lambda x: x.Index) if v.Hash != target.Hash }
        self.inputs_          = [ k for k in self.variables_ ]
        training_range        = op.Range(0, ds.Rows)
        test_range            = op.Range(ds.Rows-1, ds.Rows) # hackish, because it can't be empty
        problem               = op.Problem(ds, training_range, test_range)
        problem.Target        = target
        problem.InputHashes   = self.inputs_
        pcfg                  = self.__init_primitive_config(self.allowed_symbols)
        problem.ConfigurePrimitiveSet(pcfg)
        pset = problem.PrimitiveSet

        creator               = self.__init_creator(self.initialization_method, pset, self.inputs_)
        coeff_initializer     = op.UniformIntCoefficientAnalyzer() if self.symbolic_mode else op.NormalCoefficientInitializer()

        if self.symbolic_mode:
            self.optimizer_iterations = 0 # do not tune coefficients in symbolic mode
            coeff_initializer.ParameterizeDistribution(-5, +5)
        else:
            coeff_initializer.ParameterizeDistribution(0, 1)

        single_objective      = True if len(self.objectives) == 1 else False

        dtable = op.DispatchTable()

        # these lists are used as placeholders in order to extend the lifetimes of the objects
        evaluators = [] # placeholder for the evaluator(s)

        update_rule = self.__init_sgd_update_rule()
        optimizer = self.__init_optimizer(dtable, problem, self.optimizer, self.optimizer_likelihood, self.optimizer_iterations, self.optimizer_batch_size, update_rule)
        mdl_opt = self.__init_optimizer(dtable, problem, self.optimizer, self.optimizer_likelihood, 100, self.optimizer_batch_size, update_rule)

        # evaluators for minimum description length and information criteria
        mdl_eval = op.MinimumDescriptionLengthEvaluator(problem, dtable, 'gauss')
        mdl_eval.Sigma = self.uncertainty

        bic_eval = op.BayesianInformationCriterionEvaluator(problem, dtable)
        aik_eval = op.AkaikeInformationCriterionEvaluator(problem, dtable)

        for obj in self.objectives:
            eval_        = self.__init_evaluator(obj, problem, dtable)
            eval_.Budget = self.max_evaluations
            evaluators.append(eval_)

        if single_objective:
            evaluator = evaluators[0]
        else:
            evaluator = op.MultiEvaluator(problem)
            for eval_ in evaluators:
                evaluator.Add(eval_)
            evaluator.Budget = self.max_evaluations

        comparison            = op.SingleObjectiveComparison(0) if single_objective else op.CrowdedComparison()

        female_selector       = self.__init_selector(self.female_selector, comparison)
        male_selector         = self.__init_selector(self.male_selector, comparison)
        reinserter            = self.__init_reinserter(self.reinserter, comparison)
        cx                    = op.SubtreeCrossover(self.crossover_internal_probability, self.max_depth, self.max_length)

        mut                   = op.MultiMutation()
        mut_list = [] # this list is needed as a placeholder to prolong the lifetimes of the mutation operators (since the multi-mutation only stores references)
        for k in self.mutation:
            v = self.mutation[k]
            m = self.__init_mutation(k, self.inputs_, pset, creator, coeff_initializer)
            mut.Add(m, v)
            mut_list.append(m)

        coeff_optimizer       = op.CoefficientOptimizer(optimizer, self.lamarckian_probability)
        generator             = self.__init_generator(self.offspring_generator, evaluator, cx, mut, female_selector, male_selector, coeff_optimizer)

        min_arity, max_arity  = pset.FunctionArityLimits()
        tree_initializer      = op.UniformLengthTreeInitializer(creator)
        tree_initializer.ParameterizeDistribution(min_arity+1, min(self.initialization_max_length, self.max_length))
        tree_initializer.MinDepth = 1

        # btc and ptc2 do not need a depth restriction
        tree_initializer.MaxDepth = self.initialization_max_depth if self.initialization_method == 'koza' else 1000

        if isinstance(self.random_state, np.random.Generator):
            self.random_state = self.random_state.bit_generator.random_raw()

        config = op.GeneticAlgorithmConfig(
                     generations      = self.generations,
                     max_evaluations  = self.max_evaluations,
                     local_iterations = self.optimizer_iterations,
                     population_size  = self.population_size,
                     pool_size        = self.pool_size,
                     p_crossover      = self.crossover_probability,
                     p_mutation       = self.mutation_probability,
                     p_local          = self.local_search_probability,
                     p_lamarck        = self.lamarckian_probability,
                     epsilon          = self.epsilon,
                     seed             = self.random_state,
                     time_limit       = self.time_limit
                     )

        sorter = None if single_objective else op.RankSorter()
        gp     = op.GeneticProgrammingAlgorithm(problem, config, tree_initializer, coeff_initializer, generator, reinserter) if single_objective \
                 else op.NSGA2Algorithm(problem, config, tree_initializer, coeff_initializer, generator, reinserter, sorter)
        rng    = op.RomuTrio(np.uint64(config.Seed))

        gp.Run(rng, None, self.n_threads)

        def get_solution_stats(solution):
            """Takes a solution (operon individual) and computes a set of stats"""
            # perform linear scaling
            y_pred = op.Evaluate(dtable, solution.Genotype, ds, training_range)
            scale, offset = op.FitLeastSquares(y_pred, y)
            nodes = solution.Genotype.Nodes
            if scale != 1 and self.add_model_scale_term:
                nodes += [ op.Node.Constant(scale), op.Node.Mul() ]
            if offset != 0 and self.add_model_intercept_term:
                nodes += [ op.Node.Constant(offset), op.Node.Add() ]
            solution.Genotype = op.Tree(nodes).UpdateNodes()

            # get solution variables
            solution_vars = [self.variables_[x.HashValue] for x in solution.Genotype.Nodes if x.IsVariable]

            stats = {
                'model' : op.InfixFormatter.Format(solution.Genotype, self.variables_, 6),
                'variables' : set(solution_vars),
                'tree' : solution.Genotype,
                'objective_values' : evaluator(rng, solution),
                'mean_squared_error' : mean_squared_error(y, scale * y_pred + offset),
                'minimum_description_length' : mdl_eval(rng, solution)[0],
                'bayesian_information_criterion' : bic_eval(rng, solution)[0],
                'akaike_information_criterion' : aik_eval(rng, solution)[0],
            }

            return stats


        front = [gp.BestModel] if single_objective else gp.BestFront
        self.pareto_front_ = [get_solution_stats(m) for m in front]
        best = min(self.pareto_front_, key=lambda x: x[self.model_selection_criterion])
        self.model_ = best['tree']

        self.stats_ = {
            'model_length': self.model_.Length - 4, # do not count scaling nodes?
            'model_complexity': self.model_.Length - 4 + 2 * sum(1 for x in self.model_.Nodes if x.IsVariable),
            'generations': gp.Generation,
            'evaluation_count': evaluator.CallCount,
            'residual_evaluations': evaluator.ResidualEvaluations,
            'jacobian_evaluations': evaluator.JacobianEvaluations,
            'random_state': self.random_state
        }

        self.individuals_ = [x for x in gp.Individuals]

        self.is_fitted_ = True
        # `fit` should always return `self`
        return self


    def evaluate_model(self, model, X):
        X = check_array(X, accept_sparse=False)
        ds = op.Dataset(X)
        rg = op.Range(0, ds.Rows)
        return op.Evaluate(model, ds, rg).reshape(-1,)


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
        return self.evaluate_model(self.model_, X)
