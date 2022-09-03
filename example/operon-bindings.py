# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

import random, time, sys, os, json
import numpy as np
import pandas as pd
from scipy import stats

import pyoperon as Operon
from pmlb import fetch_data

# get some training data - see https://epistasislab.github.io/pmlb/
D = fetch_data('1027_ESL', return_X_y=False, local_cache_dir='./datasets').to_numpy()

# initialize a dataset from a numpy array
ds             = Operon.Dataset(D)

# define the training and test ranges
training_range = Operon.Range(0, ds.Rows // 2)
test_range     = Operon.Range(ds.Rows // 2, ds.Rows)

# define the regression target
target         = ds.Variables[-1] # take the last column in the dataset as the target

# take all other variables as inputs
inputs         = Operon.VariableCollection(v for v in ds.Variables if v.Name != target.Name)

# initialize a rng
rng            = Operon.RomuTrio(random.randint(1, 1000000))

# initialize a problem object which encapsulates the data, input, target and training/test ranges
problem        = Operon.Problem(ds, inputs, target.Name, training_range, test_range)

# initialize an algorithm configuration
config         = Operon.GeneticAlgorithmConfig(generations=1000, max_evaluations=1000000, local_iterations=0, population_size=1000, pool_size=1000, p_crossover=1.0, p_mutation=0.25, epsilon=1e-5, seed=1, time_limit=86400)

# use tournament selection with a group size of 5
# we are doing single-objective optimization so the objective index is 0
selector       = Operon.TournamentSelector(objective_index=0)
selector.TournamentSize = 5

# initialize the primitive set (add, sub, mul, div, exp, log, sin, cos), constants and variables are implicitly added
pset           = Operon.PrimitiveSet()
pset.SetConfig(Operon.PrimitiveSet.Arithmetic | Operon.NodeType.Exp | Operon.NodeType.Log | Operon.NodeType.Sin | Operon.NodeType.Cos)

# define tree length and depth limits
minL, maxL     = 1, 50
maxD           = 10

# define a tree creator (responsible for producing trees of given lengths)
btc            = Operon.BalancedTreeCreator(pset, inputs, bias=0.0)
tree_initializer = Operon.UniformLengthTreeInitializer(btc)
tree_initializer.ParameterizeDistribution(minL, maxL)
tree_initializer.MaxDepth = maxD

# define a coefficient initializer (this will initialize the coefficients in the tree)
coeff_initializer = Operon.NormalCoefficientInitializer()
coeff_initializer.ParameterizeDistribution(0, 1)

# define several kinds of mutation
mut_onepoint   = Operon.NormalOnePointMutation()
mut_changeVar  = Operon.ChangeVariableMutation(inputs)
mut_changeFunc = Operon.ChangeFunctionMutation(pset)
mut_replace    = Operon.ReplaceSubtreeMutation(btc, coeff_initializer, maxD, maxL)

# use a multi-mutation operator to apply them at random
mutation       = Operon.MultiMutation()
mutation.Add(mut_onepoint, 1)
mutation.Add(mut_changeVar, 1)
mutation.Add(mut_changeFunc, 1)
mutation.Add(mut_replace, 1)

# define crossover
crossover_internal_probability = 0.9 # probability to pick an internal node as a cut point
crossover      = Operon.SubtreeCrossover(crossover_internal_probability, maxD, maxL)

# define fitness evaluation
interpreter    = Operon.Interpreter() # tree interpreter
error_metric   = Operon.R2()          # use the coefficient of determination as fitness
evaluator      = Operon.Evaluator(problem, interpreter, error_metric, True) # initialize evaluator, use linear scaling = True
evaluator.Budget = 1000 * 1000             # computational budget
evaluator.LocalOptimizationIterations = 0  # number of local optimization iterations (coefficient tuning using gradient descent)

# define how new offspring are created
generator      = Operon.BasicOffspringGenerator(evaluator, crossover, mutation, selector, selector)

# define how the offspring are merged back into the population - here we replace the worst parents with the best offspring
reinserter     = Operon.ReplaceWorstReinserter(objective_index=0)
gp             = Operon.GeneticProgrammingAlgorithm(problem, config, tree_initializer, coeff_initializer, generator, reinserter)

# report some progress
gen = 0
max_ticks = 50
interval = 1 if config.Generations < max_ticks else int(np.round(config.Generations / max_ticks, 0))
t0 = time.time()

def report():
    global gen
    best = gp.BestModel
    bestfit = best.GetFitness(0)
    sys.stdout.write('\r')
    cursor = int(np.round(gen / config.Generations * max_ticks))
    for i in range(cursor):
        sys.stdout.write('\u2588')
    sys.stdout.write(' ' * (max_ticks-cursor))
    sys.stdout.write(f'{100 * gen/config.Generations:.1f}%, generation {gen}/{config.Generations}, train quality: {-bestfit:.6f}, elapsed: {time.time()-t0:.2f}s')
    sys.stdout.flush()
    gen += 1

# run the algorithm
gp.Run(rng, report, threads=16)

# get the best solution and print it
best = gp.BestModel
model_string = Operon.InfixFormatter.Format(best.Genotype, ds, 6)
print(f'\n{model_string}')
