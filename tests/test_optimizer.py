# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2026 Heal Research

"""Tests for CoefficientOptimizer/OptimizerSummary bindings.

operon's OptimizerBase::Optimize/CoefficientOptimizer::operator() return
tl::expected<FitResult, FitFailure> (operon PR #122); source/optimizer.cpp
shims that back into the plain-struct OptimizerSummary Python API these
tests lock down. Exercises both the success path (fit improves) and the
failure path (max_iter=0, mirroring GrammarEnumerationAlgorithm's own
precondition on this exact contract in the C++ test suite).
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import pyoperon as op
    HAS_EXTENSION = True
except ImportError:
    HAS_EXTENSION = False

needs_extension = pytest.mark.skipif(
    not HAS_EXTENSION, reason='pyoperon C++ extension not available'
)


@pytest.fixture
def linear_fixture():
    """y = 2 * x; a single-variable tree initialized with the wrong
    coefficient (1.0), so a coefficient fit has real room to improve."""
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

    dtable = op.DispatchTable()
    rng = op.RandomGenerator(np.uint64(0))
    return problem, dtable, rng, tree


@needs_extension
class TestCoefficientOptimizer:
    def test_success_path_improves_and_applies_coefficients(self, linear_fixture):
        problem, dtable, rng, tree = linear_fixture
        optimizer = op.LMOptimizer(dtable, problem, max_iter=10)
        coeff_opt = op.CoefficientOptimizer(optimizer)

        optimized_tree, summary = coeff_opt(rng, tree)

        assert summary.Success is True
        assert summary.FinalParameters == pytest.approx([2.0], abs=1e-3)
        # applied back into the returned tree, not just reported in summary
        assert optimized_tree.GetCoefficients() == pytest.approx([2.0], abs=1e-3)

    def test_failure_path_leaves_tree_untouched(self, linear_fixture):
        problem, dtable, rng, tree = linear_fixture
        # max_iter=0 makes CoefficientOptimizer return early without ever
        # calling Optimize() - the same "Iterations() == 0" contract
        # GrammarEnumerationAlgorithm::Run's C++ test depends on.
        optimizer = op.LMOptimizer(dtable, problem, max_iter=0)
        coeff_opt = op.CoefficientOptimizer(optimizer)

        optimized_tree, summary = coeff_opt(rng, tree)

        assert summary.Success is False
        # diagnostics still populated (empty/zero defaults), not raising -
        # this is exactly the case the FitOutcome shim has to get right
        assert summary.FinalParameters == []
        assert summary.InitialCost == 0.0
        assert summary.Iterations == 0
        # never applied since the fit never ran
        assert optimized_tree.GetCoefficients() == pytest.approx([1.0])
