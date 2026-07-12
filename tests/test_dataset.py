# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2024 Heal Research

"""Tests for the pyoperon.Dataset bindings."""

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


@needs_extension
def test_variablenames_roundtrip_on_ndarray_dataset():
    # regression test for #57: constructing a Dataset directly from an
    # ndarray (rather than loading from a file) made VariableNames raise
    # TypeError, because the binding TU never included the nanobind
    # std::string caster.
    ds = op.Dataset(np.asfortranarray(np.random.rand(20, 3).astype(np.float32)))
    assert ds.VariableNames == ['X1', 'X2', 'X3']
    ds.VariableNames = ['a', 'b', 'c']
    assert ds.VariableNames == ['a', 'b', 'c']


@pytest.fixture
def _named_dataset():
    return op.Dataset(['X', 'Y', 'F'], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])


@needs_extension
class TestGetVariable:
    """GetVariable's C++ implementation returns tl::expected<Variable, DatasetError>,
    shimmed back to std::optional at the binding boundary; these lock down the
    Python-facing None-on-miss contract.
    """

    def test_known_name_returns_variable(self, _named_dataset):
        var = _named_dataset.GetVariable('F')
        assert var is not None
        assert var.Name == 'F'

    def test_unknown_name_returns_none(self, _named_dataset):
        assert _named_dataset.GetVariable('NOPE') is None

    def test_known_hash_returns_variable(self, _named_dataset):
        target = _named_dataset.GetVariable('F')
        assert _named_dataset.GetVariable(target.Hash).Name == 'F'

    def test_unknown_hash_returns_none(self, _named_dataset):
        assert _named_dataset.GetVariable(0) is None
