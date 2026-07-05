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
