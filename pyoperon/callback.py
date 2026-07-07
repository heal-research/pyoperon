# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2026 Heal Research

"""Training callbacks for :class:`pyoperon.sklearn.SymbolicRegressor`.

Modeled on the callback APIs of Keras/Lightning/pymoo (see
https://github.com/heal-research/pyoperon/issues/18): subclass
:class:`Callback` and override any of its hooks. ``on_generation_end`` may
return ``True`` to request early termination of the run.

Operon's C++ layer only exposes a single per-generation hook
(``ReportCallback``), so ``on_fit_begin``/``on_fit_end`` are orchestrated
entirely on the Python side by ``SymbolicRegressor.fit()`` (called just
before/after ``Run()``), while ``on_generation_end`` maps directly onto the
C++ callback.
"""

from __future__ import annotations

from typing import Any


class Callback:
    """Base class for ``SymbolicRegressor`` training callbacks.

    ``model`` (passed to every hook) is the underlying
    ``GeneticProgrammingAlgorithm``/``NSGA2Algorithm`` instance, exposing
    e.g. ``Generation``, ``BestModel``, ``Individuals``, ``Config`` for
    introspection - the same role Keras callbacks' ``model`` argument plays.
    """

    def on_fit_begin(self, model: Any) -> None:
        pass

    def on_generation_end(self, model: Any) -> bool | None:
        pass

    def on_fit_end(self, model: Any) -> None:
        pass


class CallbackList(Callback):
    """Fans a hook call out to a list of callbacks.

    ``on_generation_end`` ORs early-stop requests together - matching
    operon's ``ReportCallback`` semantics, where any ``True`` return stops
    the run - and always calls every callback (no short-circuiting), so
    that later callbacks still observe each generation.
    """

    def __init__(self, callbacks: list[Callback] | None = None):
        self.callbacks = list(callbacks) if callbacks else []

    def on_fit_begin(self, model: Any) -> None:
        for cb in self.callbacks:
            cb.on_fit_begin(model)

    def on_generation_end(self, model: Any) -> bool:
        stop = False
        for cb in self.callbacks:
            if cb.on_generation_end(model):
                stop = True
        return stop

    def on_fit_end(self, model: Any) -> None:
        for cb in self.callbacks:
            cb.on_fit_end(model)


class EarlyStopping(Callback):
    """Stop once the best fitness hasn't improved for `patience` generations.

    Fitness follows operon's convention (smaller is better - the same one
    `SymbolicRegressor`/the low-level bindings use to pick `BestModel` via
    `min`), so "improved" means the monitored value decreased by more than
    `min_delta`.
    """

    def __init__(
        self,
        min_delta: float = 0.0,
        patience: int = 10,
        objective_index: int = 0,
    ):
        self.min_delta = min_delta
        self.patience = patience
        self.objective_index = objective_index
        self._best: float | None = None
        self._wait = 0

    def on_fit_begin(self, model: Any) -> None:
        # Reset here rather than in __init__: sklearn's clone()/
        # cross_val_score reuse the same callback instance across folds
        # without deep-copying it, so per-fit state must reset per fit(),
        # not per construction.
        self._best = None
        self._wait = 0

    def on_generation_end(self, model: Any) -> bool:
        fitness = model.BestModel.GetFitness(self.objective_index)
        if self._best is None or fitness < self._best - self.min_delta:
            self._best = fitness
            self._wait = 0
        else:
            self._wait += 1
        return self._wait >= self.patience
