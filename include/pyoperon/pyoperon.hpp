// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include <nanobind/ndarray.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/function.h>

#include <type_traits>

#include <operon/core/types.hpp>
#include <operon/core/dataset.hpp>
#include <operon/core/individual.hpp>
#include <operon/operators/evaluator.hpp>

#include <fmt/ranges.h>

namespace nb = nanobind;

// enable pass-by-reference semantics for this vector type
NB_MAKE_OPAQUE(std::vector<Operon::Variable>);
NB_MAKE_OPAQUE(std::vector<Operon::Individual>);

template<typename T>
auto MakeView(Operon::Span<T const> view)
{
    return nb::ndarray<T const, nb::numpy, nb::shape<-1>, nb::f_contig>(
        /* data = */ view.data(),
        /* ndim = */ {view.size()},
        /* owner = */ nb::handle() // null owner
    );
}

template<typename T>
auto MakeSpan(nanobind::ndarray<T> arr) -> Operon::Span<T>
{
    return Operon::Span<T>{ arr.data(), arr.size() };
}

void InitAlgorithm(nb::module_&);
void InitBenchmark(nb::module_&);
void InitCreator(nb::module_&);
void InitCrossover(nb::module_&);
void InitDataset(nb::module_&);
void InitEval(nb::module_&);
void InitGenerator(nb::module_&);
void InitInitializer(nb::module_&);
void InitMutation(nb::module_&);
void InitNode(nb::module_&);
void InitNondominatedSorter(nb::module_&);
void InitOptimizer(nb::module_&);
void InitProblem(nb::module_&);
void InitPset(nb::module_&);
void InitReinserter(nb::module_&m);
void InitSelector(nb::module_&m);
void InitTree(nb::module_&);
