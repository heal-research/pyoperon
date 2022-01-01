// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <type_traits>

#include <operon/core/types.hpp>
#include <operon/core/dataset.hpp>
#include <operon/core/individual.hpp>

namespace py = pybind11;

// enable pass-by-reference semantics for this vector type
PYBIND11_MAKE_OPAQUE(std::vector<Operon::Variable>);
PYBIND11_MAKE_OPAQUE(std::vector<Operon::Individual>);

template<typename T>
auto MakeView(Operon::Span<T const> view) -> py::array_t<T const>
{
    auto sz = static_cast<pybind11::ssize_t>(view.size());
    py::array_t<T const> arr(sz, view.data(), py::capsule(view.data()));
    ENSURE(arr.owndata() == false);
    ENSURE(arr.data() == view.data());
    return arr;
}

template<typename T>
auto MakeSpan(py::array_t<T> arr) -> Operon::Span<T>
{
    py::buffer_info info = arr.request();
    return Operon::Span<T>(static_cast<T*>(info.ptr), static_cast<typename Operon::Span<T>::size_type>(info.size));
}

void InitAlgorithm(py::module_&);
void InitCreator(py::module_&);
void InitCrossover(py::module_&);
void InitDataset(py::module_&);
void InitEval(py::module_&);
void InitGenerator(py::module_&);
void InitInitializer(py::module_&);
void InitMutation(py::module_&);
void InitNondominatedSorter(py::module_&);
void InitNode(py::module_&);
void InitProblem(py::module_&);
void InitPset(py::module_&);
void InitReinserter(py::module_&m);
void InitSelector(py::module_&m);
void InitTree(py::module_&);
