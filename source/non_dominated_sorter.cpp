// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "pyoperon/pyoperon.hpp"

#include <operon/operators/non_dominated_sorter.hpp>

namespace py = pybind11;

void InitNondominatedSorter(py::module_ &m)
{
    py::class_<Operon::NondominatedSorterBase>(m, "NonDominatedSorterBase");

    py::class_<Operon::RankSorter, Operon::NondominatedSorterBase>(m, "RankSorter")
        .def(py::init<>())
        .def("Sort", &Operon::RankSorter::Sort);
}
