// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "pyoperon/pyoperon.hpp"

#include <operon/operators/crossover.hpp>

void InitCrossover(py::module_ &m)
{
    // crossover
    py::class_<Operon::CrossoverBase> crossoverBase(m, "CrossoverBase");

    py::class_<Operon::SubtreeCrossover, Operon::CrossoverBase>(m, "SubtreeCrossover")
        .def(py::init<double, size_t, size_t>(),
                py::arg("internal_probability"),
                py::arg("depth_limit"),
                py::arg("length_limit"))
        .def("__call__", &Operon::SubtreeCrossover::operator());
}
