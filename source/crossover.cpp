// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "pyoperon/pyoperon.hpp"

#include <operon/operators/crossover.hpp>

void InitCrossover(nb::module_ &m)
{
    // crossover
    nb::class_<Operon::CrossoverBase> crossoverBase(m, "CrossoverBase");

    nb::class_<Operon::SubtreeCrossover, Operon::CrossoverBase>(m, "SubtreeCrossover")
        .def(nb::init<double, size_t, size_t>(),
                nb::arg("internal_probability"),
                nb::arg("depth_limit"),
                nb::arg("length_limit"))
        .def("__call__", &Operon::SubtreeCrossover::operator());
}
