// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "pyoperon/pyoperon.hpp"
#include <operon/core/range.hpp>
#include <operon/core/problem.hpp>

void InitProblem(py::module_ &m)
{
    // problem
    py::class_<Operon::Problem>(m, "Problem")
        .def(py::init([](Operon::Dataset const& ds, std::vector<Operon::Variable> const& variables, Operon::Variable target,
                        Operon::Range trainingRange, Operon::Range testRange) {
            Operon::Span<const Operon::Variable> vars(variables.data(), variables.size());
            return Operon::Problem(ds, vars, target, trainingRange, testRange);
        }))
        .def_property_readonly("PrimitiveSet", [](Operon::Problem& self) { return self.GetPrimitiveSet(); });
}
