// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "pyoperon/pyoperon.hpp"
#include <operon/core/range.hpp>
#include <operon/core/problem.hpp>

void InitProblem(py::module_ &m)
{
    // problem
    py::class_<Operon::Problem>(m, "Problem")
        .def(py::init<Operon::Dataset, Operon::Range, Operon::Range>())
        .def("ConfigurePrimitiveSet", &Operon::Problem::ConfigurePrimitiveSet)
        .def_property("Target",
            &Operon::Problem::TargetVariable,
            [](Operon::Problem& self, Operon::Variable const& var) { self.SetTarget(var.Hash); }
        )
        .def_property("InputHashes",
                &Operon::Problem::GetInputs, // getter
                &Operon::Problem::SetInputs<std::vector<Operon::Hash> const&> // setter
        )
        .def_property("PrimitiveSet",
            [](Operon::Problem const& self) { return self.GetPrimitiveSet(); },
            [](Operon::Problem& self, Operon::PrimitiveSet pset) { self.GetPrimitiveSet() = pset; }
        )
        .def_property_readonly("PrimitiveSet", [](Operon::Problem& self) { return self.GetPrimitiveSet(); });
}
