// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "pyoperon/pyoperon.hpp"
#include <operon/core/range.hpp>
#include <operon/core/problem.hpp>

void InitProblem(nb::module_ &m)
{
    // problem
    nb::class_<Operon::Problem>(m, "Problem")
        .def("__init__", [](Operon::Problem* problem, Operon::Dataset* dataset) {
            new (problem) Operon::Problem(gsl::not_null<Operon::Dataset*>(dataset));
        })
        .def("ConfigurePrimitiveSet", &Operon::Problem::ConfigurePrimitiveSet)
        .def("ConfigurePrimitiveSet", [](Operon::Problem& self, uint32_t config){
            self.ConfigurePrimitiveSet(static_cast<Operon::NodeType>(config));
        })
        .def_prop_rw("TrainingRange", &Operon::Problem::TrainingRange, [](Operon::Problem& self, Operon::Range range) {
            self.SetTrainingRange(range);
        })
        .def_prop_rw("TestRange", &Operon::Problem::TestRange, [](Operon::Problem& self, Operon::Range range) {
            self.SetTestRange(range);
        })
        .def_prop_rw("Target",
            &Operon::Problem::TargetVariable,
            [](Operon::Problem& self, Operon::Variable const& var) { self.SetTarget(var.Hash); }
        )
        .def_prop_rw("InputHashes",
            &Operon::Problem::GetInputs, // getter
            &Operon::Problem::SetInputs<std::vector<Operon::Hash> const&> // setter
        )
        .def_prop_rw("PrimitiveSet",
            [](Operon::Problem const& self) { return self.GetPrimitiveSet(); },
            [](Operon::Problem& self, Operon::PrimitiveSet pset) { self.GetPrimitiveSet() = pset; }
        )
        .def_prop_ro("PrimitiveSet", [](Operon::Problem& self) { return self.GetPrimitiveSet(); });
}
