// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "pyoperon/pyoperon.hpp"
#include <operon/operators/reinserter.hpp>

void InitReinserter(nb::module_ &m)
{
    // reinserter
    nb::class_<Operon::ReinserterBase> rb(m, "ReinserterBase");

    nb::class_<Operon::ReplaceWorstReinserter, Operon::ReinserterBase>(m, "ReplaceWorstReinserter")
        .def("__init__", [](Operon::ReplaceWorstReinserter* op, size_t i) {
            Operon::SingleObjectiveComparison comp(i);
            new (op) Operon::ReplaceWorstReinserter(comp); }, nb::arg("objective_index"))
        // these next two constructors are provided to substitute a python-allocated comparison object (SLOW to call)
        // to an identical one allocated on the C++ side (FAST)
        .def("__init__", [](Operon::ReplaceWorstReinserter* op, Operon::SingleObjectiveComparison const& comp) {
                Operon::SingleObjectiveComparison temp(comp.GetObjectiveIndex());
                new (op) Operon::ReplaceWorstReinserter(temp);
            })
        .def("__init__", [](Operon::ReplaceWorstReinserter* op, Operon::CrowdedComparison const&) {
                Operon::CrowdedComparison temp;
                new (op) Operon::ReplaceWorstReinserter(temp);
            })
        .def(nb::init<Operon::ComparisonCallback const&>())
        .def("__call__", &Operon::ReplaceWorstReinserter::operator());

    nb::class_<Operon::KeepBestReinserter, Operon::ReinserterBase>(m, "KeepBestReinserter")
        .def("__init__", [](Operon::KeepBestReinserter* op, size_t i) {
            Operon::SingleObjectiveComparison comp(i);
            new (op) Operon::KeepBestReinserter(comp); }, nb::arg("objective_index"))
        // these next two constructors are provided to substitute a python-allocated comparison object (SLOW to call)
        // to an identical one allocated on the C++ side (FAST)
        .def("__init__", [](Operon::KeepBestReinserter* op, Operon::SingleObjectiveComparison const& comp) {
                Operon::SingleObjectiveComparison temp(comp.GetObjectiveIndex());
                new (op) Operon::KeepBestReinserter(temp);
            })
        .def("__init__", [](Operon::KeepBestReinserter* op, Operon::CrowdedComparison const&) {
                Operon::CrowdedComparison temp;
                new (op) Operon::KeepBestReinserter(temp);
            })
        .def(nb::init<Operon::ComparisonCallback const&>())
        .def("__call__", &Operon::KeepBestReinserter::operator());
}
