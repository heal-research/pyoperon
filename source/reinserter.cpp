// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "pyoperon/pyoperon.hpp"
#include <operon/operators/reinserter.hpp>

void InitReinserter(py::module_ &m)
{
    // reinserter
    py::class_<Operon::ReinserterBase> rb(m, "ReinserterBase");

    py::class_<Operon::ReplaceWorstReinserter, Operon::ReinserterBase>(m, "ReplaceWorstReinserter")
        .def(py::init([](size_t i) { 
            Operon::SingleObjectiveComparison comp(i);
            return Operon::ReplaceWorstReinserter(comp); }), py::arg("objective_index"))
        // these next two constructors are provided to substitute a python-allocated comparison object (SLOW to call)
        // to an identical one allocated on the C++ side (FAST)
        .def(py::init([](Operon::SingleObjectiveComparison const& comp) {
                Operon::SingleObjectiveComparison temp(comp.GetObjectiveIndex());
                return Operon::ReplaceWorstReinserter(temp);
            }))
        .def(py::init([](Operon::CrowdedComparison const&) {
                Operon::CrowdedComparison temp;
                return Operon::ReplaceWorstReinserter(temp);
            }))
        .def(py::init<Operon::ComparisonCallback const&>())
        .def("__call__", &Operon::ReplaceWorstReinserter::operator());

    py::class_<Operon::KeepBestReinserter, Operon::ReinserterBase>(m, "KeepBestReinserter")
        .def(py::init([](size_t i) {
            Operon::SingleObjectiveComparison comp(i);
            return Operon::KeepBestReinserter(comp); }), py::arg("objective_index"))
        // these next two constructors are provided to substitute a python-allocated comparison object (SLOW to call)
        // to an identical one allocated on the C++ side (FAST)
        .def(py::init([](Operon::SingleObjectiveComparison const& comp) {
                Operon::SingleObjectiveComparison temp(comp.GetObjectiveIndex());
                return Operon::KeepBestReinserter(temp);
            }))
        .def(py::init([](Operon::CrowdedComparison const&) {
                Operon::CrowdedComparison temp;
                return Operon::KeepBestReinserter(temp);
            }))
        .def(py::init<Operon::ComparisonCallback const&>())
        .def("__call__", &Operon::KeepBestReinserter::operator());
}
