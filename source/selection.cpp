// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "pyoperon/pyoperon.hpp"

void init_selection(py::module_ &m)
{
    // selection
    py::class_<Operon::SelectorBase> sel(m, "SelectorBase");

    py::class_<Operon::TournamentSelector, Operon::SelectorBase>(m, "TournamentSelector")
        .def(py::init([](size_t i){ 
            Operon::SingleObjectiveComparison comp(i);
            return Operon::TournamentSelector(comp); }), py::arg("objective_index"))
        .def(py::init([](Operon::SingleObjectiveComparison const& comp) {
                Operon::SingleObjectiveComparison temp(comp.GetObjectiveIndex());
                return Operon::TournamentSelector(temp);
            }))
        .def(py::init([](Operon::CrowdedComparison const&) {
                Operon::CrowdedComparison temp;
                return Operon::TournamentSelector(temp);
            }))
        .def(py::init<Operon::ComparisonCallback const&>())
        .def("__call__", &Operon::TournamentSelector::operator())
        .def_property("TournamentSize", &Operon::TournamentSelector::GetTournamentSize, &Operon::TournamentSelector::SetTournamentSize);

    py::class_<Operon::RankTournamentSelector, Operon::SelectorBase>(m, "RankTournamentSelector")
        .def(py::init([](size_t i){
            Operon::SingleObjectiveComparison comp(i);
            return Operon::RankTournamentSelector(comp); }), py::arg("objective_index"))
        .def(py::init([](Operon::SingleObjectiveComparison const& comp) {
                Operon::SingleObjectiveComparison temp(comp.GetObjectiveIndex());
                return Operon::RankTournamentSelector(temp);
            }))
        .def(py::init([](Operon::CrowdedComparison const&) {
                Operon::CrowdedComparison temp;
                return Operon::RankTournamentSelector(temp);
            }))
        .def(py::init<Operon::ComparisonCallback const&>())
        .def("__call__", &Operon::RankTournamentSelector::operator())
        .def("Prepare", &Operon::RankTournamentSelector::Prepare)
        .def_property("TournamentSize", &Operon::RankTournamentSelector::GetTournamentSize, &Operon::RankTournamentSelector::SetTournamentSize);

    py::class_<Operon::ProportionalSelector, Operon::SelectorBase>(m, "ProportionalSelector")
        .def(py::init([](size_t i){
                Operon::SingleObjectiveComparison comp(i);
                return Operon::ProportionalSelector(comp); }), py::arg("objective_index"))
        .def(py::init([](Operon::SingleObjectiveComparison const& comp) {
                Operon::SingleObjectiveComparison temp(comp.GetObjectiveIndex());
                return Operon::ProportionalSelector(temp);
            }))
        .def(py::init([](Operon::CrowdedComparison const&) {
                Operon::CrowdedComparison temp;
                return Operon::ProportionalSelector(temp);
            }))
        .def(py::init<Operon::ComparisonCallback const&>())
        .def("__call__", &Operon::ProportionalSelector::operator())
        .def("Prepare", py::overload_cast<const Operon::Span<const Operon::Individual>>(&Operon::ProportionalSelector::Prepare, py::const_))
        .def("SetObjIndex", &Operon::ProportionalSelector::SetObjIndex);

    py::class_<Operon::RandomSelector, Operon::SelectorBase>(m, "RandomSelector")
        .def(py::init<>())
        .def("__call__", &Operon::RandomSelector::operator())
        .def("Prepare", py::overload_cast<const Operon::Span<const Operon::Individual>>(&Operon::RandomSelector::Prepare, py::const_));

}
