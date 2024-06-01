// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "pyoperon/pyoperon.hpp"
#include <operon/operators/selector.hpp>

void InitSelector(nb::module_ &m)
{
    // selection
    nb::class_<Operon::SelectorBase> sel(m, "SelectorBase");

    nb::class_<Operon::TournamentSelector, Operon::SelectorBase>(m, "TournamentSelector")
        .def("__init__", [](Operon::TournamentSelector* op, size_t i){
            Operon::SingleObjectiveComparison comp(i);
            new (op) Operon::TournamentSelector(comp); }, nb::arg("objective_index"))
        .def("__init__", [](Operon::TournamentSelector* op, Operon::SingleObjectiveComparison const& comp) {
                Operon::SingleObjectiveComparison temp(comp.GetObjectiveIndex());
                new (op) Operon::TournamentSelector(temp);
            })
        .def("__init__", [](Operon::TournamentSelector* op, Operon::CrowdedComparison const&) {
                Operon::CrowdedComparison temp;
                new (op) Operon::TournamentSelector(temp);
            })
        .def(nb::init<Operon::ComparisonCallback const&>())
        .def("__call__", &Operon::TournamentSelector::operator())
        .def_prop_rw("TournamentSize", &Operon::TournamentSelector::GetTournamentSize, &Operon::TournamentSelector::SetTournamentSize);

    nb::class_<Operon::RankTournamentSelector, Operon::SelectorBase>(m, "RankTournamentSelector")
        .def("__init__", [](Operon::RankTournamentSelector* op, size_t i){
            Operon::SingleObjectiveComparison comp(i);
            new (op) Operon::RankTournamentSelector(comp); }, nb::arg("objective_index"))
        .def("__init__", [](Operon::RankTournamentSelector* op, Operon::SingleObjectiveComparison const& comp) {
                Operon::SingleObjectiveComparison temp(comp.GetObjectiveIndex());
                new (op) Operon::RankTournamentSelector(temp);
            })
        .def("__init__", [](Operon::RankTournamentSelector* op, Operon::CrowdedComparison const&) {
                Operon::CrowdedComparison temp;
                new (op) Operon::RankTournamentSelector(temp);
            })
        .def(nb::init<Operon::ComparisonCallback const&>())
        .def("__call__", &Operon::RankTournamentSelector::operator())
        .def("Prepare", &Operon::RankTournamentSelector::Prepare)
        .def_prop_rw("TournamentSize", &Operon::RankTournamentSelector::GetTournamentSize, &Operon::RankTournamentSelector::SetTournamentSize);

    nb::class_<Operon::ProportionalSelector, Operon::SelectorBase>(m, "ProportionalSelector")
        .def("__init__", [](Operon::ProportionalSelector* op, size_t i){
                Operon::SingleObjectiveComparison comp(i);
                new (op) Operon::ProportionalSelector(comp); }, nb::arg("objective_index"))
        .def("__init__", [](Operon::ProportionalSelector* op, Operon::SingleObjectiveComparison const& comp) {
                Operon::SingleObjectiveComparison temp(comp.GetObjectiveIndex());
                new (op) Operon::ProportionalSelector(temp);
            })
        .def("__init__", [](Operon::ProportionalSelector* op, Operon::CrowdedComparison const&) {
                Operon::CrowdedComparison temp;
                new (op) Operon::ProportionalSelector(temp);
            })
        .def(nb::init<Operon::ComparisonCallback const&>())
        .def("__call__", &Operon::ProportionalSelector::operator())
        .def("Prepare", nb::overload_cast<const Operon::Span<const Operon::Individual>>(&Operon::ProportionalSelector::Prepare, nb::const_))
        .def("SetObjIndex", &Operon::ProportionalSelector::SetObjIndex);

    nb::class_<Operon::RandomSelector, Operon::SelectorBase>(m, "RandomSelector")
        .def(nb::init<>())
        .def("__call__", &Operon::RandomSelector::operator())
        .def("Prepare", nb::overload_cast<const Operon::Span<const Operon::Individual>>(&Operon::RandomSelector::Prepare, nb::const_));

}
