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
        .def("Prepare", [](Operon::ProportionalSelector const& self, std::vector<Operon::Individual> const& individuals) {
            self.Prepare(Operon::Span<const Operon::Individual>(individuals.data(), individuals.size()));
        }, nb::arg("individuals").noconvert(), nb::keep_alive<1, 2>())
        .def("SetObjIndex", &Operon::ProportionalSelector::SetObjIndex);

    nb::class_<Operon::RandomSelector, Operon::SelectorBase>(m, "RandomSelector")
        .def(nb::init<>())
        .def("__call__", &Operon::RandomSelector::operator())
        .def("Prepare", [](Operon::RandomSelector const& self, std::vector<Operon::Individual> const& individuals) {
            self.Prepare(Operon::Span<const Operon::Individual>(individuals.data(), individuals.size()));
        }, nb::arg("individuals").noconvert(), nb::keep_alive<1, 2>());

}
