// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "pyoperon/pyoperon.hpp"
#include <operon/algorithms/gp.hpp>
#include <operon/algorithms/nsga2.hpp>
#include <operon/operators/initializer.hpp>
#include <operon/operators/non_dominated_sorter.hpp>
#include <operon/operators/reinserter.hpp>

#include <nanobind/stl/function.h>

using namespace nb::literals;

void InitAlgorithm(nb::module_ &m)
{
    // Bound via plain lambdas (not static_cast<...>(&Class::Method) /
    // nb::overload_cast<>) so these keep working regardless of whether the
    // underlying getter is a const/non-const overload pair or a single
    // C++23 deducing-this member - a lambda just calls the member on a
    // concrete object and lets overload resolution (or deducing-this) do
    // the rest, rather than needing a specific overloaded symbol's address.
    nb::class_<Operon::GeneticAlgorithmBase>(m, "GeneticAlgorithmBase")
        .def_prop_ro("Generation", [](Operon::GeneticAlgorithmBase const& self) { return self.Generation(); }, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("Individuals", [](Operon::GeneticAlgorithmBase const& self) -> std::vector<Operon::Individual> const& { return self.Individuals(); }, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("Parents", [](Operon::GeneticAlgorithmBase const& self) { return self.Parents(); }, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("Offspring", [](Operon::GeneticAlgorithmBase const& self) { return self.Offspring(); }, nb::call_guard<nb::gil_scoped_release>())
        ;

    nb::class_<Operon::GeneticProgrammingAlgorithm, Operon::GeneticAlgorithmBase>(m, "GeneticProgrammingAlgorithm")
        .def(nb::init<Operon::GeneticAlgorithmConfig, Operon::Problem const*, Operon::TreeInitializerBase const*, Operon::CoefficientInitializerBase const*, Operon::OffspringGeneratorBase const*, Operon::ReinserterBase const*>(),
                nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>(), nb::keep_alive<1, 5>(), nb::keep_alive<1, 6>(), nb::keep_alive<1, 7>())
        .def("Run", nb::overload_cast<Operon::RandomGenerator&, Operon::ReportCallback, size_t, bool>(&Operon::GeneticProgrammingAlgorithm::Run),
                nb::call_guard<nb::gil_scoped_release>(), nb::arg("rng"), nb::arg("callback") = nullptr, nb::arg("threads") = 0, nb::arg("warm_start") = false)
        .def("Reset", &Operon::GeneticProgrammingAlgorithm::Reset, nb::call_guard<nb::gil_scoped_release>())
        .def("RestoreIndividuals", &Operon::GeneticProgrammingAlgorithm::RestoreIndividuals, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_rw("IsFitted",
                [](Operon::GeneticProgrammingAlgorithm const& self) { return self.IsFitted(); },
                [](Operon::GeneticProgrammingAlgorithm& self, bool value) { self.IsFitted() = value; },
                nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("BestModel", [](Operon::GeneticProgrammingAlgorithm const& self) {
                auto minElem = std::min_element(self.Parents().begin(), self.Parents().end(), [&](auto const& a, auto const& b) { return a[0] < b[0]; });
                return *minElem;
            }, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("Config", &Operon::GeneticProgrammingAlgorithm::GetConfig, nb::call_guard<nb::gil_scoped_release>());

    nb::class_<Operon::NSGA2, Operon::GeneticAlgorithmBase>(m, "NSGA2Algorithm")
        .def(nb::init<Operon::GeneticAlgorithmConfig, Operon::Problem const*, Operon::TreeInitializerBase const*, Operon::CoefficientInitializerBase const*, Operon::OffspringGeneratorBase const*, Operon::ReinserterBase const*, Operon::NondominatedSorterBase const*>(),
                nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>(), nb::keep_alive<1, 5>(), nb::keep_alive<1, 6>(), nb::keep_alive<1, 7>(), nb::keep_alive<1, 8>())
        .def("Run", nb::overload_cast<Operon::RandomGenerator&, Operon::ReportCallback, size_t, bool>(&Operon::NSGA2::Run),
                nb::call_guard<nb::gil_scoped_release>(), nb::arg("rng"), "callback"_a = nb::none(), "threads"_a = 0, "warm_start"_a = false)
        .def("Reset", &Operon::NSGA2::Reset)
        .def("RestoreIndividuals", &Operon::NSGA2::RestoreIndividuals, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_rw("IsFitted",
                [](Operon::NSGA2 const& self) { return self.IsFitted(); },
                [](Operon::NSGA2& self, bool value) { self.IsFitted() = value; },
                nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("BestModel", [](Operon::NSGA2 const& self) {
                auto minElem = std::min_element(self.Best().begin(), self.Best().end(), [&](auto const& a, auto const& b) { return a[0] < b[0];});
                return *minElem;
            }, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("BestFront", [](Operon::NSGA2 const& self) {
                    auto best = self.Best();
                    return std::vector<Operon::Individual>(best.begin(), best.end());
                }, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("Config", &Operon::NSGA2::GetConfig, nb::call_guard<nb::gil_scoped_release>());
}
