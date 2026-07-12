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
    // GeneticAlgorithmBase's getters are C++23 deducing-this members
    // (`auto Generation(this Self&& self)` etc.), not const/non-const
    // overload pairs, so there's no single member-function-pointer type to
    // take the address of for nb::overload_cast/static_cast the way the old
    // getter-pair style allowed. Lambdas calling through the const-qualified
    // `self` sidestep that - they just invoke whichever deducing-this
    // overload the compiler picks for a `const&`.
    nb::class_<Operon::GeneticAlgorithmBase>(m, "GeneticAlgorithmBase")
        .def_prop_ro("Generation", [](Operon::GeneticAlgorithmBase const& self) { return self.Generation(); }, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("Individuals", [](Operon::GeneticAlgorithmBase const& self) -> auto const& { return self.Individuals(); }, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("Parents", [](Operon::GeneticAlgorithmBase const& self) { return self.Parents(); }, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("Offspring", [](Operon::GeneticAlgorithmBase const& self) { return self.Offspring(); }, nb::call_guard<nb::gil_scoped_release>())
        ;

    nb::class_<Operon::GeneticProgrammingAlgorithm, Operon::GeneticAlgorithmBase>(m, "GeneticProgrammingAlgorithm")
        .def(nb::init<Operon::GeneticAlgorithmConfig, Operon::Problem const*, Operon::TreeInitializerBase const*, Operon::CoefficientInitializerBase const*, Operon::OffspringGeneratorBase const*, Operon::ReinserterBase const*>(),
                nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>(), nb::keep_alive<1, 5>(), nb::keep_alive<1, 6>(), nb::keep_alive<1, 7>())
        // Operon::ReportCallback is std::move_only_function, which nanobind
        // has no built-in caster for (only nanobind/stl/function.h, i.e.
        // std::function). This lambda takes the std::function nanobind can
        // bind from a Python callable and converts it to ReportCallback at
        // the call site instead.
        .def("Run", [](Operon::GeneticProgrammingAlgorithm& self, Operon::RandomGenerator& rng, std::function<bool()> callback, size_t threads, bool warmStart) {
                self.Run(rng, Operon::ReportCallback(std::move(callback)), threads, warmStart);
            }, nb::call_guard<nb::gil_scoped_release>(), nb::arg("rng"), nb::arg("callback") = nullptr, nb::arg("threads") = 0, nb::arg("warm_start") = false)
        .def("Reset", &Operon::GeneticProgrammingAlgorithm::Reset, nb::call_guard<nb::gil_scoped_release>())
        .def("RestoreIndividuals", &Operon::GeneticProgrammingAlgorithm::RestoreIndividuals, nb::call_guard<nb::gil_scoped_release>())
        // See GeneticAlgorithmBase's getters above for why this is a lambda
        // rather than nb::overload_cast against a deducing-this member.
        .def_prop_rw("IsFitted",[](Operon::GeneticProgrammingAlgorithm const& self) { return self.IsFitted(); },[](Operon::GeneticProgrammingAlgorithm& self, bool value) {
                self.IsFitted() = value;
        }, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("BestModel", [](Operon::GeneticProgrammingAlgorithm const& self) {
                auto minElem = std::min_element(self.Parents().begin(), self.Parents().end(), [&](auto const& a, auto const& b) { return a[0] < b[0]; });
                return *minElem;
            }, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("Config", &Operon::GeneticProgrammingAlgorithm::GetConfig, nb::call_guard<nb::gil_scoped_release>());

    nb::class_<Operon::NSGA2, Operon::GeneticAlgorithmBase>(m, "NSGA2Algorithm")
        .def(nb::init<Operon::GeneticAlgorithmConfig, Operon::Problem const*, Operon::TreeInitializerBase const*, Operon::CoefficientInitializerBase const*, Operon::OffspringGeneratorBase const*, Operon::ReinserterBase const*, Operon::NondominatedSorterBase const*>(),
                nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>(), nb::keep_alive<1, 5>(), nb::keep_alive<1, 6>(), nb::keep_alive<1, 7>(), nb::keep_alive<1, 8>())
        // See GeneticProgrammingAlgorithm's Run binding above for why this is
        // a std::function-taking lambda rather than nb::overload_cast directly.
        .def("Run", [](Operon::NSGA2& self, Operon::RandomGenerator& rng, std::function<bool()> callback, size_t threads, bool warmStart) {
                self.Run(rng, Operon::ReportCallback(std::move(callback)), threads, warmStart);
            }, nb::call_guard<nb::gil_scoped_release>(), nb::arg("rng"), "callback"_a = nb::none(), "threads"_a = 0, "warm_start"_a = false)
        .def("Reset", &Operon::NSGA2::Reset)
        .def("RestoreIndividuals", &Operon::NSGA2::RestoreIndividuals, nb::call_guard<nb::gil_scoped_release>())
        // See GeneticAlgorithmBase's getters above for why this is a lambda
        // rather than nb::overload_cast against a deducing-this member.
        .def_prop_rw("IsFitted",[](Operon::NSGA2 const& self) { return self.IsFitted(); },[](Operon::NSGA2& self, bool value) {
                self.IsFitted() = value;
            }, nb::call_guard<nb::gil_scoped_release>())
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
