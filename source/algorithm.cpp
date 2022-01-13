// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "pyoperon/pyoperon.hpp"
#include <operon/algorithms/gp.hpp>
#include <operon/algorithms/nsga2.hpp>
#include <operon/operators/initializer.hpp>
#include <operon/operators/non_dominated_sorter.hpp>
#include <operon/operators/reinserter.hpp>

#include <pybind11/detail/common.h>

void InitAlgorithm(py::module_ &m)
{
    py::class_<Operon::GeneticProgrammingAlgorithm>(m, "GeneticProgrammingAlgorithm")
        .def(py::init<Operon::Problem const&, Operon::GeneticAlgorithmConfig const&, Operon::TreeInitializerBase const&,
                Operon::CoefficientInitializerBase const&, Operon::OffspringGeneratorBase const&, Operon::ReinserterBase const&>())
        .def("Run", py::overload_cast<Operon::RandomGenerator&, std::function<void()>, size_t>(&Operon::GeneticProgrammingAlgorithm::Run),
                py::call_guard<py::gil_scoped_release>(), py::arg("rng"), py::arg("callback") = nullptr, py::arg("threads") = 0)
        .def("Reset", &Operon::GeneticProgrammingAlgorithm::Reset)
        .def("BestModel", [](Operon::GeneticProgrammingAlgorithm const& self, Operon::Comparison const& comparison) {
                    auto min_elem = std::min_element(self.Parents().begin(), self.Parents().end(), [&](auto const& a, auto const& b) { return comparison(a, b);});
                    return *min_elem;
                })
        .def_property_readonly("Generation", &Operon::GeneticProgrammingAlgorithm::Generation)
        .def_property_readonly("Parents", static_cast<Operon::Span<Operon::Individual const> (Operon::GeneticProgrammingAlgorithm::*)() const>(&Operon::GeneticProgrammingAlgorithm::Parents))
        .def_property_readonly("Offspring", static_cast<Operon::Span<Operon::Individual const> (Operon::GeneticProgrammingAlgorithm::*)() const>(&Operon::GeneticProgrammingAlgorithm::Offspring))
        .def_property_readonly("Config", &Operon::GeneticProgrammingAlgorithm::GetConfig);

    py::class_<Operon::NSGA2>(m, "NSGA2Algorithm")
        .def(py::init<Operon::Problem const&, Operon::GeneticAlgorithmConfig const&, Operon::TreeInitializerBase const&, Operon::CoefficientInitializerBase const&,
                Operon::OffspringGeneratorBase const&, Operon::ReinserterBase const&, Operon::NondominatedSorterBase const&>())
        .def("Run", py::overload_cast<Operon::RandomGenerator&, std::function<void()>, size_t>(&Operon::NSGA2::Run),
                py::call_guard<py::gil_scoped_release>(), py::arg("rng"), py::arg("callback") = nullptr, py::arg("threads") = 0)
        .def("Reset", &Operon::NSGA2::Reset)
        .def("BestModel", [](Operon::NSGA2 const& self, Operon::Comparison const& comparison) {
                    auto min_elem = std::min_element(self.Best().begin(), self.Best().end(), [&](auto const& a, auto const& b) { return comparison(a, b);});
                    return *min_elem;
                })
        .def_property_readonly("Generation", &Operon::NSGA2::Generation)
        .def_property_readonly("Parents", static_cast<Operon::Span<Operon::Individual const> (Operon::NSGA2::*)() const>(&Operon::NSGA2::Parents))
        .def_property_readonly("Offspring", static_cast<Operon::Span<Operon::Individual const> (Operon::NSGA2::*)() const>(&Operon::NSGA2::Offspring))
        .def_property_readonly("BestFront", [](Operon::NSGA2 const& self) {
                auto best = self.Best();
                return std::vector<Operon::Individual>(best.begin(), best.end());
                })
        .def_property_readonly("Config", &Operon::NSGA2::GetConfig);
}
