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
    py::class_<Operon::GeneticAlgorithmBase>(m, "GeneticAlgorithmBase")
        .def_property_readonly("Generation", static_cast<size_t (Operon::GeneticAlgorithmBase::*)() const>(&Operon::GeneticAlgorithmBase::Generation), py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("Individuals", static_cast<std::vector<Operon::Individual> const& (Operon::GeneticAlgorithmBase::*)() const>(&Operon::GeneticAlgorithmBase::Individuals), py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("Parents", static_cast<std::span<Operon::Individual const> (Operon::GeneticAlgorithmBase::*)() const>(&Operon::GeneticAlgorithmBase::Parents), py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("Offspring", static_cast<std::span<Operon::Individual const> (Operon::GeneticAlgorithmBase::*)() const>(&Operon::GeneticAlgorithmBase::Offspring), py::call_guard<py::gil_scoped_release>())
        ;

    py::class_<Operon::GeneticProgrammingAlgorithm, Operon::GeneticAlgorithmBase>(m, "GeneticProgrammingAlgorithm")
        .def(py::init<Operon::Problem const&, Operon::GeneticAlgorithmConfig const&, Operon::TreeInitializerBase const&,
                Operon::CoefficientInitializerBase const&, Operon::OffspringGeneratorBase const&, Operon::ReinserterBase const&>())
        .def("Run", py::overload_cast<Operon::RandomGenerator&, std::function<void()>, size_t>(&Operon::GeneticProgrammingAlgorithm::Run),
                py::call_guard<py::gil_scoped_release>(), py::arg("rng"), py::arg("callback") = nullptr, py::arg("threads") = 0)
        .def("Reset", &Operon::GeneticProgrammingAlgorithm::Reset, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("BestModel", [](Operon::GeneticProgrammingAlgorithm const& self) {
                auto minElem = std::min_element(self.Parents().begin(), self.Parents().end(), [&](auto const& a, auto const& b) { return a[0] < b[0]; });
                return *minElem;
            }, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("Config", &Operon::GeneticProgrammingAlgorithm::GetConfig, py::call_guard<py::gil_scoped_release>());

    py::class_<Operon::NSGA2, Operon::GeneticAlgorithmBase>(m, "NSGA2Algorithm")
        .def(py::init<Operon::Problem const&, Operon::GeneticAlgorithmConfig const&, Operon::TreeInitializerBase const&, Operon::CoefficientInitializerBase const&,
                Operon::OffspringGeneratorBase const&, Operon::ReinserterBase const&, Operon::NondominatedSorterBase const&>())
        .def("Run", py::overload_cast<Operon::RandomGenerator&, std::function<void()>, size_t>(&Operon::NSGA2::Run),
                py::call_guard<py::gil_scoped_release>(), py::arg("rng"), py::arg("callback") = nullptr, py::arg("threads") = 0)
        .def("Reset", &Operon::NSGA2::Reset)
        .def_property_readonly("BestModel", [](Operon::NSGA2 const& self) {
                auto minElem = std::min_element(self.Best().begin(), self.Best().end(), [&](auto const& a, auto const& b) { return a[0] < b[0];});
                return *minElem;
            }, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("BestFront", [](Operon::NSGA2 const& self) {
                    auto best = self.Best();
                    return std::vector<Operon::Individual>(best.begin(), best.end());
                }, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("Config", &Operon::NSGA2::GetConfig, py::call_guard<py::gil_scoped_release>());
}
