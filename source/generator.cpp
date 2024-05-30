// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "pyoperon/pyoperon.hpp"

#include <operon/operators/generator.hpp>

void InitGenerator(py::module_ &m)
{
    // offspring generator base
    py::class_<Operon::OffspringGeneratorBase>(m, "OffspringGeneratorBase")
        .def_property_readonly("Terminate", &Operon::OffspringGeneratorBase::Terminate)
        .def("Prepare", [](Operon::OffspringGeneratorBase& self, std::vector<Operon::Individual> const& individuals) {
            Operon::Span<const Operon::Individual> s(individuals.data(), individuals.size());
            self.Prepare(s);
        })
        .def("__call__", [](Operon::OffspringGeneratorBase& self, Operon::RandomGenerator& rng, double pCross, double pMut, double pLocal, Operon::Span<Operon::Scalar> buf = {}) {
                return self(rng, pCross, pMut, pLocal, buf);
                },
            py::arg("rng"),
            py::arg("crossover_probability"),
            py::arg("mutation_probability"),
            py::arg("local search probability"),
            py::arg("evaluation buffer")
        )
        .def("__call__", [](Operon::OffspringGeneratorBase& self, Operon::RandomGenerator& rng, double pCross, double pMut, double pLocal) {
                return self(rng, pCross, pMut, pLocal, std::span<Operon::Scalar>{});
                },
            py::arg("rng"),
            py::arg("crossover_probability"),
            py::arg("mutation_probability"),
            py::arg("local search probability")
        );

    // basic offspring generator
    py::class_<Operon::BasicOffspringGenerator, Operon::OffspringGeneratorBase>(m, "BasicOffspringGenerator")
        .def(py::init<Operon::EvaluatorBase&, Operon::CrossoverBase&, Operon::MutatorBase&,
                Operon::SelectorBase&, Operon::SelectorBase&, Operon::CoefficientOptimizer const*>())
        .def("__call__", [](Operon::OffspringGeneratorBase& self, Operon::RandomGenerator& rng, double pc, double pm, double pl, size_t n) {
            std::vector<Operon::Individual> v;
            v.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                if (auto res = self(rng, pc, pm, pl, std::span<Operon::Scalar>{}); res.has_value()) {
                    v.push_back(res.value());
                }
            }
            return v;
        });

    // offspring selection generator
    py::class_<Operon::OffspringSelectionGenerator, Operon::OffspringGeneratorBase>(m, "OffspringSelectionGenerator")
        .def(py::init<Operon::EvaluatorBase&, Operon::CrossoverBase&, Operon::MutatorBase&,
                Operon::SelectorBase&, Operon::SelectorBase&>())
        .def("__call__", [](Operon::OffspringGeneratorBase& self, Operon::RandomGenerator& rng, double pc, double pm, double pl, size_t n) {
            std::vector<Operon::Individual> v;
            v.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                if (auto res = self(rng, pc, pm, pl, std::span<Operon::Scalar>{}); res.has_value()) {
                    v.push_back(res.value());
                }
            }
            return v;
        })
        .def_property("MaxSelectionPressure",
                py::overload_cast<>(&Operon::OffspringSelectionGenerator::MaxSelectionPressure, py::const_), // getter
                py::overload_cast<size_t>(&Operon::OffspringSelectionGenerator::MaxSelectionPressure)        // setter
                )
        .def_property("ComparisonFactor",
                py::overload_cast<>(&Operon::OffspringSelectionGenerator::ComparisonFactor, py::const_), // getter
                py::overload_cast<double>(&Operon::OffspringSelectionGenerator::ComparisonFactor)        // setter
                )
        .def_property_readonly("SelectionPressure", &Operon::OffspringSelectionGenerator::SelectionPressure);

    // brood generator
    py::class_<Operon::BroodOffspringGenerator, Operon::OffspringGeneratorBase>(m, "BroodOffspringGenerator")
        .def(py::init<Operon::EvaluatorBase&, Operon::CrossoverBase&, Operon::MutatorBase&,
                Operon::SelectorBase&, Operon::SelectorBase&>())
        .def("__call__", [](Operon::OffspringGeneratorBase& self, Operon::RandomGenerator& rng, double pc, double pm, double pl, size_t n) {
            std::vector<Operon::Individual> v;
            v.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                if (auto res = self(rng, pc, pm, pl, std::span<Operon::Scalar>{}); res.has_value()) {
                    v.push_back(res.value());
                }
            }
            return v;
        })
        .def_property("BroodSize",
                py::overload_cast<>(&Operon::BroodOffspringGenerator::BroodSize, py::const_), // getter
                py::overload_cast<size_t>(&Operon::BroodOffspringGenerator::BroodSize)        // setter
                );

    // polygenic generator
    py::class_<Operon::PolygenicOffspringGenerator, Operon::OffspringGeneratorBase>(m, "PolygenicOffspringGenerator")
        .def(py::init<Operon::EvaluatorBase&, Operon::CrossoverBase&, Operon::MutatorBase&,
                Operon::SelectorBase&, Operon::SelectorBase&>())
        .def("__call__", [](Operon::OffspringGeneratorBase& self, Operon::RandomGenerator& rng, double pc, double pm, double pl, size_t n) {
            std::vector<Operon::Individual> v;
            v.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                if (auto res = self(rng, pc, pm, pl, std::span<Operon::Scalar>{}); res.has_value()) {
                    v.push_back(res.value());
                }
            }
            return v;
        })
        .def_property("BroodSize",
                py::overload_cast<>(&Operon::PolygenicOffspringGenerator::PolygenicSize, py::const_), // getter
                py::overload_cast<size_t>(&Operon::PolygenicOffspringGenerator::PolygenicSize)        // setter
                );
}
