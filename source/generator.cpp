// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "pyoperon/pyoperon.hpp"

#include <operon/operators/generator.hpp>

void InitGenerator(nb::module_ &m)
{
    // offspring generator base
    nb::class_<Operon::OffspringGeneratorBase>(m, "OffspringGeneratorBase")
        .def_prop_ro("Terminate", &Operon::OffspringGeneratorBase::Terminate)
        .def("Prepare", [](Operon::OffspringGeneratorBase& self, std::vector<Operon::Individual> const& individuals) {
            Operon::Span<const Operon::Individual> s(individuals.data(), individuals.size());
            self.Prepare(s);
        })
        .def("__call__", [](Operon::OffspringGeneratorBase& self, Operon::RandomGenerator& rng, double pCross, double pMut, double pLocal, Operon::Span<Operon::Scalar> buf = {}) {
                return self(rng, pCross, pMut, pLocal, buf);
                },
            nb::arg("rng"),
            nb::arg("crossover_probability"),
            nb::arg("mutation_probability"),
            nb::arg("local search probability"),
            nb::arg("evaluation buffer")
        )
        .def("__call__", [](Operon::OffspringGeneratorBase& self, Operon::RandomGenerator& rng, double pCross, double pMut, double pLocal) {
                return self(rng, pCross, pMut, pLocal, std::span<Operon::Scalar>{});
                },
            nb::arg("rng"),
            nb::arg("crossover_probability"),
            nb::arg("mutation_probability"),
            nb::arg("local search probability")
        );

    // basic offspring generator
    nb::class_<Operon::BasicOffspringGenerator, Operon::OffspringGeneratorBase>(m, "BasicOffspringGenerator")
        .def(nb::init<Operon::EvaluatorBase&, Operon::CrossoverBase&, Operon::MutatorBase&,
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
    nb::class_<Operon::OffspringSelectionGenerator, Operon::OffspringGeneratorBase>(m, "OffspringSelectionGenerator")
        .def(nb::init<Operon::EvaluatorBase&, Operon::CrossoverBase&, Operon::MutatorBase&,
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
        .def_prop_rw("MaxSelectionPressure",
                nb::overload_cast<>(&Operon::OffspringSelectionGenerator::MaxSelectionPressure, nb::const_), // getter
                nb::overload_cast<size_t>(&Operon::OffspringSelectionGenerator::MaxSelectionPressure)        // setter
                )
        .def_prop_rw("ComparisonFactor",
                nb::overload_cast<>(&Operon::OffspringSelectionGenerator::ComparisonFactor, nb::const_), // getter
                nb::overload_cast<double>(&Operon::OffspringSelectionGenerator::ComparisonFactor)        // setter
                )
        .def_prop_ro("SelectionPressure", &Operon::OffspringSelectionGenerator::SelectionPressure);

    // brood generator
    nb::class_<Operon::BroodOffspringGenerator, Operon::OffspringGeneratorBase>(m, "BroodOffspringGenerator")
        .def(nb::init<Operon::EvaluatorBase&, Operon::CrossoverBase&, Operon::MutatorBase&,
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
        .def_prop_rw("BroodSize",
                nb::overload_cast<>(&Operon::BroodOffspringGenerator::BroodSize, nb::const_), // getter
                nb::overload_cast<size_t>(&Operon::BroodOffspringGenerator::BroodSize)        // setter
                );

    // polygenic generator
    nb::class_<Operon::PolygenicOffspringGenerator, Operon::OffspringGeneratorBase>(m, "PolygenicOffspringGenerator")
        .def(nb::init<Operon::EvaluatorBase&, Operon::CrossoverBase&, Operon::MutatorBase&,
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
        .def_prop_rw("BroodSize",
                nb::overload_cast<>(&Operon::PolygenicOffspringGenerator::PolygenicSize, nb::const_), // getter
                nb::overload_cast<size_t>(&Operon::PolygenicOffspringGenerator::PolygenicSize)        // setter
                );
}
