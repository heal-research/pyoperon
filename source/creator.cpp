// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "pyoperon/pyoperon.hpp"

#include <operon/operators/creator.hpp>
#include <operon/core/pset.hpp>

void InitCreator(py::module_ &m)
{
    // tree creator
    py::class_<Operon::CreatorBase> creatorBase(m, "CreatorBase");

    py::class_<Operon::BalancedTreeCreator, Operon::CreatorBase>(m, "BalancedTreeCreator")
        .def(py::init([](Operon::PrimitiveSet const& grammar, std::vector<Operon::Hash> const& variables, double bias) {
            return Operon::BalancedTreeCreator(grammar, Operon::Span<Operon::Hash const>(variables.data(), variables.size()), bias);
        }), py::arg("grammar"), py::arg("variables"), py::arg("bias"))
        .def("__call__", &Operon::BalancedTreeCreator::operator())
        .def_property("IrregularityBias", &Operon::BalancedTreeCreator::GetBias, &Operon::BalancedTreeCreator::SetBias);

    py::class_<Operon::ProbabilisticTreeCreator, Operon::CreatorBase>(m, "ProbabilisticTreeCreator")
        .def(py::init([](Operon::PrimitiveSet const& grammar, std::vector<Operon::Hash> const& variables, double bias) {
            return Operon::ProbabilisticTreeCreator(grammar, Operon::Span<Operon::Hash const>(variables.data(), variables.size()), bias);
        }), py::arg("grammar"), py::arg("variables"), py::arg("bias"))
        .def("__call__", &Operon::ProbabilisticTreeCreator::operator());

    py::class_<Operon::GrowTreeCreator, Operon::CreatorBase>(m, "GrowTreeCreator")
        .def(py::init([](Operon::PrimitiveSet const& grammar, std::vector<Operon::Hash> const& variables) {
            return Operon::GrowTreeCreator(grammar, Operon::Span<Operon::Hash const>(variables.data(), variables.size()));
        }), py::arg("grammar"), py::arg("variables"))
        .def("__call__", &Operon::GrowTreeCreator::operator());
}
