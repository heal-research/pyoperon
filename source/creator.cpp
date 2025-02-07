// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "pyoperon/pyoperon.hpp"

#include <operon/operators/creator.hpp>
#include <operon/core/pset.hpp>

void InitCreator(nb::module_ &m)
{
    // tree creator
    nb::class_<Operon::CreatorBase> creatorBase(m, "CreatorBase");

    nb::class_<Operon::BalancedTreeCreator, Operon::CreatorBase>(m, "BalancedTreeCreator")
        .def(nb::init<Operon::PrimitiveSet const*, std::vector<Operon::Hash>, double>()
            , nb::arg("grammar"), nb::arg("variables"), nb::arg("bias"))
        .def("__call__", &Operon::BalancedTreeCreator::operator())
        .def_prop_rw("IrregularityBias", &Operon::BalancedTreeCreator::GetBias, &Operon::BalancedTreeCreator::SetBias);

    nb::class_<Operon::ProbabilisticTreeCreator, Operon::CreatorBase>(m, "ProbabilisticTreeCreator")
        .def(nb::init<Operon::PrimitiveSet const*, std::vector<Operon::Hash>, double>()
            , nb::arg("grammar"), nb::arg("variables"), nb::arg("bias"))
        .def("__call__", &Operon::ProbabilisticTreeCreator::operator())
        .def_prop_rw("IrregularityBias", &Operon::ProbabilisticTreeCreator::GetBias, &Operon::ProbabilisticTreeCreator::SetBias);

    nb::class_<Operon::GrowTreeCreator, Operon::CreatorBase>(m, "GrowTreeCreator")
        .def(nb::init<Operon::PrimitiveSet const*, std::vector<Operon::Hash>>()
            , nb::arg("grammar"), nb::arg("variables"))
        .def("__call__", &Operon::GrowTreeCreator::operator());
}
