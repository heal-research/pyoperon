// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "pyoperon/pyoperon.hpp"
#include <operon/operators/creator.hpp>
#include <operon/operators/initializer.hpp>
#include <operon/operators/mutation.hpp>

void InitMutation(py::module_ &m)
{
    using NormalReal = std::normal_distribution<Operon::Scalar>; // distribution for perturbing leaf coefficients
    using UniformReal = std::uniform_real_distribution<Operon::Scalar>;
    using UniformInt = std::uniform_int_distribution<int>;

    // mutation
    py::class_<Operon::MutatorBase> mutatorBase(m, "MutatorBase");

    py::class_<Operon::OnePointMutation<NormalReal>, Operon::MutatorBase>(m, "NormalOnePointMutation")
        .def(py::init<>())
        .def("__call__", &Operon::OnePointMutation<NormalReal>::operator())
        .def("ParameterizeDistribution", &Operon::OnePointMutation<NormalReal>::ParameterizeDistribution<Operon::Scalar, Operon::Scalar>);

    py::class_<Operon::OnePointMutation<UniformInt>, Operon::MutatorBase>(m, "UniformIntOnePointMutation")
        .def(py::init<>())
        .def("__call__", &Operon::OnePointMutation<UniformInt>::operator())
        .def("ParameterizeDistribution", &Operon::OnePointMutation<UniformInt>::ParameterizeDistribution<int, int>);

    py::class_<Operon::OnePointMutation<UniformReal>, Operon::MutatorBase>(m, "UniformRealOnePointMutation")
        .def(py::init<>())
        .def("__call__", &Operon::OnePointMutation<UniformReal>::operator())
        .def("ParameterizeDistribution", &Operon::OnePointMutation<UniformReal>::ParameterizeDistribution<Operon::Scalar, Operon::Scalar>);

    py::class_<Operon::DiscretePointMutation, Operon::MutatorBase>(m, "DiscretePointMutation")
        .def(py::init<>())
        .def("__call__", &Operon::DiscretePointMutation::operator())
        .def("Add", &Operon::DiscretePointMutation::Add);

    py::class_<Operon::ChangeVariableMutation, Operon::MutatorBase>(m, "ChangeVariableMutation")
        .def(py::init([](std::vector<Operon::Variable> const& variables) {
                    return Operon::ChangeVariableMutation(Operon::Span<const Operon::Variable>(variables.data(), variables.size()));
                }),
            py::arg("variables"))
        .def("__call__", &Operon::ChangeVariableMutation::operator());

    py::class_<Operon::ChangeFunctionMutation, Operon::MutatorBase>(m, "ChangeFunctionMutation")
        .def(py::init<Operon::PrimitiveSet>())
        .def("__call__", &Operon::ChangeFunctionMutation::operator());

    py::class_<Operon::ReplaceSubtreeMutation, Operon::MutatorBase>(m, "ReplaceSubtreeMutation")
        .def(py::init<Operon::CreatorBase&, Operon::CoefficientInitializerBase&, size_t, size_t>())
        .def("__call__", &Operon::ReplaceSubtreeMutation::operator());

    py::class_<Operon::RemoveSubtreeMutation, Operon::MutatorBase>(m, "RemoveSubtreeMutation")
        .def(py::init<Operon::PrimitiveSet>())
        .def("__call__", &Operon::RemoveSubtreeMutation::operator());

    py::class_<Operon::InsertSubtreeMutation, Operon::MutatorBase>(m, "InsertSubtreeMutation")
        .def(py::init<Operon::CreatorBase&, Operon::CoefficientInitializerBase&, size_t, size_t>())
        .def("__call__", &Operon::InsertSubtreeMutation::operator());

    py::class_<Operon::MultiMutation, Operon::MutatorBase>(m, "MultiMutation")
        .def(py::init<>())
        .def("__call__", &Operon::MultiMutation::operator())
        .def("Add", &Operon::MultiMutation::Add)
        .def_property_readonly("Count", &Operon::MultiMutation::Count);

}
