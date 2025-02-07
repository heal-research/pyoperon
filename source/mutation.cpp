// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "pyoperon/pyoperon.hpp"
#include <operon/operators/creator.hpp>
#include <operon/operators/initializer.hpp>
#include <operon/operators/mutation.hpp>

void InitMutation(nb::module_ &m)
{
    using NormalReal = std::normal_distribution<Operon::Scalar>; // distribution for perturbing leaf coefficients
    using UniformReal = std::uniform_real_distribution<Operon::Scalar>;
    using UniformInt = std::uniform_int_distribution<int>;

    // mutation
    nb::class_<Operon::MutatorBase> mutatorBase(m, "MutatorBase");

    nb::class_<Operon::OnePointMutation<NormalReal>, Operon::MutatorBase>(m, "NormalOnePointMutation")
        .def(nb::init<>())
        .def("__call__", &Operon::OnePointMutation<NormalReal>::operator())
        .def("ParameterizeDistribution", &Operon::OnePointMutation<NormalReal>::ParameterizeDistribution<Operon::Scalar, Operon::Scalar>);

    nb::class_<Operon::OnePointMutation<UniformInt>, Operon::MutatorBase>(m, "UniformIntOnePointMutation")
        .def(nb::init<>())
        .def("__call__", &Operon::OnePointMutation<UniformInt>::operator())
        .def("ParameterizeDistribution", &Operon::OnePointMutation<UniformInt>::ParameterizeDistribution<int, int>);

    nb::class_<Operon::OnePointMutation<UniformReal>, Operon::MutatorBase>(m, "UniformRealOnePointMutation")
        .def(nb::init<>())
        .def("__call__", &Operon::OnePointMutation<UniformReal>::operator())
        .def("ParameterizeDistribution", &Operon::OnePointMutation<UniformReal>::ParameterizeDistribution<Operon::Scalar, Operon::Scalar>);

    nb::class_<Operon::MultiPointMutation<NormalReal>, Operon::MutatorBase>(m, "NormalMultiPointMutation")
        .def(nb::init<>())
        .def("__call__", &Operon::MultiPointMutation<NormalReal>::operator())
        .def("ParameterizeDistribution", &Operon::MultiPointMutation<NormalReal>::ParameterizeDistribution<Operon::Scalar, Operon::Scalar>);

    nb::class_<Operon::MultiPointMutation<UniformInt>, Operon::MutatorBase>(m, "UniformIntMultiPointMutation")
        .def(nb::init<>())
        .def("__call__", &Operon::MultiPointMutation<UniformInt>::operator())
        .def("ParameterizeDistribution", &Operon::MultiPointMutation<UniformInt>::ParameterizeDistribution<int, int>);

    nb::class_<Operon::MultiPointMutation<UniformReal>, Operon::MutatorBase>(m, "UniformRealMultiPointMutation")
        .def(nb::init<>())
        .def("__call__", &Operon::MultiPointMutation<UniformReal>::operator())
        .def("ParameterizeDistribution", &Operon::MultiPointMutation<UniformReal>::ParameterizeDistribution<Operon::Scalar, Operon::Scalar>);

    nb::class_<Operon::DiscretePointMutation, Operon::MutatorBase>(m, "DiscretePointMutation")
        .def(nb::init<>())
        .def("__call__", &Operon::DiscretePointMutation::operator())
        .def("Add", &Operon::DiscretePointMutation::Add);

    nb::class_<Operon::ChangeVariableMutation, Operon::MutatorBase>(m, "ChangeVariableMutation")
        .def("__init__", [](Operon::ChangeVariableMutation* mut, std::vector<Operon::Hash> const& variables) {
                new (mut) Operon::ChangeVariableMutation(
                    Operon::Span<Operon::Hash const>(variables.data(), variables.size())
                );
            }, nb::arg("variables"))
        .def("__call__", &Operon::ChangeVariableMutation::operator());

    nb::class_<Operon::ChangeFunctionMutation, Operon::MutatorBase>(m, "ChangeFunctionMutation")
        .def(nb::init<Operon::PrimitiveSet>())
        .def("__call__", &Operon::ChangeFunctionMutation::operator());

    nb::class_<Operon::ReplaceSubtreeMutation, Operon::MutatorBase>(m, "ReplaceSubtreeMutation")
        .def(nb::init<Operon::CreatorBase const*, Operon::CoefficientInitializerBase const*, size_t, size_t>())
        .def("__call__", &Operon::ReplaceSubtreeMutation::operator());

    nb::class_<Operon::RemoveSubtreeMutation, Operon::MutatorBase>(m, "RemoveSubtreeMutation")
        .def(nb::init<Operon::PrimitiveSet>())
        .def("__call__", &Operon::RemoveSubtreeMutation::operator());

    nb::class_<Operon::InsertSubtreeMutation, Operon::MutatorBase>(m, "InsertSubtreeMutation")
        .def(nb::init<Operon::CreatorBase const*, Operon::CoefficientInitializerBase const*, size_t, size_t>())
        .def("__call__", &Operon::InsertSubtreeMutation::operator());

    nb::class_<Operon::MultiMutation, Operon::MutatorBase>(m, "MultiMutation")
        .def(nb::init<>())
        .def("__call__", &Operon::MultiMutation::operator())
        .def("Add", &Operon::MultiMutation::Add)
        .def("Add", [](Operon::MultiMutation& self, Operon::MutatorBase const& mut, double prob) {
            self.Add(&mut, prob);
        })
        .def_prop_ro("Count", &Operon::MultiMutation::Count);

}
