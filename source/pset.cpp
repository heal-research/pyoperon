// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "pyoperon/pyoperon.hpp"
#include <nanobind/stl/pair.h>

#include <operon/core/pset.hpp>

void InitPset(nb::module_ &m)
{
    // primitive set
    nb::class_<Operon::PrimitiveSet>(m, "PrimitiveSet")
        .def(nb::init<>())
        .def_prop_ro_static("Arithmetic", [](nb::object /* self */) { return Operon::PrimitiveSet::Arithmetic; })
        .def_prop_ro_static("TypeCoherent", [](nb::object /* self */) { return Operon::PrimitiveSet::TypeCoherent; })
        .def("IsEnabled", nb::overload_cast<Operon::Hash>(&Operon::PrimitiveSet::IsEnabled, nb::const_))
        .def("IsEnabled", nb::overload_cast<Operon::Node>(&Operon::PrimitiveSet::IsEnabled, nb::const_))
        .def("Enable", nb::overload_cast<Operon::Hash>(&Operon::PrimitiveSet::Enable))
        .def("Enable", nb::overload_cast<Operon::Node>(&Operon::PrimitiveSet::Enable))
        .def("Disable", nb::overload_cast<Operon::Hash>(&Operon::PrimitiveSet::Disable))
        .def("Disable", nb::overload_cast<Operon::Node>(&Operon::PrimitiveSet::Disable))
        .def("Config", &Operon::PrimitiveSet::Config)
        .def("SetConfig", &Operon::PrimitiveSet::SetConfig)
        .def("Frequency", nb::overload_cast<Operon::Hash>(&Operon::PrimitiveSet::Frequency, nb::const_))
        .def("Frequency", nb::overload_cast<Operon::Node>(&Operon::PrimitiveSet::Frequency, nb::const_))
        .def("GetMinimumArity", nb::overload_cast<Operon::Hash>(&Operon::PrimitiveSet::MinimumArity, nb::const_))
        .def("GetMinimumArity", nb::overload_cast<Operon::Node>(&Operon::PrimitiveSet::MinimumArity, nb::const_))
        .def("GetMaximumArity", nb::overload_cast<Operon::Hash>(&Operon::PrimitiveSet::MaximumArity, nb::const_))
        .def("GetMaximumArity", nb::overload_cast<Operon::Node>(&Operon::PrimitiveSet::MaximumArity, nb::const_))
        .def("GetMinMaxArity", nb::overload_cast<Operon::Hash>(&Operon::PrimitiveSet::MinMaxArity, nb::const_))
        .def("GetMinMaxArity", nb::overload_cast<Operon::Node>(&Operon::PrimitiveSet::MinMaxArity, nb::const_))
        .def("SetFrequency", nb::overload_cast<Operon::Hash, size_t>(&Operon::PrimitiveSet::SetFrequency))
        .def("SetFrequency", nb::overload_cast<Operon::Node, size_t>(&Operon::PrimitiveSet::SetFrequency))
        .def("SetMinimumArity", nb::overload_cast<Operon::Hash, size_t>(&Operon::PrimitiveSet::SetMinimumArity))
        .def("SetMinimumArity", nb::overload_cast<Operon::Node, size_t>(&Operon::PrimitiveSet::SetMinimumArity))
        .def("SetMaximumArity", nb::overload_cast<Operon::Hash, size_t>(&Operon::PrimitiveSet::SetMaximumArity))
        .def("SetMaximumArity", nb::overload_cast<Operon::Node, size_t>(&Operon::PrimitiveSet::SetMaximumArity))
        .def("SetMinMaxArity", nb::overload_cast<Operon::Hash, size_t, size_t>(&Operon::PrimitiveSet::SetMinMaxArity))
        .def("SetMinMaxArity", nb::overload_cast<Operon::Node, size_t, size_t>(&Operon::PrimitiveSet::SetMinMaxArity))
        .def("FunctionArityLimits", &Operon::PrimitiveSet::FunctionArityLimits)
        .def("SampleRandomSymbol", &Operon::PrimitiveSet::SampleRandomSymbol);

}
