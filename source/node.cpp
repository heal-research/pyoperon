// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "pyoperon/pyoperon.hpp"

#include <nanobind/operators.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <operon/core/node.hpp>
#include <utility>

namespace nb = nanobind;

void InitNode(nb::module_ &m)
{
    // node type
    nb::enum_<Operon::NodeType>(m, "NodeType", nb::is_arithmetic())
        .value("Add", Operon::NodeType::Add)
        .value("Mul", Operon::NodeType::Mul)
        .value("Sub", Operon::NodeType::Sub)
        .value("Div", Operon::NodeType::Div)
        .value("Fmin", Operon::NodeType::Fmax)
        .value("Fmax", Operon::NodeType::Fmin)
        .value("Aq", Operon::NodeType::Aq)
        .value("Pow", Operon::NodeType::Pow)
        .value("Abs", Operon::NodeType::Abs)
        .value("Acos", Operon::NodeType::Acos)
        .value("Asin", Operon::NodeType::Asin)
        .value("Atan", Operon::NodeType::Atan)
        .value("Cbrt", Operon::NodeType::Cbrt)
        .value("Ceil", Operon::NodeType::Ceil)
        .value("Cos", Operon::NodeType::Cos)
        .value("Cosh", Operon::NodeType::Cosh)
        .value("Exp", Operon::NodeType::Exp)
        .value("Floor", Operon::NodeType::Floor)
        .value("Log", Operon::NodeType::Log)
        .value("Logabs", Operon::NodeType::Logabs)
        .value("Log1p", Operon::NodeType::Log1p)
        .value("Sin", Operon::NodeType::Sin)
        .value("Sinh", Operon::NodeType::Sinh)
        .value("Sqrt", Operon::NodeType::Sqrt)
        .value("Sqrtabs", Operon::NodeType::Sqrtabs)
        .value("Tan", Operon::NodeType::Tan)
        .value("Tanh", Operon::NodeType::Tanh)
        .value("Square", Operon::NodeType::Square)
        .value("Dyn", Operon::NodeType::Dynamic)
        .value("Constant", Operon::NodeType::Constant)
        .value("Variable", Operon::NodeType::Variable);
        // expose overloaded operators
        // .def(nb::self & nb::self)
        // .def(nb::self &= nb::self)
        // .def(nb::self | nb::self)
        // .def(nb::self |= nb::self)
        // .def(nb::self ^ nb::self)
        // .def(nb::self ^= nb::self)
        // .def(~nb::self);
    // node
    nb::class_<Operon::Node>(m, "Node")
        .def(nb::init<Operon::NodeType>())
        .def(nb::init<Operon::NodeType, Operon::Hash>())
        .def_prop_ro("Name", &Operon::Node::Name)
        .def_prop_ro("IsLeaf", &Operon::Node::IsLeaf)
        .def_prop_ro("IsConstant", &Operon::Node::IsConstant)
        .def_prop_ro("IsVariable", &Operon::Node::IsVariable)
        .def_prop_ro("IsCommutative", &Operon::Node::IsCommutative)
        .def_rw("Value", &Operon::Node::Value)
        .def_rw("HashValue", &Operon::Node::HashValue)
        .def_rw("CalculatedHashValue", &Operon::Node::CalculatedHashValue)
        .def_rw("Arity", &Operon::Node::Arity)
        .def_rw("Length", &Operon::Node::Length)
        .def_rw("Depth", &Operon::Node::Depth)
        .def_rw("Level", &Operon::Node::Level)
        .def_rw("Parent", &Operon::Node::Parent)
        .def_rw("Type", &Operon::Node::Type)
        .def_rw("IsEnabled", &Operon::Node::IsEnabled)
        .def_rw("Optimize", &Operon::Node::Optimize)
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def(nb::self < nb::self)
        .def(nb::self <= nb::self)
        .def(nb::self > nb::self)
        .def(nb::self >= nb::self)
        // node factory for convenience
        .def("Add", []() { return Operon::Node(Operon::NodeType::Add); })
        .def("Sub", []() { return Operon::Node(Operon::NodeType::Sub); })
        .def("Mul", []() { return Operon::Node(Operon::NodeType::Mul); })
        .def("Div", []() { return Operon::Node(Operon::NodeType::Div); })
        .def("Aq", []() { return Operon::Node(Operon::NodeType::Aq); })
        .def("Pow", []() { return Operon::Node(Operon::NodeType::Pow); })
        .def("Exp", []() { return Operon::Node(Operon::NodeType::Exp); })
        .def("Log", []() { return Operon::Node(Operon::NodeType::Log); })
        .def("Sin", []() { return Operon::Node(Operon::NodeType::Sin); })
        .def("Cos", []() { return Operon::Node(Operon::NodeType::Cos); })
        .def("Tan", []() { return Operon::Node(Operon::NodeType::Tan); })
        .def("Tanh", []() { return Operon::Node(Operon::NodeType::Tanh); })
        .def("Sqrt", []() { return Operon::Node(Operon::NodeType::Sqrt); })
        .def("Cbrt", []() { return Operon::Node(Operon::NodeType::Cbrt); })
        .def("Square", []() { return Operon::Node(Operon::NodeType::Square); })
        .def("Dyn", []() { return Operon::Node(Operon::NodeType::Dynamic); })
        .def("Constant", [](double v) {
                Operon::Node constant(Operon::NodeType::Constant);
                constant.Value = static_cast<Operon::Scalar>(v);
                return constant;
                })
        .def("Variable", [](double w) {
                Operon::Node variable(Operon::NodeType::Variable);
                variable.Value = static_cast<Operon::Scalar>(w);
                return variable;
                })
        // pickle support
        .def("__getstate__",
            [](Operon::Node const& n) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return std::make_tuple(
                    n.HashValue,
                    n.CalculatedHashValue,
                    n.Value,
                    n.Arity,
                    n.Length,
                    n.Depth,
                    n.Level,
                    n.Parent,
                    n.Type,
                    n.IsEnabled,
                    n.Optimize
                );
            })
        .def("__setstate__",
            [](Operon::Node& n, std::tuple<Operon::Hash, Operon::Hash, Operon::Scalar, uint16_t, uint16_t, uint16_t, uint16_t, uint16_t, Operon::NodeType, bool, bool> const& t) {
                auto constexpr tupleSize{ 11 };
                if (std::tuple_size_v<std::remove_cvref_t<decltype(t)>> != tupleSize) {
                    throw std::runtime_error("Invalid state!");
                }

                // Use placement new to populate the node
                n.HashValue           = std::get<0>(t);
                n.CalculatedHashValue = std::get<1>(t);
                n.Value               = std::get<2>(t);
                n.Arity               = std::get<3>(t);
                n.Length              = std::get<4>(t);
                n.Depth               = std::get<5>(t);
                n.Level               = std::get<6>(t);
                n.Parent              = std::get<7>(t);
                n.Type                = std::get<8>(t);
                n.IsEnabled           = std::get<9>(t);
                n.Optimize            = std::get<10>(t);

                return n;
            })
        ;

}
