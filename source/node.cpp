// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "pyoperon/pyoperon.hpp"

#include <nanobind/operators.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <operon/core/node.hpp>
#include <operon/core/standard_library.hpp>
#include <utility>

namespace nb = nanobind;

namespace {
    auto MaxArity(Operon::BuiltinOp op) -> uint16_t
    {
        return Operon::StandardLibrary::ArityLimits(op).second;
    }

    auto MakeBuiltin(Operon::BuiltinOp op) -> Operon::Node
    {
        return Operon::Node::Function(static_cast<Operon::Hash>(op), MaxArity(op));
    }
} // namespace

void InitNode(nb::module_ &m)
{
    // node type: terminal categories (built-in math ops live in BuiltinOp)
    nb::enum_<Operon::NodeType>(m, "NodeType", nb::is_arithmetic())
        .value("Constant", Operon::NodeType::Constant)
        .value("Variable", Operon::NodeType::Variable)
        .value("Ref", Operon::NodeType::Ref)
        .value("Function", Operon::NodeType::Function);

    // built-in math ops
    nb::enum_<Operon::BuiltinOp>(m, "BuiltinOp", nb::is_arithmetic())
        .value("Add", Operon::BuiltinOp::Add)
        .value("Mul", Operon::BuiltinOp::Mul)
        .value("Sub", Operon::BuiltinOp::Sub)
        .value("Div", Operon::BuiltinOp::Div)
        .value("Fmin", Operon::BuiltinOp::Fmin)
        .value("Fmax", Operon::BuiltinOp::Fmax)
        .value("Aq", Operon::BuiltinOp::Aq)
        .value("Pow", Operon::BuiltinOp::Pow)
        .value("Powabs", Operon::BuiltinOp::Powabs)
        .value("Abs", Operon::BuiltinOp::Abs)
        .value("Acos", Operon::BuiltinOp::Acos)
        .value("Asin", Operon::BuiltinOp::Asin)
        .value("Atan", Operon::BuiltinOp::Atan)
        .value("Cbrt", Operon::BuiltinOp::Cbrt)
        .value("Ceil", Operon::BuiltinOp::Ceil)
        .value("Cos", Operon::BuiltinOp::Cos)
        .value("Cosh", Operon::BuiltinOp::Cosh)
        .value("Exp", Operon::BuiltinOp::Exp)
        .value("Floor", Operon::BuiltinOp::Floor)
        .value("Log", Operon::BuiltinOp::Log)
        .value("Logabs", Operon::BuiltinOp::Logabs)
        .value("Log1p", Operon::BuiltinOp::Log1p)
        .value("Sin", Operon::BuiltinOp::Sin)
        .value("Sinh", Operon::BuiltinOp::Sinh)
        .value("Sqrt", Operon::BuiltinOp::Sqrt)
        .value("Sqrtabs", Operon::BuiltinOp::Sqrtabs)
        .value("Tan", Operon::BuiltinOp::Tan)
        .value("Tanh", Operon::BuiltinOp::Tanh)
        .value("Square", Operon::BuiltinOp::Square);

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
        .def_rw("RefTo", &Operon::Node::RefTo)
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def(nb::self < nb::self)
        .def(nb::self <= nb::self)
        .def(nb::self > nb::self)
        .def(nb::self >= nb::self)
        // node factory for convenience
        .def("Add", []() { return MakeBuiltin(Operon::BuiltinOp::Add); })
        .def("Sub", []() { return MakeBuiltin(Operon::BuiltinOp::Sub); })
        .def("Mul", []() { return MakeBuiltin(Operon::BuiltinOp::Mul); })
        .def("Div", []() { return MakeBuiltin(Operon::BuiltinOp::Div); })
        .def("Aq", []() { return MakeBuiltin(Operon::BuiltinOp::Aq); })
        .def("Pow", []() { return MakeBuiltin(Operon::BuiltinOp::Pow); })
        .def("Powabs", []() { return MakeBuiltin(Operon::BuiltinOp::Powabs); })
        .def("Exp", []() { return MakeBuiltin(Operon::BuiltinOp::Exp); })
        .def("Log", []() { return MakeBuiltin(Operon::BuiltinOp::Log); })
        .def("Sin", []() { return MakeBuiltin(Operon::BuiltinOp::Sin); })
        .def("Cos", []() { return MakeBuiltin(Operon::BuiltinOp::Cos); })
        .def("Tan", []() { return MakeBuiltin(Operon::BuiltinOp::Tan); })
        .def("Tanh", []() { return MakeBuiltin(Operon::BuiltinOp::Tanh); })
        .def("Sqrt", []() { return MakeBuiltin(Operon::BuiltinOp::Sqrt); })
        .def("Cbrt", []() { return MakeBuiltin(Operon::BuiltinOp::Cbrt); })
        .def("Square", []() { return MakeBuiltin(Operon::BuiltinOp::Square); })
        .def("Function", [](Operon::Hash hash, uint16_t arity) { return Operon::Node::Function(hash, arity); })
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
        .def("Ref", [](uint16_t target) { return Operon::Node::Ref(target); })
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
                    n.Optimize,
                    n.RefTo
                );
            })
        .def("__setstate__",
            [](Operon::Node& n, std::tuple<Operon::Hash, Operon::Hash, Operon::Scalar, uint16_t, uint16_t, uint16_t, uint16_t, uint16_t, Operon::NodeType, bool, bool, uint16_t> const& t) {
                auto constexpr tupleSize{ 12 };
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
                n.RefTo               = std::get<11>(t);

                return n;
            })
        ;

}
