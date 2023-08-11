// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "pyoperon/pyoperon.hpp"
#include <operon/core/subtree.hpp>
#include <operon/core/tree.hpp>

void InitTree(py::module_ &m)
{
    // tree
    py::class_<Operon::Tree>(m, "Tree")
        .def(py::init<std::initializer_list<Operon::Node>>())
        .def(py::init<Operon::Vector<Operon::Node>>())
        .def(py::init<const Operon::Tree&>())
        .def("UpdateNodes", &Operon::Tree::UpdateNodes)
        .def("Sort", &Operon::Tree::Sort)
        .def("Hash", &Operon::Tree::Hash)
        .def("Reduce", &Operon::Tree::Reduce)
        .def("SetEnabled", &Operon::Tree::SetEnabled)
        .def("SetCoefficients", [](Operon::Tree& tree, py::array_t<Operon::Scalar const> coefficients){
            tree.SetCoefficients(MakeSpan(coefficients));
        }, py::arg("coefficients"))
        .def("GetCoefficients", &Operon::Tree::GetCoefficients)
        .def_property_readonly("CoefficientsCount", &Operon::Tree::CoefficientsCount)
        .def_property_readonly("Nodes", static_cast<Operon::Vector<Operon::Node>& (Operon::Tree::*)()&>(&Operon::Tree::Nodes))
        .def_property_readonly("Nodes", static_cast<Operon::Vector<Operon::Node> const& (Operon::Tree::*)() const&>(&Operon::Tree::Nodes))
        //.def_property_readonly("Nodes", static_cast<Operon::Vector<Operon::Node>&& (Operon::Tree::*)() &&>(&Operon::Tree::Nodes))
        .def_property_readonly("Indices", [](Operon::Tree const& tree, std::size_t i) { return tree.Indices(i); })
        .def_property_readonly("Children", py::overload_cast<std::size_t>(&Operon::Tree::Children))
        .def_property_readonly("Children", py::overload_cast<std::size_t>(&Operon::Tree::Children, py::const_))
        .def_property_readonly("Length", &Operon::Tree::Length)
        .def_property_readonly("VisitationLength", &Operon::Tree::VisitationLength)
        .def_property_readonly("Depth", static_cast<size_t (Operon::Tree::*)() const>(&Operon::Tree::Depth))
        .def_property_readonly("Empty", &Operon::Tree::Empty)
        .def_property_readonly("HashValue", &Operon::Tree::HashValue)
        .def("__getitem__", py::overload_cast<size_t>(&Operon::Tree::operator[]))
        .def("__getitem__", py::overload_cast<size_t>(&Operon::Tree::operator[], py::const_))
        .def(py::pickle(
            [](Operon::Tree const& tree) {
                return py::make_tuple(tree.Nodes());
            },
            [](py::tuple t) {
                if (t.size() != 1) {
                    throw std::runtime_error("Invalid state!");
                }
                return Operon::Tree(t[0].cast<Operon::Vector<Operon::Node>>()).UpdateNodes();
            }
        ));

}
