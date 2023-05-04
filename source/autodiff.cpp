
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <operon/interpreter/interpreter.hpp>
#include <operon/autodiff/autodiff.hpp>
#include "pyoperon/pyoperon.hpp"

namespace py = pybind11;


void InitAutodiff(py::module_ &m)
{
    using ReverseDerivativeCalculator = Operon::Autodiff::DerivativeCalculator<Operon::Interpreter, Operon::Autodiff::AutodiffMode::Reverse>;
    using ForwardDerivativeCalculator = Operon::Autodiff::DerivativeCalculator<Operon::Interpreter, Operon::Autodiff::AutodiffMode::Forward>;

    py::class_<ReverseDerivativeCalculator>(m, "ReverseDerivativeCalculator")
        .def(py::init([](Operon::Interpreter const& interpreter) {
                return ReverseDerivativeCalculator(interpreter, Operon::Autodiff::AutodiffMode::Reverse);
            }))
        .def("__call__", [](ReverseDerivativeCalculator const& self, Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Range const range) {
                auto coeff = tree.GetCoefficients();
                auto result = py::array_t<Operon::Scalar>(static_cast<pybind11::ssize_t>(range.Size() * coeff.size()));
                auto span = MakeSpan(result);
                self.operator()<Eigen::RowMajor>(tree, dataset, range, { coeff }, span);
                return result;
            })
        .def("__call__", [](ReverseDerivativeCalculator const& self, Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Range const range, py::array_t<Operon::Scalar> result) {
                auto coeff = tree.GetCoefficients();
                auto span = MakeSpan(result);
                self.operator()<Eigen::RowMajor>(tree, dataset, range, { coeff }, span);
            })
        ;

    py::class_<ForwardDerivativeCalculator>(m, "ForwardDerivativeCalculator")
        .def(py::init([](Operon::Interpreter const& interpreter) {
                return ForwardDerivativeCalculator(interpreter, Operon::Autodiff::AutodiffMode::Forward);
            }))
        .def("__call__", [](ForwardDerivativeCalculator const& self, Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Range const range) {
                auto coeff = tree.GetCoefficients();
                auto result = py::array_t<Operon::Scalar>(static_cast<pybind11::ssize_t>(range.Size() * coeff.size()));
                auto span = MakeSpan(result);
                self.operator()<Eigen::RowMajor>(tree, dataset, range, { coeff }, span);
                return result;
            })
        .def("__call__", [](ForwardDerivativeCalculator const& self, Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Range const range, py::array_t<Operon::Scalar> result) {
                auto coeff = tree.GetCoefficients();
                auto span = MakeSpan(result);
                self.operator()<Eigen::RowMajor>(tree, dataset, range, { coeff }, span);
            })
        ;
}
