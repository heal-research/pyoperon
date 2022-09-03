// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include <operon/optimizer/optimizer.hpp>
#include "pyoperon/pyoperon.hpp"

namespace py = pybind11;

void InitOptimizer(py::module_ &m)
{
    using EigenOptimizer = Operon::NonlinearLeastSquaresOptimizer<Operon::OptimizerType::EIGEN>;
    using TinyOptimizer = Operon::NonlinearLeastSquaresOptimizer<Operon::OptimizerType::TINY>;
    using CeresOptimizer = Operon::NonlinearLeastSquaresOptimizer<Operon::OptimizerType::CERES>;

    py::class_<Operon::OptimizerSummary>(m, "OptimizerSummary")
        .def_readwrite("InitialCost", &Operon::OptimizerSummary::InitialCost)
        .def_readwrite("FinalCost", &Operon::OptimizerSummary::FinalCost)
        .def_readwrite("Iterations", &Operon::OptimizerSummary::Iterations)
        .def_readwrite("FunctionEvaluations", &Operon::OptimizerSummary::FunctionEvaluations)
        .def_readwrite("JacobianEvaluations", &Operon::OptimizerSummary::JacobianEvaluations)
        .def_readwrite("Success", &Operon::OptimizerSummary::Success);

    py::class_<EigenOptimizer>(m, "Optimizer")
        .def(py::init<Operon::Interpreter const&, Operon::Tree const&, Operon::Dataset const&>())
        .def("Optimize", &EigenOptimizer::Optimize); 

    m.def("Optimize", [](Operon::Interpreter const& interpreter, Operon::Tree const& tree, Operon::Dataset const& ds, py::array_t<Operon::Scalar const> target, Operon::Range range, size_t iterations) {
        EigenOptimizer optimizer(interpreter, tree, ds);
        Operon::OptimizerSummary summary{};
        auto coeff = optimizer.Optimize(MakeSpan(target), range, iterations, summary);
        return std::make_tuple(coeff, summary);
    });
}
