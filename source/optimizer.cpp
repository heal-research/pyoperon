// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include <operon/autodiff/autodiff.hpp>
#include <operon/optimizer/optimizer.hpp>
#include "pyoperon/pyoperon.hpp"

namespace py = pybind11;

void InitOptimizer(py::module_ &m)
{
    using DerivativeCalculator = Operon::Autodiff::Reverse::DerivativeCalculator<Operon::Interpreter>;
    using EigenOptimizer = Operon::NonlinearLeastSquaresOptimizer<DerivativeCalculator, Operon::OptimizerType::Eigen>;
    using TinyOptimizer = Operon::NonlinearLeastSquaresOptimizer<DerivativeCalculator, Operon::OptimizerType::Tiny>;
    using CeresOptimizer = Operon::NonlinearLeastSquaresOptimizer<DerivativeCalculator, Operon::OptimizerType::Ceres>;

    py::class_<Operon::OptimizerSummary>(m, "OptimizerSummary")
        .def_readwrite("InitialCost", &Operon::OptimizerSummary::InitialCost)
        .def_readwrite("FinalCost", &Operon::OptimizerSummary::FinalCost)
        .def_readwrite("Iterations", &Operon::OptimizerSummary::Iterations)
        .def_readwrite("FunctionEvaluations", &Operon::OptimizerSummary::FunctionEvaluations)
        .def_readwrite("JacobianEvaluations", &Operon::OptimizerSummary::JacobianEvaluations)
        .def_readwrite("Success", &Operon::OptimizerSummary::Success);

    py::class_<EigenOptimizer>(m, "Optimizer")
        .def(py::init<DerivativeCalculator const&, Operon::Tree const&, Operon::Dataset const&>())
        .def("Optimize", &EigenOptimizer::Optimize); 

    m.def("Optimize", [](DerivativeCalculator const& dc, Operon::Tree const& tree, Operon::Dataset const& ds, py::array_t<Operon::Scalar const> target, Operon::Range range, std::size_t iterations) {
        EigenOptimizer optimizer(dc, tree, ds);
        Operon::OptimizerSummary summary{};
        auto coeff = optimizer.Optimize(MakeSpan(target), range, iterations, summary);
        return std::make_tuple(coeff, summary);
    });

    m.def("Optimize", [](Operon::Interpreter const& interpreter, Operon::Tree const& tree, Operon::Dataset const& ds, py::array_t<Operon::Scalar const> target, Operon::Range range, size_t iterations) {
        DerivativeCalculator dc{interpreter};
        EigenOptimizer optimizer(dc, tree, ds);
        Operon::OptimizerSummary summary{};
        auto coeff = optimizer.Optimize(MakeSpan(target), range, iterations, summary);
        return std::make_tuple(coeff, summary);
    });
}
