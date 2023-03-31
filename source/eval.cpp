// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <operon/operators/evaluator.hpp>
#include "pyoperon/pyoperon.hpp"

namespace py = pybind11;

namespace detail {
    template<typename T>
    auto FitLeastSquares(py::array_t<T> lhs, py::array_t<T> rhs) -> std::pair<double, double>
    {
        auto s1 = MakeSpan(lhs);
        auto s2 = MakeSpan(rhs);
        return Operon::FitLeastSquares(s1, s2);
    }
} // namespace detail

void InitEval(py::module_ &m)
{
    // free functions
    // we use a lambda to avoid defining a fourth arg for the defaulted C++ function arg
    m.def("Evaluate", [](Operon::Interpreter const& i, Operon::Tree const& t, Operon::Dataset const& d, Operon::Range r) {
        auto result = py::array_t<Operon::Scalar>(static_cast<pybind11::ssize_t>(r.Size()));
        auto span = MakeSpan(result);
        py::gil_scoped_release release;
        i.operator()<Operon::Scalar>(t, d, r, span);
        py::gil_scoped_acquire acquire;
        return result;
        }, py::arg("interpreter"), py::arg("tree"), py::arg("dataset"), py::arg("range"));

    m.def("EvaluateTrees", [](std::vector<Operon::Tree> const& trees, Operon::Dataset const& ds, Operon::Range range, py::array_t<Operon::Scalar> result, size_t nthread) {
            auto span = MakeSpan(result);
            py::gil_scoped_release release;
            Operon::EvaluateTrees(trees, ds, range, span, nthread);
            py::gil_scoped_acquire acquire;
            }, py::arg("trees"), py::arg("dataset"), py::arg("range"), py::arg("result").noconvert(), py::arg("nthread") = 1);

    m.def("CalculateFitness", [](Operon::Interpreter const& i, Operon::Tree const& t, Operon::Dataset const& d, Operon::Range r, std::string const& target, std::string const& metric) {
        auto estimated = i.operator()<Operon::Scalar>(t, d, r);
        auto values = d.GetValues(target).subspan(r.Start(), r.Size());

        if (metric == "c2") { return Operon::C2{}(estimated, values); }
        if (metric == "r2") { return Operon::R2{}(estimated, values); }
        if (metric == "mse") { return Operon::MSE{}(estimated, values); }
        if (metric == "rmse") { return Operon::RMSE{}(estimated, values); }
        if (metric == "nmse") { return Operon::NMSE{}(estimated, values); }
        if (metric == "mae") { return Operon::MAE{}(estimated, values); }
        throw std::runtime_error("Invalid fitness metric"); 

    }, py::arg("interpreter"), py::arg("tree"), py::arg("dataset"), py::arg("range"), py::arg("target"), py::arg("metric") = "rsquared");

    m.def("CalculateFitness", [](Operon::Interpreter const& i, std::vector<Operon::Tree> const& trees, Operon::Dataset const& d, Operon::Range r, std::string const& target, std::string const& metric) {
        std::unique_ptr<Operon::ErrorMetric> error;
        if (metric == "c2") { error = std::make_unique<Operon::C2>(); }
        else if (metric == "r2") { error = std::make_unique<Operon::R2>(); }
        else if (metric == "mse") { error = std::make_unique<Operon::MSE>(); }
        else if (metric == "rmse") { error = std::make_unique<Operon::RMSE>(); }
        else if (metric == "nmse") { error = std::make_unique<Operon::NMSE>(); }
        else if (metric == "mae") { error = std::make_unique<Operon::MAE>(); }
        else { throw std::runtime_error("Unsupported error metric"); }

        auto result = py::array_t<double>(static_cast<pybind11::ssize_t>(trees.size()));
        auto buf = result.request();
        auto values = d.GetValues(target).subspan(r.Start(), r.Size());

        // TODO: make this run in parallel with taskflow
        std::transform(trees.begin(), trees.end(), static_cast<double*>(buf.ptr), [&](auto const& t) -> double {
            auto estimated = i.operator()<Operon::Scalar>(t, d, r);
            return (*error)(estimated, values);
        });

        return result;
    }, py::arg("interpreter"), py::arg("trees"), py::arg("dataset"), py::arg("range"), py::arg("target"), py::arg("metric") = "rsquared");


    m.def("FitLeastSquares", [](py::array_t<float> lhs, py::array_t<float> rhs) -> std::pair<double, double> {
        return detail::FitLeastSquares<float>(lhs, rhs);
    });

    m.def("FitLeastSquares", [](py::array_t<double> lhs, py::array_t<double> rhs) -> std::pair<double, double> {
        return detail::FitLeastSquares<double>(lhs, rhs);
    });

    using DispatchTable = Operon::DispatchTable<Operon::Scalar, Operon::Dual>;

    // dispatch table
    py::class_<DispatchTable>(m, "DispatchTable")
        .def(py::init<>());

    // interpreter
    py::class_<Operon::Interpreter>(m, "Interpreter")
        .def(py::init<>())
        .def(py::init<DispatchTable>());

    // error metric
    py::class_<Operon::ErrorMetric>(m, "ErrorMetric")
        .def("__call__", [](Operon::ErrorMetric const& self, py::array_t<Operon::Scalar> lhs, py::array_t<Operon::Scalar> rhs) {
            return self(MakeSpan<Operon::Scalar>(lhs), MakeSpan<Operon::Scalar>(rhs)); // NOLINT
        });

    py::class_<Operon::MSE, Operon::ErrorMetric>(m, "MSE").def(py::init<>());
    py::class_<Operon::NMSE, Operon::ErrorMetric>(m, "NMSE").def(py::init<>());
    py::class_<Operon::RMSE, Operon::ErrorMetric>(m, "RMSE").def(py::init<>());
    py::class_<Operon::MAE, Operon::ErrorMetric>(m, "MAE").def(py::init<>());
    py::class_<Operon::R2, Operon::ErrorMetric>(m, "R2").def(py::init<>());
    py::class_<Operon::C2, Operon::ErrorMetric>(m, "C2").def(py::init<>());

    // evaluator
    py::class_<Operon::EvaluatorBase>(m, "EvaluatorBase")
        .def_property("LocalOptimizationIterations", &Operon::EvaluatorBase::LocalOptimizationIterations, &Operon::EvaluatorBase::SetLocalOptimizationIterations)
        .def_property("Budget", &Operon::EvaluatorBase::Budget, &Operon::EvaluatorBase::SetBudget)
        .def_property_readonly("TotalEvaluations", &Operon::EvaluatorBase::TotalEvaluations)
        // .def("__call__", &Operon::EvaluatorBase::operator())
        .def("__call__", [](Operon::EvaluatorBase const& self, Operon::RandomGenerator& rng, Operon::Individual& ind) { return self(rng, ind, {}); })
        .def_property_readonly("CallCount", [](Operon::EvaluatorBase& self) { return self.CallCount.load(); })
        .def_property_readonly("ResidualEvaluations", [](Operon::EvaluatorBase& self) { return self.ResidualEvaluations.load(); })
        .def_property_readonly("JacobianEvaluations", [](Operon::EvaluatorBase& self) { return self.JacobianEvaluations.load(); });

    py::class_<Operon::Evaluator, Operon::EvaluatorBase>(m, "Evaluator")
        .def(py::init<Operon::Problem&, Operon::Interpreter&, Operon::ErrorMetric const&, bool>());

    py::class_<Operon::UserDefinedEvaluator, Operon::EvaluatorBase>(m, "UserDefinedEvaluator")
        .def(py::init<Operon::Problem&, std::function<typename Operon::EvaluatorBase::ReturnType(Operon::RandomGenerator*, Operon::Individual&)> const&>());

    py::class_<Operon::LengthEvaluator, Operon::EvaluatorBase>(m, "LengthEvaluator")
        .def(py::init<Operon::Problem&>());

    py::class_<Operon::ShapeEvaluator, Operon::EvaluatorBase>(m, "ShapeEvaluator")
        .def(py::init<Operon::Problem&>());

    py::class_<Operon::DiversityEvaluator, Operon::EvaluatorBase>(m, "DiversityEvaluator")
        .def(py::init<Operon::Problem&>());

    py::class_<Operon::MultiEvaluator, Operon::EvaluatorBase>(m, "MultiEvaluator")
        .def(py::init<Operon::Problem&>())
        .def("Add", &Operon::MultiEvaluator::Add);

    py::class_<Operon::AggregateEvaluator, Operon::EvaluatorBase>(m, "AggregateEvaluator")
        .def(py::init<Operon::EvaluatorBase&>())
        .def_property("AggregateType", &Operon::AggregateEvaluator::GetAggregateType, &Operon::AggregateEvaluator::SetAggregateType);

    py::class_<Operon::MinimumDescriptionLengthEvaluator, Operon::Evaluator>(m, "MinimumDescriptionLengthEvaluator")
        .def(py::init<Operon::Problem&, Operon::Interpreter&>())
        .def_property("LocalOptimizationIterations",
                &Operon::MinimumDescriptionLengthEvaluator::LocalOptimizationIterations,   // getter
                &Operon::MinimumDescriptionLengthEvaluator::SetLocalOptimizationIterations<std::size_t> // setter
        );

    py::class_<Operon::BayesianInformationCriterionEvaluator, Operon::Evaluator>(m, "BayesianInformationCriterionEvaluator")
        .def(py::init<Operon::Problem&, Operon::Interpreter&>())
        .def_property("LocalOptimizationIterations",
                &Operon::BayesianInformationCriterionEvaluator::LocalOptimizationIterations,   // getter
                &Operon::BayesianInformationCriterionEvaluator::SetLocalOptimizationIterations<std::size_t> // setter
        );

    py::class_<Operon::AkaikeInformationCriterionEvaluator, Operon::Evaluator>(m, "AkaikeInformationCriterionEvaluator")
        .def(py::init<Operon::Problem&, Operon::Interpreter&>())
        .def_property("LocalOptimizationIterations",
                &Operon::AkaikeInformationCriterionEvaluator::LocalOptimizationIterations,   // getter
                &Operon::AkaikeInformationCriterionEvaluator::SetLocalOptimizationIterations<std::size_t> // setter
        );
}
