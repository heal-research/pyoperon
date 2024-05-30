// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2024 Heal Research

#include <stdexcept>
#include <operon/optimizer/optimizer.hpp>
#include <operon/optimizer/solvers/sgd.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pyoperon/pyoperon.hpp"
#include "pyoperon/optimizer.hpp"

namespace py = pybind11;

namespace detail {

template<typename T>
auto FitLeastSquares(py::array_t<T> lhs, py::array_t<T> rhs) -> std::pair<double, double>
{
    auto s1 = MakeSpan(lhs);
    auto s2 = MakeSpan(rhs);
    return Operon::FitLeastSquares(s1, s2);
}

template<typename T>
auto PoissonLikelihood(py::array_t<T> x, py::array_t<T> y) {
    return TPoissonLikelihood::ComputeLikelihood(MakeSpan(x), MakeSpan(y), {});
}

template<typename T>
auto PoissonLikelihood(py::array_t<T> x, py::array_t<T> y, py::array_t<T> w) {
    return TPoissonLikelihood::ComputeLikelihood(MakeSpan(x), MakeSpan(y), MakeSpan(w));
}

// small wrapper to unify the differently-templated MDL evaluators under a single class
class MDLEvaluator {
    std::unique_ptr<TEvaluatorBase> eval_;
    enum { Gauss, Poisson, PoissonLog } lik_ = Gauss; // TODO: very ugly, find something better

public:
    MDLEvaluator(Operon::Problem& problem, TDispatch const& dtable, std::string const& lik) {
        if (lik == "gauss") {
            eval_ = std::make_unique<TMDLEvaluatorGauss>(problem, dtable);
            lik_ = Gauss;
        } else if (lik == "poisson") {
            eval_ = std::make_unique<TMDLEvaluatorPoisson>(problem, dtable);
            lik_ = Poisson;
        } else if (lik == "poisson_log") {
            eval_ = std::make_unique<TMDLEvaluatorPoissonLog>(problem, dtable);
            lik_ = PoissonLog;
        } else {
            throw std::runtime_error(fmt::format("unknown likelihood: {}", lik));
        }
    }

    auto SetSigma(std::vector<Operon::Scalar> const& sigma) -> void {
        switch(lik_) {
            case Gauss: dynamic_cast<TMDLEvaluatorGauss*>(eval_.get())->SetSigma(sigma); break;
            case Poisson: dynamic_cast<TMDLEvaluatorPoisson*>(eval_.get())->SetSigma(sigma); break;
            case PoissonLog: dynamic_cast<TMDLEvaluatorPoissonLog*>(eval_.get())->SetSigma(sigma); break;
            default: throw std::runtime_error("unknown likelihood");
        }
    }

    auto GetSigma() -> std::span<Operon::Scalar const> {
        switch(lik_) {
            case Gauss: return dynamic_cast<TMDLEvaluatorGauss*>(eval_.get())->Sigma();
            case Poisson: return dynamic_cast<TMDLEvaluatorPoisson*>(eval_.get())->Sigma();
            case PoissonLog: return dynamic_cast<TMDLEvaluatorPoissonLog*>(eval_.get())->Sigma();
            default: throw std::runtime_error("unknown likelihood");
        }
    }

    [[nodiscard]] auto Get() const { return eval_.get(); }
};
} // namespace detail

void InitEval(py::module_ &m)
{
    // free functions
    // we use a lambda to avoid defining a fourth arg for the defaulted C++ function arg
    m.def("Evaluate", [](Operon::Tree const& t, Operon::Dataset const& d, Operon::Range r) {
        auto result = py::array_t<Operon::Scalar>(static_cast<pybind11::ssize_t>(r.Size()));
        auto span = MakeSpan(result);
        py::gil_scoped_release release;
        TDispatch dtable;
        TInterpreter{dtable, d, t}.Evaluate({}, r, span);
        py::gil_scoped_acquire acquire;
        return result;
        }, py::arg("tree"), py::arg("dataset"), py::arg("range"));

    m.def("Evaluate", [](TDispatch const& dtable, Operon::Tree const& t, Operon::Dataset const& d, Operon::Range r) {
        auto result = py::array_t<Operon::Scalar>(static_cast<pybind11::ssize_t>(r.Size()));
        auto span = MakeSpan(result);
        py::gil_scoped_release release;
        TInterpreter{dtable, d, t}.Evaluate({}, r, span);
        py::gil_scoped_acquire acquire;
        return result;
        }, py::arg("dtable"), py::arg("tree"), py::arg("dataset"), py::arg("range"));

    m.def("EvaluateTrees", [](std::vector<Operon::Tree> const& trees, Operon::Dataset const& ds, Operon::Range range, py::array_t<Operon::Scalar> result, size_t nthread) {
            auto span = MakeSpan(result);
            py::gil_scoped_release release;
            Operon::EvaluateTrees(trees, ds, range, span, nthread);
            py::gil_scoped_acquire acquire;
            }, py::arg("trees"), py::arg("dataset"), py::arg("range"), py::arg("result").noconvert(), py::arg("nthread") = 1);

    m.def("CalculateFitness", [](Operon::Tree const& t, Operon::Dataset const& d, Operon::Range r, std::string const& target, std::string const& metric) {
        auto values = d.GetValues(target).subspan(r.Start(), r.Size());

        TDispatch dtable;
        auto estimated = TInterpreter{dtable, d, t}.Evaluate({}, r);

        if (metric == "c2") { return Operon::C2{}(estimated, values); }
        if (metric == "r2") { return Operon::R2{}(estimated, values); }
        if (metric == "mse") { return Operon::MSE{}(estimated, values); }
        if (metric == "rmse") { return Operon::RMSE{}(estimated, values); }
        if (metric == "nmse") { return Operon::NMSE{}(estimated, values); }
        if (metric == "mae") { return Operon::MAE{}(estimated, values); }
        throw std::runtime_error("Invalid fitness metric");

    }, py::arg("tree"), py::arg("dataset"), py::arg("range"), py::arg("target"), py::arg("metric") = "rsquared");

    m.def("PoissonLikelihood", [](py::array_t<float> x, py::array_t<float> y){
        return detail::PoissonLikelihood(std::move(x), std::move(y));
    });

    m.def("PoissonLikelihood", [](py::array_t<float> x, py::array_t<float> y, py::array_t<float> w){
        return detail::PoissonLikelihood(std::move(x), std::move(y), std::move(w));
    });

    m.def("CalculateFitness", [](std::vector<Operon::Tree> const& trees, Operon::Dataset const& d, Operon::Range r, std::string const& target, std::string const& metric) {
        std::unique_ptr<Operon::ErrorMetric> error;
        if (metric == "c2") { error = std::make_unique<Operon::C2>(); }
        else if (metric == "r2") { error = std::make_unique<Operon::R2>(); }
        else if (metric == "mse") { error = std::make_unique<Operon::MSE>(); }
        else if (metric == "rmse") { error = std::make_unique<Operon::RMSE>(); }
        else if (metric == "nmse") { error = std::make_unique<Operon::NMSE>(); }
        else if (metric == "mae") { error = std::make_unique<Operon::MAE>(); }
        else { throw std::runtime_error("Unsupported error metric"); }

        TDispatch dtable;

        auto result = py::array_t<double>(static_cast<pybind11::ssize_t>(trees.size()));
        auto buf = result.request();
        auto values = d.GetValues(target).subspan(r.Start(), r.Size());

        // TODO: make this run in parallel with taskflow
        std::transform(trees.begin(), trees.end(), static_cast<double*>(buf.ptr), [&](auto const& t) -> double {
            auto estimated = TInterpreter{dtable, d, t}.Evaluate({}, r);
            return (*error)(estimated, values);
        });

        return result;
    }, py::arg("trees"), py::arg("dataset"), py::arg("range"), py::arg("target"), py::arg("metric") = "rsquared");


    m.def("FitLeastSquares", [](py::array_t<float> lhs, py::array_t<float> rhs) -> std::pair<double, double> {
        return detail::FitLeastSquares<float>(lhs, rhs);
    });

    m.def("FitLeastSquares", [](py::array_t<double> lhs, py::array_t<double> rhs) -> std::pair<double, double> {
        return detail::FitLeastSquares<double>(lhs, rhs);
    });

    // dispatch table
    py::class_<TDispatch>(m, "DispatchTable")
        .def(py::init<>());

    py::class_<TInterpreterBase>(m, "InterpreterBase");

    // interpreter
    py::class_<TInterpreter, TInterpreterBase>(m, "Interpreter")
        .def(py::init<TDispatch const&, Operon::Dataset const&, Operon::Tree const&>())
        .def("Evaluate", [](TInterpreter const& self, Operon::Range range){
            return self.Evaluate({}, range);
        });

    // error metric
    py::class_<Operon::ErrorMetric>(m, "ErrorMetric")
        .def("__call__", [](Operon::ErrorMetric const& self, py::array_t<Operon::Scalar> lhs, py::array_t<Operon::Scalar> rhs) {
            return self(MakeSpan<Operon::Scalar>(lhs), MakeSpan<Operon::Scalar>(rhs)); // NOLINT
        });

    py::class_<Operon::SSE, Operon::ErrorMetric>(m, "SSE").def(py::init<>());
    py::class_<Operon::MSE, Operon::ErrorMetric>(m, "MSE").def(py::init<>());
    py::class_<Operon::NMSE, Operon::ErrorMetric>(m, "NMSE").def(py::init<>());
    py::class_<Operon::RMSE, Operon::ErrorMetric>(m, "RMSE").def(py::init<>());
    py::class_<Operon::MAE, Operon::ErrorMetric>(m, "MAE").def(py::init<>());
    py::class_<Operon::R2, Operon::ErrorMetric>(m, "R2").def(py::init<>());
    py::class_<Operon::C2, Operon::ErrorMetric>(m, "C2").def(py::init<>());

    // evaluator
    py::class_<TEvaluatorBase>(m, "EvaluatorBase")
        .def_property("Budget", &TEvaluatorBase::Budget, &TEvaluatorBase::SetBudget)
        .def_property_readonly("TotalEvaluations", &TEvaluatorBase::TotalEvaluations)
        //.def("__call__", &TEvaluatorBase::operator())
        .def("__call__", [](TEvaluatorBase const& self, Operon::RandomGenerator& rng, Operon::Individual& ind) { return self(rng, ind, {}); })
        .def_property_readonly("CallCount", [](TEvaluatorBase& self) { return self.CallCount.load(); })
        .def_property_readonly("ResidualEvaluations", [](TEvaluatorBase& self) { return self.ResidualEvaluations.load(); })
        .def_property_readonly("JacobianEvaluations", [](TEvaluatorBase& self) { return self.JacobianEvaluations.load(); });

    py::class_<TEvaluator, TEvaluatorBase>(m, "Evaluator")
        .def(py::init<Operon::Problem&, TDispatch const&, Operon::ErrorMetric const&, bool>());

    py::class_<Operon::UserDefinedEvaluator, TEvaluatorBase>(m, "UserDefinedEvaluator")
        .def(py::init<Operon::Problem&, std::function<typename TEvaluatorBase::ReturnType(Operon::RandomGenerator*, Operon::Individual&)> const&>())
        .def("__call__", [](TEvaluatorBase const& self, Operon::RandomGenerator& rng, Operon::Individual& ind) {
            py::gil_scoped_release release;
            return self(rng, ind, {});
            py::gil_scoped_acquire acquire;
        });

    py::class_<Operon::LengthEvaluator, TEvaluatorBase>(m, "LengthEvaluator")
        .def(py::init<Operon::Problem&>());

    py::class_<Operon::ShapeEvaluator, TEvaluatorBase>(m, "ShapeEvaluator")
        .def(py::init<Operon::Problem&>());

    py::class_<Operon::DiversityEvaluator, TEvaluatorBase>(m, "DiversityEvaluator")
        .def(py::init<Operon::Problem&>());

    py::class_<Operon::MultiEvaluator, TEvaluatorBase>(m, "MultiEvaluator")
        .def(py::init<Operon::Problem&>())
        .def("Add", &Operon::MultiEvaluator::Add);

    py::enum_<Operon::AggregateEvaluator::AggregateType>(m, "AggregateType")
        .value("Min", Operon::AggregateEvaluator::AggregateType::Min)
        .value("Max", Operon::AggregateEvaluator::AggregateType::Max)
        .value("Median", Operon::AggregateEvaluator::AggregateType::Median)
        .value("Mean", Operon::AggregateEvaluator::AggregateType::Mean)
        .value("HarmonicMean", Operon::AggregateEvaluator::AggregateType::HarmonicMean)
        .value("Sum", Operon::AggregateEvaluator::AggregateType::Sum);

    py::class_<Operon::AggregateEvaluator, TEvaluatorBase>(m, "AggregateEvaluator")
        .def(py::init<TEvaluatorBase&>())
        .def_property("AggregateType", &Operon::AggregateEvaluator::GetAggregateType, &Operon::AggregateEvaluator::SetAggregateType);

    py::class_<detail::MDLEvaluator>(m, "MinimumDescriptionLengthEvaluator")
        .def(py::init<Operon::Problem&, TDispatch const&, std::string const&>())
        .def("__call__", [](detail::MDLEvaluator const& self, Operon::RandomGenerator& rng, Operon::Individual& ind) {
            return (*self.Get())(rng, ind, std::span<Operon::Scalar>{});
        })
        .def_property("Sigma", nullptr /*get*/ , &detail::MDLEvaluator::SetSigma /*set*/);

    py::class_<TBICEvaluator, TEvaluator>(m, "BayesianInformationCriterionEvaluator")
        .def(py::init<Operon::Problem&, TDispatch const&>());

    py::class_<TAIKEvaluator, TEvaluator>(m, "AkaikeInformationCriterionEvaluator")
        .def(py::init<Operon::Problem&, TDispatch const&>());

    py::class_<TGaussEvaluator, TEvaluator>(m, "GaussianLikelihoodEvaluator")
        .def(py::init<Operon::Problem&, TDispatch const&>())
        .def_property("Sigma", &TGaussEvaluator::Sigma , &TGaussEvaluator::SetSigma /*set*/);;

    py::class_<TPoissonEvaluator, TEvaluator>(m, "PoissonLikelihoodEvaluator")
        .def(py::init<Operon::Problem&, TDispatch const&>())
        .def_property("Sigma", [](TPoissonEvaluator const& self) {
            auto sigma = self.Sigma();
            return std::vector<Operon::Scalar>(sigma.begin(), sigma.end());
        }, &TPoissonEvaluator::SetSigma /*set*/);
}