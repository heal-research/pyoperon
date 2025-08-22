// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2024 Heal Research

#include <stdexcept>
#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>

#include <operon/optimizer/optimizer.hpp>
#include <operon/optimizer/solvers/sgd.hpp>

#include "pyoperon/pyoperon.hpp"
#include "pyoperon/optimizer.hpp"

namespace nb = nanobind;

namespace detail {

template<typename T>
auto FitLeastSquares(nb::ndarray<T> lhs, nb::ndarray<T> rhs) -> std::pair<double, double>
{
    auto s1 = MakeSpan(lhs);
    auto s2 = MakeSpan(rhs);
    return Operon::FitLeastSquares(s1, s2);
}

template<typename T>
auto PoissonLikelihood(nb::ndarray<T> x, nb::ndarray<T> y) {
    return TPoissonLikelihood::ComputeLikelihood(MakeSpan(x), MakeSpan(y), {});
}

template<typename T>
auto PoissonLikelihood(nb::ndarray<T> x, nb::ndarray<T> y, nb::ndarray<T> w) {
    return TPoissonLikelihood::ComputeLikelihood(MakeSpan(x), MakeSpan(y), MakeSpan(w));
}

// small wrapper to unify the differently-templated MDL evaluators under a single class
class MDLEvaluator {
    std::unique_ptr<TEvaluatorBase> eval_;
    enum { Gauss, Poisson, PoissonLog } lik_ = Gauss; // TODO: very ugly, find something better

public:
    MDLEvaluator(Operon::Problem const& problem, TDispatch const& dtable, std::string const& lik) {
        if (lik == "gauss") {
            eval_ = std::make_unique<TMDLEvaluatorGauss>(&problem, &dtable);
            lik_ = Gauss;
        } else if (lik == "poisson") {
            eval_ = std::make_unique<TMDLEvaluatorPoisson>(&problem, &dtable);
            lik_ = Poisson;
        } else if (lik == "poisson_log") {
            eval_ = std::make_unique<TMDLEvaluatorPoissonLog>(&problem, &dtable);
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

void InitEval(nb::module_ &m)
{
    // free functions
    // we use a lambda to avoid defining a fourth arg for the defaulted C++ function arg
    m.def("Evaluate", [](Operon::Tree const& t, Operon::Dataset const& d, Operon::Range r) {
        auto* data = new Operon::Scalar[r.Size()];
        nb::capsule owner(data, [](void* p) noexcept { delete[] (Operon::Scalar*)p; });
        Operon::Span<Operon::Scalar> span{data, r.Size()};
        nb::gil_scoped_release release;
        TDispatch dtable;
        TInterpreter{&dtable, &d, &t}.Evaluate({}, r, span);
        nb::gil_scoped_acquire acquire;
        std::array shape{r.Size()};
        return nb::ndarray<nb::numpy, Operon::Scalar, nb::ndim<1>>(data, 1, shape.data(), owner);
    }, nb::arg("tree"), nb::arg("dataset"), nb::arg("range"));

    m.def("Evaluate", [](TDispatch const& dtable, Operon::Tree const& t, Operon::Dataset const& d, Operon::Range r) {
        auto* data = new Operon::Scalar[r.Size()];
        nb::capsule owner(data, [](void* p) noexcept { delete[] (Operon::Scalar*)p; });
        Operon::Span<Operon::Scalar> span{data, r.Size()};
        nb::gil_scoped_release release;
        TInterpreter{&dtable, &d, &t}.Evaluate({}, r, span);
        nb::gil_scoped_acquire acquire;
        std::array shape{r.Size()};
        return nb::ndarray<nb::numpy, Operon::Scalar, nb::ndim<1>>(data, 1, shape.data(), owner);
    }, nb::arg("dtable"), nb::arg("tree"), nb::arg("dataset"), nb::arg("range"));

    m.def("EvaluateTrees", [](std::vector<Operon::Tree> const& trees, Operon::Dataset const& ds, Operon::Range range, nb::ndarray<Operon::Scalar> result, size_t nthread) {
        auto span = MakeSpan(result);
        nb::gil_scoped_release release;
        Operon::EvaluateTrees(trees, &ds, range, span, nthread);
        nb::gil_scoped_acquire acquire;
    }, nb::arg("trees"), nb::arg("dataset"), nb::arg("range"), nb::arg("result").noconvert(), nb::arg("nthread") = 1);

    m.def("CalculateFitness", [](Operon::Tree const& t, Operon::Dataset const& d, Operon::Range r, std::string const& target, std::string const& metric) {
        auto values = d.GetValues(target).subspan(r.Start(), r.Size());

        TDispatch dtable;
        auto estimated = TInterpreter{&dtable, &d, &t}.Evaluate({}, r);

        if (metric == "c2") { return Operon::C2{}(estimated, values); }
        if (metric == "r2") { return Operon::R2{}(estimated, values); }
        if (metric == "mse") { return Operon::MSE{}(estimated, values); }
        if (metric == "rmse") { return Operon::RMSE{}(estimated, values); }
        if (metric == "nmse") { return Operon::NMSE{}(estimated, values); }
        if (metric == "mae") { return Operon::MAE{}(estimated, values); }
        throw std::runtime_error("Invalid fitness metric");

    }, nb::arg("tree"), nb::arg("dataset"), nb::arg("range"), nb::arg("target"), nb::arg("metric") = "rsquared");

    m.def("PoissonLikelihood", [](nb::ndarray<Operon::Scalar> x, nb::ndarray<Operon::Scalar> y){
        return detail::PoissonLikelihood(std::move(x), std::move(y));
    });

    m.def("PoissonLikelihood", [](nb::ndarray<Operon::Scalar> x, nb::ndarray<Operon::Scalar> y, nb::ndarray<Operon::Scalar> w){
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

        auto* data = new Operon::Scalar[r.Size()];
        nb::capsule owner(data, [](void* p) noexcept { delete[] (Operon::Scalar*)p; });
        Operon::Span<Operon::Scalar> span{data, r.Size()};
        auto values = d.GetValues(target).subspan(r.Start(), r.Size());

        // TODO: make this run in parallel with taskflow
        std::transform(trees.begin(), trees.end(), data, [&](auto const& t) -> double {
            auto estimated = TInterpreter{&dtable, &d, &t}.Evaluate({}, r);
            return (*error)(estimated, values);
        });

        return nb::ndarray<nb::numpy, Operon::Scalar>(data, {-1UL}, owner);
    }, nb::arg("trees"), nb::arg("dataset"), nb::arg("range"), nb::arg("target"), nb::arg("metric") = "rsquared");


    m.def("FitLeastSquares", [](nb::ndarray<float> lhs, nb::ndarray<float> rhs) -> std::pair<double, double> {
        return detail::FitLeastSquares<float>(lhs, rhs);
    });

    m.def("FitLeastSquares", [](nb::ndarray<double> lhs, nb::ndarray<double> rhs) -> std::pair<double, double> {
        return detail::FitLeastSquares<double>(lhs, rhs);
    });

    // dispatch table
    nb::class_<TDispatch>(m, "DispatchTable")
        .def(nb::init<>());

    nb::class_<TInterpreterBase>(m, "InterpreterBase");

    // interpreter
    nb::class_<TInterpreter, TInterpreterBase>(m, "Interpreter")
        .def("Evaluate", [](TInterpreter const& self, Operon::Range range){
            return self.Evaluate({}, range);
        });

    // error metric
    nb::class_<Operon::ErrorMetric>(m, "ErrorMetric")
        .def("__call__", [](Operon::ErrorMetric const& self, nb::ndarray<Operon::Scalar> lhs, nb::ndarray<Operon::Scalar> rhs) {
            return self(MakeSpan<Operon::Scalar>(lhs), MakeSpan<Operon::Scalar>(rhs)); // NOLINT
        });

    nb::class_<Operon::SSE, Operon::ErrorMetric>(m, "SSE").def(nb::init<>());
    nb::class_<Operon::MSE, Operon::ErrorMetric>(m, "MSE").def(nb::init<>());
    nb::class_<Operon::NMSE, Operon::ErrorMetric>(m, "NMSE").def(nb::init<>());
    nb::class_<Operon::RMSE, Operon::ErrorMetric>(m, "RMSE").def(nb::init<>());
    nb::class_<Operon::MAE, Operon::ErrorMetric>(m, "MAE").def(nb::init<>());
    nb::class_<Operon::R2, Operon::ErrorMetric>(m, "R2").def(nb::init<>());
    nb::class_<Operon::C2, Operon::ErrorMetric>(m, "C2").def(nb::init<>());

    // evaluator
    nb::class_<TEvaluatorBase>(m, "EvaluatorBase")
        .def_prop_rw("Budget", &TEvaluatorBase::Budget, &TEvaluatorBase::SetBudget)
        .def_prop_ro("TotalEvaluations", &TEvaluatorBase::TotalEvaluations)
        .def("__call__", [](TEvaluatorBase const& self, Operon::RandomGenerator& rng, Operon::Individual const& ind) { return self(rng, ind); })
        .def_prop_ro("CallCount", [](TEvaluatorBase& self) { return self.CallCount.load(); })
        .def_prop_ro("ResidualEvaluations", [](TEvaluatorBase& self) { return self.ResidualEvaluations.load(); })
        .def_prop_ro("JacobianEvaluations", [](TEvaluatorBase& self) { return self.JacobianEvaluations.load(); });

    nb::class_<TEvaluator, TEvaluatorBase>(m, "Evaluator")
        .def(nb::init<Operon::Problem const*, TDispatch const*, Operon::ErrorMetric, bool, std::vector<Operon::Scalar>>());

    nb::class_<Operon::UserDefinedEvaluator, TEvaluatorBase>(m, "UserDefinedEvaluator")
        .def(nb::init<Operon::Problem const*, std::function<typename TEvaluatorBase::ReturnType(Operon::RandomGenerator*, Operon::Individual const&)> const&>())
        .def("__call__", [](TEvaluatorBase const& self, Operon::RandomGenerator& rng, Operon::Individual& ind) {
            nb::gil_scoped_release release;
            return self(rng, ind, {});
            nb::gil_scoped_acquire acquire;
        });

    nb::class_<Operon::LengthEvaluator, TEvaluatorBase>(m, "LengthEvaluator")
        .def(nb::init<Operon::Problem const*>());

    nb::class_<Operon::ShapeEvaluator, TEvaluatorBase>(m, "ShapeEvaluator")
        .def(nb::init<Operon::Problem const*>());

    nb::class_<Operon::DiversityEvaluator, TEvaluatorBase>(m, "DiversityEvaluator")
        .def(nb::init<Operon::Problem const*>());

    nb::class_<Operon::MultiEvaluator, TEvaluatorBase>(m, "MultiEvaluator")
        .def(nb::init<Operon::Problem const*>())
        .def("Add", &Operon::MultiEvaluator::Add);

    nb::enum_<Operon::AggregateEvaluator::AggregateType>(m, "AggregateType")
        .value("Min", Operon::AggregateEvaluator::AggregateType::Min)
        .value("Max", Operon::AggregateEvaluator::AggregateType::Max)
        .value("Median", Operon::AggregateEvaluator::AggregateType::Median)
        .value("Mean", Operon::AggregateEvaluator::AggregateType::Mean)
        .value("HarmonicMean", Operon::AggregateEvaluator::AggregateType::HarmonicMean)
        .value("Sum", Operon::AggregateEvaluator::AggregateType::Sum);

    nb::class_<Operon::AggregateEvaluator, TEvaluatorBase>(m, "AggregateEvaluator")
        .def(nb::init<TEvaluatorBase const*>())
        .def_prop_rw("AggregateType", &Operon::AggregateEvaluator::GetAggregateType, &Operon::AggregateEvaluator::SetAggregateType);

    nb::class_<detail::MDLEvaluator>(m, "MinimumDescriptionLengthEvaluator")
        .def(nb::init<Operon::Problem const&, TDispatch const&, std::string const&>())
        .def("__call__", [](detail::MDLEvaluator const& self, Operon::RandomGenerator& rng, Operon::Individual const& ind) {
            return (*self.Get())(rng, ind);
        })
        .def_prop_rw("Sigma", nullptr /*get*/ , &detail::MDLEvaluator::SetSigma /*set*/);

    nb::class_<TBICEvaluator, TEvaluator>(m, "BayesianInformationCriterionEvaluator")
        .def(nb::init<Operon::Problem const*, TDispatch const*>());

    nb::class_<TAIKEvaluator, TEvaluator>(m, "AkaikeInformationCriterionEvaluator")
        .def(nb::init<Operon::Problem const*, TDispatch const*>());

    nb::class_<TGaussEvaluator, TEvaluator>(m, "GaussianLikelihoodEvaluator")
        .def(nb::init<Operon::Problem const*, TDispatch const*>())
        .def_prop_rw("Sigma", &TGaussEvaluator::Sigma , &TGaussEvaluator::SetSigma /*set*/);;

    nb::class_<TPoissonEvaluator, TEvaluator>(m, "PoissonLikelihoodEvaluator")
        .def(nb::init<Operon::Problem const*, TDispatch const*>())
        .def_prop_rw("Sigma", [](TPoissonEvaluator const& self) {
            auto sigma = self.Sigma();
            return std::vector<Operon::Scalar>(sigma.begin(), sigma.end());
        }, &TPoissonEvaluator::SetSigma /*set*/);
}