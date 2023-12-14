// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include <stdexcept>
#include <operon/optimizer/optimizer.hpp>
#include <operon/optimizer/solvers/sgd.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <operon/operators/evaluator.hpp>
#include "pyoperon/pyoperon.hpp"

namespace py = pybind11;

// type aliases for convenience
using TDispatch                = Operon::DefaultDispatch;
using TInterpreter             = Operon::Interpreter<Operon::Scalar, TDispatch>;
using TInterpreterBase         = Operon::InterpreterBase<Operon::Scalar>;

using TEvaluatorBase           = Operon::EvaluatorBase;
using TEvaluator               = Operon::Evaluator<TDispatch>;
using TMDLEvaluator            = Operon::MinimumDescriptionLengthEvaluator<TDispatch>;
using TBICEvaluator            = Operon::BayesianInformationCriterionEvaluator<TDispatch>;
using TAIKEvaluator            = Operon::AkaikeInformationCriterionEvaluator<TDispatch>;

// likelihood
using TGaussianLikelihood      = Operon::GaussianLikelihood<Operon::Scalar>;
using TPoissonLikelihood       = Operon::PoissonLikelihood<Operon::Scalar, false>;
using TPoissonLikelihoodLog    = Operon::PoissonLikelihood<Operon::Scalar, true>;

// update rule
using TUpdateRule              = Operon::UpdateRule::LearningRateUpdateRule;
using TConstantUpdateRule      = Operon::UpdateRule::Constant<Operon::Scalar>;
using TMomentumUpdateRule      = Operon::UpdateRule::Momentum<Operon::Scalar>;
using TRmsPropUpdateRule       = Operon::UpdateRule::RmsProp<Operon::Scalar>;
using TAdaDeltaUpdateRule      = Operon::UpdateRule::AdaDelta<Operon::Scalar>;
using TAdaMaxUpdateRule        = Operon::UpdateRule::AdaMax<Operon::Scalar>;
using TAdamUpdateRule          = Operon::UpdateRule::Adam<Operon::Scalar>;
using TYamAdamUpdateRule       = Operon::UpdateRule::YamAdam<Operon::Scalar>;
using TAmsGradUpdateRule       = Operon::UpdateRule::AmsGrad<Operon::Scalar>;
using TYogiUpdateRule          = Operon::UpdateRule::Yogi<Operon::Scalar>;

// optimizer
using TOptimizerBase           = Operon::OptimizerBase<TDispatch>;
using TLMOptimizerTiny         = Operon::LevenbergMarquardtOptimizer<TDispatch, Operon::OptimizerType::Tiny>;
using TLMOptimizerEigen        = Operon::LevenbergMarquardtOptimizer<TDispatch, Operon::OptimizerType::Eigen>;
using TLMOptimizerCeres        = Operon::LevenbergMarquardtOptimizer<TDispatch, Operon::OptimizerType::Ceres>;

using TLBGSOptimizerGauss      = Operon::LBFGSOptimizer<TDispatch, TGaussianLikelihood>;
using TLBGSOptimizerPoisson    = Operon::LBFGSOptimizer<TDispatch, TPoissonLikelihood>;
using TLBGSOptimizerPoissonLog = Operon::LBFGSOptimizer<TDispatch, TPoissonLikelihoodLog>;

using TSGDOptimizerGauss       = Operon::SGDOptimizer<TDispatch, TGaussianLikelihood>;
using TSGDOptimizerPoisson     = Operon::SGDOptimizer<TDispatch, TPoissonLikelihood>;
using TSGDOptimizerPoissonLog  = Operon::SGDOptimizer<TDispatch, TPoissonLikelihoodLog>;

namespace detail {
template<typename T>
auto FitLeastSquares(py::array_t<T> lhs, py::array_t<T> rhs) -> std::pair<double, double>
{
    auto s1 = MakeSpan(lhs);
    auto s2 = MakeSpan(rhs);
    return Operon::FitLeastSquares(s1, s2);
}

class Optimizer {
    std::unique_ptr<TOptimizerBase> optimizer_;

public:
        // Optimizer(TDispatch const& dtable, Operon::Problem const& problem, std::string const& optName, std::string const& likName, std::string const& updName, std::size_t iter, std::size_t bsize) {
        //     if (optName == "lm") {
        //         optimizer_ = std::make_unique<TLMOptimizerEigen>(dtable, problem);
        //     } else if (optName == "lbfgs") {
        //         if (likName == "gaussian") {
        //             optimizer_ = std::make_unique<TLBGSOptimizerGauss>(dtable, problem);
        //         } else if (likName == "poisson") {
        //             optimizer_ = std::make_unique<TLBGSOptimizerPoisson>(dtable, problem);
        //         } else if (likName == "poisson_log") {
        //             optimizer_ = std::make_unique<TLBGSOptimizerPoissonLog>(dtable, problem);
        //         }
        //     } else if (optName == "sgd") {
        //         if (likName == "gaussian") {
        //             // optimizer_ = std::make_unique<
        //         }
        //     }
        //     optimizer_->SetIterations(iter);
        //     optimizer_->SetBatchSize(bsize);
        // }

        auto SetBatchSize(std::size_t value) const { optimizer_->SetBatchSize(value); }
        [[nodiscard]] auto BatchSize() const { return optimizer_->BatchSize(); }

        auto SetIterations(std::size_t value) const { optimizer_->SetIterations(value); }
        [[nodiscard]] auto Iterations() const { return optimizer_->Iterations(); }

        [[nodiscard]] auto GetDispatchTable() const { return optimizer_->GetDispatchTable(); }
        [[nodiscard]] auto GetProblem() const { return optimizer_->GetProblem(); }

        [[nodiscard]] auto Optimize(Operon::RandomGenerator& rng, Operon::Tree const& tree) const {
            return optimizer_->Optimize(rng, tree);
        }

        [[nodiscard]] auto ComputeLikelihood(Operon::Span<Operon::Scalar const> x, Operon::Span<Operon::Scalar const> y, Operon::Span<Operon::Scalar const> w) const {
            return optimizer_->ComputeLikelihood(x, y, w);
        }

        [[nodiscard]] auto ComputeFisherMatrix(Operon::Span<Operon::Scalar const> pred, Operon::Span<Operon::Scalar const> jac, Operon::Span<Operon::Scalar const> sigma) const {
            return optimizer_->ComputeFisherMatrix(pred, jac, sigma);
        }

        auto Set(std::unique_ptr<TOptimizerBase> optimizer) {
            optimizer_ = std::move(optimizer);
        }

        [[nodiscard]] auto Get() const { return optimizer_.get(); }
};

class LMOptimizer : public Optimizer {
    public:
    LMOptimizer(TDispatch const& dispatch, Operon::Problem const& problem, std::size_t maxIter, std::size_t batchSize)
    {
        Optimizer::Set(std::make_unique<TLMOptimizerEigen>(dispatch, problem));
        auto const* opt = Optimizer::Get();
        opt->SetIterations(maxIter);
        opt->SetBatchSize(batchSize);
    }
};

class LBFGSOptimizer : public Optimizer {
public:
    LBFGSOptimizer(TDispatch const& dispatch, Operon::Problem const& problem, std::string const& likelihood, std::size_t maxIter, std::size_t batchSize)
    {
        if (likelihood == "gaussian") {
            Optimizer::Set(std::make_unique<TLBGSOptimizerGauss>(dispatch, problem));
        } else if (likelihood == "poisson") {
            Optimizer::Set(std::make_unique<TLBGSOptimizerPoisson>(dispatch, problem));
        } else if (likelihood == "poisson_log") {
            Optimizer::Set(std::make_unique<TLBGSOptimizerPoissonLog>(dispatch, problem));
        } else {
            throw std::invalid_argument(fmt::format("{} is not a valid likelihood\n", likelihood));
        }

        auto const* opt = Optimizer::Get();
        opt->SetIterations(maxIter);
        opt->SetBatchSize(batchSize);
    }
};

class SGDOptimizer : public Optimizer {
    std::unique_ptr<TUpdateRule> rule_;

public:
    SGDOptimizer(TDispatch const& dispatch, Operon::Problem const& problem, Operon::UpdateRule::LearningRateUpdateRule const& updateRule, std::string const& likelihood, std::size_t maxIter, std::size_t batchSize)
    {
        if (likelihood == "gaussian") {
            Optimizer::Set(std::make_unique<TSGDOptimizerGauss>(dispatch, problem, updateRule));
        } else if (likelihood == "poisson") {
            Optimizer::Set(std::make_unique<TSGDOptimizerPoisson>(dispatch, problem, updateRule));
        } else if (likelihood == "poisson_log") {
            Optimizer::Set(std::make_unique<TSGDOptimizerPoissonLog>(dispatch, problem, updateRule));
        } else {
            throw std::invalid_argument(fmt::format("{} is not a valid likelihood\n", likelihood));
        }

        auto const* opt = Optimizer::Get();
        opt->SetIterations(maxIter);
        opt->SetBatchSize(batchSize);
    }
};
} // namespace detail

void InitOptimizer(py::module_ &m)
{
    using detail::Optimizer;
    using Operon::Problem;
    using Operon::UpdateRule::LearningRateUpdateRule;
    using std::string;
    using std::size_t;

    py::class_<Operon::OptimizerSummary>(m, "OptimizerSummary")
        .def_readwrite("InitialCost", &Operon::OptimizerSummary::InitialCost)
        .def_readwrite("FinalCost", &Operon::OptimizerSummary::FinalCost)
        .def_readwrite("Iterations", &Operon::OptimizerSummary::Iterations)
        .def_readwrite("FunctionEvaluations", &Operon::OptimizerSummary::FunctionEvaluations)
        .def_readwrite("JacobianEvaluations", &Operon::OptimizerSummary::JacobianEvaluations)
        .def_readwrite("Success", &Operon::OptimizerSummary::Success)
        .def_readwrite("InitialParameters", &Operon::OptimizerSummary::InitialParameters)
        .def_readwrite("FinalParameters", &Operon::OptimizerSummary::FinalParameters);

    py::class_<detail::Optimizer>(m, "Optimizer")
        // .def(py::init<TDispatch const&, Problem const&, string const&, string const, size_t, size_t, bool>()
        //         , py::arg("dtable")
        //         , py::arg("problem")
        //         , py::arg("optimizer") = "lbfgs"
        //         , py::arg("likelihood") = "gaussian"
        //         , py::arg("iterations") = 10
        //         , py::arg("batchsize") = TDispatch::BatchSize<Operon::Scalar>
        //         , py::arg("loginput") = false)
        .def_property("BatchSize", &Optimizer::SetBatchSize, &Optimizer::BatchSize)
        .def_property("Iterations", &Optimizer::SetIterations, &Optimizer::Iterations)
        .def_property_readonly("DispatchTable", &Optimizer::GetDispatchTable)
        .def_property_readonly("Problem", &Optimizer::GetProblem)
        .def_property_readonly("OptimizerImpl", &Optimizer::Get)
        .def("Optimize", &Optimizer::Optimize)
        .def("ComputeLikelihood", &Optimizer::ComputeLikelihood)
        .def("ComputeFisherMatrix", &Optimizer::ComputeFisherMatrix);

    py::class_<detail::LMOptimizer, detail::Optimizer>(m, "LMOptimizer")
        .def(py::init<TDispatch const&, Problem const&, std::size_t, std::size_t>()
            , py::arg("dtable")
            , py::arg("problem")
            , py::arg("max_iter") = 10
            , py::arg("batch_size") = TDispatch::BatchSize<Operon::Scalar>
        );

    py::class_<detail::LBFGSOptimizer, detail::Optimizer>(m, "LBFGSOptimizer")
        .def(py::init<TDispatch const&, Problem const&, std::string const&, std::size_t, std::size_t>()
            , py::arg("dtable")
            , py::arg("problem")
            , py::arg("likelihood") = "gaussian"
            , py::arg("max_iter") = 10
            , py::arg("batch_size") = TDispatch::BatchSize<Operon::Scalar>
        );

    py::class_<detail::SGDOptimizer, detail::Optimizer>(m, "SGDOptimizer")
        .def(py::init<TDispatch const&, Problem const&, LearningRateUpdateRule const&, std::string const&, std::size_t, std::size_t>()
            , py::arg("dtable")
            , py::arg("problem")
            , py::arg("update_rule")
            , py::arg("likelihood") = "gaussian"
            , py::arg("max_iter") = 10
            , py::arg("batch_size") = TDispatch::BatchSize<Operon::Scalar>
        );

    py::class_<TUpdateRule>(m, "UpdateRule"); // base class

    py::class_<TConstantUpdateRule, TUpdateRule>(m, "ConstantUpdateRule")
        // .def(py::init([](Operon::Scalar lr) {
        //     return TConstantUpdateRule(0, lr);
        // }), py::arg("learning_rate") = 0.01);
        .def(py::init<Eigen::Index, Operon::Scalar>()
            , py::arg("dimension") = 0
            , py::arg("learning_rate") = 0.01
        );

    py::class_<TMomentumUpdateRule, TUpdateRule>(m, "MomentumUpdateRule")
        .def(py::init<Eigen::Index, Operon::Scalar, Operon::Scalar>()
            , py::arg("dimension") = 0
            , py::arg("learning_rate") = 0.01
            , py::arg("beta") = 0.9
        );

    py::class_<TRmsPropUpdateRule, TUpdateRule>(m, "RmsPropUpdateRule")
        .def(py::init<Eigen::Index, Operon::Scalar, Operon::Scalar, Operon::Scalar>()
            , py::arg("dimension") = 0
            , py::arg("learning_rate") = 0.01
            , py::arg("beta") = 0.9
            , py::arg("eps") = 1e-6
        );

    py::class_<TAdaMaxUpdateRule, TUpdateRule>(m, "AdaDeltaUpdateRule")
        .def(py::init<Eigen::Index, Operon::Scalar, Operon::Scalar, Operon::Scalar>()
            , py::arg("dimension") = 0
            , py::arg("learning_rate") = 0.01
            , py::arg("beta1") = 0.9
            , py::arg("beta2") = 0.999
        );

    py::class_<TAmsGradUpdateRule, TUpdateRule>(m, "AmsGradUpdateRule")
        .def(py::init<Eigen::Index, Operon::Scalar, Operon::Scalar, Operon::Scalar, Operon::Scalar>()
            , py::arg("dimension") = 0
            , py::arg("learning_rate") = 0.01
            , py::arg("epsilon")
            , py::arg("beta1") = 0.9
            , py::arg("beta2") = 0.999
        );
}

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

    py::class_<Operon::MSE, Operon::ErrorMetric>(m, "MSE").def(py::init<>());
    py::class_<Operon::NMSE, Operon::ErrorMetric>(m, "NMSE").def(py::init<>());
    py::class_<Operon::RMSE, Operon::ErrorMetric>(m, "RMSE").def(py::init<>());
    py::class_<Operon::MAE, Operon::ErrorMetric>(m, "MAE").def(py::init<>());
    py::class_<Operon::R2, Operon::ErrorMetric>(m, "R2").def(py::init<>());
    py::class_<Operon::C2, Operon::ErrorMetric>(m, "C2").def(py::init<>());

    // evaluator
    py::class_<TEvaluatorBase>(m, "EvaluatorBase")
        .def_property("Budget", &TEvaluatorBase::Budget, &Operon::EvaluatorBase::SetBudget)
        .def_property_readonly("TotalEvaluations", &TEvaluatorBase::TotalEvaluations)
        //.def("__call__", &TEvaluatorBase::operator())
        .def("__call__", [](Operon::EvaluatorBase const& self, Operon::RandomGenerator& rng, Operon::Individual& ind) { return self(rng, ind, {}); })
        .def_property_readonly("CallCount", [](TEvaluatorBase& self) { return self.CallCount.load(); })
        .def_property_readonly("ResidualEvaluations", [](TEvaluatorBase& self) { return self.ResidualEvaluations.load(); })
        .def_property_readonly("JacobianEvaluations", [](TEvaluatorBase& self) { return self.JacobianEvaluations.load(); });

    py::class_<TEvaluator, Operon::EvaluatorBase>(m, "Evaluator")
        .def(py::init<Operon::Problem&, TDispatch const&, Operon::ErrorMetric const&, bool>())
        //.def_property("Optimizer", &TEvaluator::GetOptimizer, &TEvaluator::SetOptimizer);
        .def_property("Optimizer", nullptr, [](TEvaluator& self, detail::Optimizer const& opt) {
           self.SetOptimizer(opt.Get());
        });

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

    py::enum_<Operon::AggregateEvaluator::AggregateType>(m, "AggregateType")
        .value("Min", Operon::AggregateEvaluator::AggregateType::Min)
        .value("Max", Operon::AggregateEvaluator::AggregateType::Max)
        .value("Median", Operon::AggregateEvaluator::AggregateType::Median)
        .value("Mean", Operon::AggregateEvaluator::AggregateType::Mean)
        .value("HarmonicMean", Operon::AggregateEvaluator::AggregateType::HarmonicMean)
        .value("Sum", Operon::AggregateEvaluator::AggregateType::Sum);

    py::class_<Operon::AggregateEvaluator, Operon::EvaluatorBase>(m, "AggregateEvaluator")
        .def(py::init<Operon::EvaluatorBase&>())
        .def_property("AggregateType", &Operon::AggregateEvaluator::GetAggregateType, &Operon::AggregateEvaluator::SetAggregateType);

    py::class_<TMDLEvaluator, TEvaluator>(m, "MinimumDescriptionLengthEvaluator")
        .def(py::init<Operon::Problem&, TDispatch const&>())
        .def_property("Sigma", nullptr /*get*/ , &TMDLEvaluator::SetSigma /*set*/);
        // .def("__call__", &TMDLEvaluator::operator());

    py::class_<TBICEvaluator, TEvaluator>(m, "BayesianInformationCriterionEvaluator")
        .def(py::init<Operon::Problem&, TDispatch const&>());

    py::class_<TAIKEvaluator, TEvaluator>(m, "AkaikeInformationCriterionEvaluator")
        .def(py::init<Operon::Problem&, TDispatch const&>());
}