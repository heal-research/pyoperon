// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include <stdexcept>
#include <operon/optimizer/optimizer.hpp>
#include <operon/operators/local_search.hpp>
#include <operon/optimizer/solvers/sgd.hpp>

#include <nanobind/stl/tuple.h>
#include <nanobind/stl/string.h>

#include "pyoperon/pyoperon.hpp"
#include "pyoperon/optimizer.hpp"

namespace detail {

// operon's CoefficientOptimizer/OptimizerBase::Optimize now return a
// tl::expected<FitResult, FitFailure> outcome (nanobind has no built-in
// caster for tl::expected). This flattens that outcome back into the same
// field-level shape pyoperon has always exposed as "OptimizerSummary",
// keeping the Python-facing API unchanged - callers that read `.Success`
// still get a bool, `.FinalCost` etc. still populated regardless of outcome
// (both FitResult and FitFailure carry full diagnostics; see
// Operon::Diagnostics() in optimizer.hpp).
struct PySummary {
    std::vector<Operon::Scalar> InitialParameters;
    std::vector<Operon::Scalar> FinalParameters;
    Operon::Scalar InitialCost{};
    Operon::Scalar FinalCost{};
    int Iterations{};
    int FunctionEvaluations{};
    int JacobianEvaluations{};
    bool Success{};
};

inline auto ToPySummary(Operon::FitOutcome const& outcome) -> PySummary {
    auto const& diag = Operon::Diagnostics(outcome);
    return PySummary{
        diag.InitialParameters,
        diag.FinalParameters,
        diag.InitialCost,
        diag.FinalCost,
        diag.Iterations,
        diag.FunctionEvaluations,
        diag.JacobianEvaluations,
        outcome.has_value()
    };
}

class LMOptimizer : public Optimizer {
    public:
    LMOptimizer(TDispatch const& dispatch, Operon::Problem const& problem, std::size_t maxIter, std::size_t batchSize)
    {
        Optimizer::Set(std::make_unique<TLMOptimizerEigen>(&dispatch, &problem));
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
            Optimizer::Set(std::make_unique<TLBFGSOptimizerGauss>(&dispatch, &problem));
        } else if (likelihood == "poisson") {
            Optimizer::Set(std::make_unique<TLBFGSOptimizerPoisson>(&dispatch, &problem));
        } else if (likelihood == "poisson_log") {
            Optimizer::Set(std::make_unique<TLBFGSOptimizerPoissonLog>(&dispatch, &problem));
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
            Optimizer::Set(std::make_unique<TSGDOptimizerGauss>(&dispatch, &problem, updateRule));
        } else if (likelihood == "poisson") {
            Optimizer::Set(std::make_unique<TSGDOptimizerPoisson>(&dispatch, &problem, updateRule));
        } else if (likelihood == "poisson_log") {
            Optimizer::Set(std::make_unique<TSGDOptimizerPoissonLog>(&dispatch, &problem, updateRule));
        } else {
            throw std::invalid_argument(fmt::format("{} is not a valid likelihood\n", likelihood));
        }

        auto const* opt = Optimizer::Get();
        opt->SetIterations(maxIter);
        opt->SetBatchSize(batchSize);
    }
};
} // namespace detail

void InitOptimizer(nb::module_ &m)
{
    using Operon::Problem;
    using Operon::UpdateRule::LearningRateUpdateRule;
    using std::string;
    using std::size_t;

    nb::class_<Operon::CoefficientOptimizer>(m, "CoefficientOptimizer")
        .def("__init__", [](Operon::CoefficientOptimizer* op, detail::Optimizer const& opt) {
            new (op) Operon::CoefficientOptimizer(opt.Get());
        }, nb::keep_alive<1, 2>())
        .def("__call__", [](Operon::CoefficientOptimizer const& self, Operon::RandomGenerator& rng, Operon::Tree tree) {
            auto [t, outcome] = self(rng, std::move(tree));
            return std::tuple{std::move(t), detail::ToPySummary(outcome)};
        });

    nb::class_<detail::PySummary>(m, "OptimizerSummary")
        .def_rw("InitialCost", &detail::PySummary::InitialCost)
        .def_rw("FinalCost", &detail::PySummary::FinalCost)
        .def_rw("Iterations", &detail::PySummary::Iterations)
        .def_rw("FunctionEvaluations", &detail::PySummary::FunctionEvaluations)
        .def_rw("JacobianEvaluations", &detail::PySummary::JacobianEvaluations)
        .def_rw("Success", &detail::PySummary::Success)
        .def_rw("InitialParameters", &detail::PySummary::InitialParameters)
        .def_rw("FinalParameters", &detail::PySummary::FinalParameters);


    nb::class_<detail::Optimizer>(m, "Optimizer");

    nb::class_<detail::LMOptimizer, detail::Optimizer>(m, "LMOptimizer")
        .def(nb::init<TDispatch const&, Problem const&, std::size_t, std::size_t>()
            , nb::arg("dtable")
            , nb::arg("problem")
            , nb::arg("max_iter") = 10
            , nb::arg("batch_size") = TDispatch::BatchSize<Operon::Scalar>
            , nb::keep_alive<1, 2>()
            , nb::keep_alive<1, 3>()
        );

    nb::class_<detail::LBFGSOptimizer, detail::Optimizer>(m, "LBFGSOptimizer")
        .def(nb::init<TDispatch const&, Problem const&, std::string const&, std::size_t, std::size_t>()
            , nb::arg("dtable")
            , nb::arg("problem")
            , nb::arg("likelihood") = "gaussian"
            , nb::arg("max_iter") = 10
            , nb::arg("batch_size") = TDispatch::BatchSize<Operon::Scalar>
            , nb::keep_alive<1, 2>()
            , nb::keep_alive<1, 3>()
        );

    nb::class_<detail::SGDOptimizer, detail::Optimizer>(m, "SGDOptimizer")
        .def(nb::init<TDispatch const&, Problem const&, LearningRateUpdateRule const&, std::string const&, std::size_t, std::size_t>()
            , nb::arg("dtable")
            , nb::arg("problem")
            , nb::arg("update_rule")
            , nb::arg("likelihood") = "gaussian"
            , nb::arg("max_iter") = 10
            , nb::arg("batch_size") = TDispatch::BatchSize<Operon::Scalar>
            , nb::keep_alive<1, 2>()
            , nb::keep_alive<1, 3>()
        );

    // SGD update rules class definitions
    nb::class_<TUpdateRule>(m, "UpdateRule"); // base class

    nb::class_<TConstantUpdateRule, TUpdateRule>(m, "ConstantUpdateRule")
        .def(nb::init<Eigen::Index, Operon::Scalar>()
            , nb::arg("dimension") = 0
            , nb::arg("learning_rate") = 0.01
        );

    nb::class_<TMomentumUpdateRule, TUpdateRule>(m, "MomentumUpdateRule")
        .def(nb::init<Eigen::Index, Operon::Scalar, Operon::Scalar>()
            , nb::arg("dimension") = 0
            , nb::arg("learning_rate") = 0.01
            , nb::arg("beta") = 0.9
        );

    nb::class_<TRmsPropUpdateRule, TUpdateRule>(m, "RmsPropUpdateRule")
        .def(nb::init<Eigen::Index, Operon::Scalar, Operon::Scalar, Operon::Scalar>()
            , nb::arg("dimension") = 0
            , nb::arg("learning_rate") = 0.01
            , nb::arg("beta") = 0.9
            , nb::arg("eps") = 1e-6
        );

    nb::class_<TAdaDeltaUpdateRule, TUpdateRule>(m, "AdaDeltaUpdateRule")
        .def(nb::init<Eigen::Index, Operon::Scalar, Operon::Scalar>()
            , nb::arg("dimension") = 0
            , nb::arg("beta") = 0.9
            , nb::arg("epsilon") = 1e-6
        );

    nb::class_<TAdaMaxUpdateRule, TUpdateRule>(m, "AdaMaxUpdateRule")
        .def(nb::init<Eigen::Index, Operon::Scalar, Operon::Scalar, Operon::Scalar>()
            , nb::arg("dimension") = 0
            , nb::arg("learning_rate") = 0.01
            , nb::arg("beta1") = 0.9
            , nb::arg("beta2") = 0.999
        );

    nb::class_<TAdamUpdateRule, TUpdateRule>(m, "AdamUpdateRule")
        .def(nb::init<Eigen::Index, Operon::Scalar, Operon::Scalar, Operon::Scalar, Operon::Scalar>()
            , nb::arg("dimension") = 0
            , nb::arg("learning_rate") = 0.01
            , nb::arg("epsilon") = 1e-8
            , nb::arg("beta1") = 0.9
            , nb::arg("beta2") = 0.999
        );

    nb::class_<TYamAdamUpdateRule, TUpdateRule>(m, "YamAdamUpdateRule")
        .def(nb::init<Eigen::Index, Operon::Scalar>()
            , nb::arg("dimension") = 0
            , nb::arg("epsilon") = 1e-6
        );

    nb::class_<TYogiUpdateRule, TUpdateRule>(m, "YogiUpdateRule")
        .def(nb::init<Eigen::Index, Operon::Scalar, Operon::Scalar, Operon::Scalar, Operon::Scalar, bool>()
            , nb::arg("dimension") = 0
            , nb::arg("learning_rate") = 0.01
            , nb::arg("epsilon") = 1e-8
            , nb::arg("beta1") = 0.9
            , nb::arg("beta2") = 0.999
            , nb::arg("debias") = false
        );

    nb::class_<TAmsGradUpdateRule, TUpdateRule>(m, "AmsGradUpdateRule")
        .def(nb::init<Eigen::Index, Operon::Scalar, Operon::Scalar, Operon::Scalar, Operon::Scalar>()
            , nb::arg("dimension") = 0
            , nb::arg("learning_rate") = 0.01
            , nb::arg("epsilon")
            , nb::arg("beta1") = 0.9
            , nb::arg("beta2") = 0.999
        );
}