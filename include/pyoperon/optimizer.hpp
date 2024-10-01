// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2024 Heal Research

#include <operon/operators/evaluator.hpp>

using TDispatch                = Operon::DefaultDispatch;
using TInterpreter             = Operon::Interpreter<Operon::Scalar, TDispatch>;
using TInterpreterBase         = Operon::InterpreterBase<Operon::Scalar>;

// likelihood
using TGaussianLikelihood      = Operon::GaussianLikelihood<Operon::Scalar>;
using TPoissonLikelihood       = Operon::PoissonLikelihood<Operon::Scalar, false>;
using TPoissonLikelihoodLog    = Operon::PoissonLikelihood<Operon::Scalar, true>;

// likelihood evaluators
using TGaussEvaluator          = Operon::GaussianLikelihoodEvaluator<TDispatch>;
using TPoissonEvaluator        = Operon::LikelihoodEvaluator<TDispatch, TPoissonLikelihood>;
using TPoissonLogEvaluator     = Operon::LikelihoodEvaluator<TDispatch, TPoissonLikelihoodLog>;

// evaluator
using TEvaluatorBase           = Operon::EvaluatorBase;
using TEvaluator               = Operon::Evaluator<TDispatch>;
using TMDLEvaluatorGauss       = Operon::MinimumDescriptionLengthEvaluator<TDispatch, TGaussianLikelihood>;
using TMDLEvaluatorPoisson     = Operon::MinimumDescriptionLengthEvaluator<TDispatch, TPoissonLikelihood>;
using TMDLEvaluatorPoissonLog  = Operon::MinimumDescriptionLengthEvaluator<TDispatch, TPoissonLikelihoodLog>;

using TBICEvaluator            = Operon::BayesianInformationCriterionEvaluator<TDispatch>;
using TAIKEvaluator            = Operon::AkaikeInformationCriterionEvaluator<TDispatch>;

// optimizer
using TOptimizerBase           = Operon::OptimizerBase;

// optimizer::lm
using TLMOptimizerEigen        = Operon::LevenbergMarquardtOptimizer<TDispatch, Operon::OptimizerType::Eigen>;

// optimizer::lbfgs
using TLBFGSOptimizerGauss      = Operon::LBFGSOptimizer<TDispatch, TGaussianLikelihood>;
using TLBFGSOptimizerPoisson    = Operon::LBFGSOptimizer<TDispatch, TPoissonLikelihood>;
using TLBFGSOptimizerPoissonLog = Operon::LBFGSOptimizer<TDispatch, TPoissonLikelihoodLog>;

// optimizer::sgd
using TSGDOptimizerGauss        = Operon::SGDOptimizer<TDispatch, TGaussianLikelihood>;
using TSGDOptimizerPoisson      = Operon::SGDOptimizer<TDispatch, TPoissonLikelihood>;
using TSGDOptimizerPoissonLog   = Operon::SGDOptimizer<TDispatch, TPoissonLikelihoodLog>;

// optimizer::sgd::update_rule
using TUpdateRule               = Operon::UpdateRule::LearningRateUpdateRule;
using TConstantUpdateRule       = Operon::UpdateRule::Constant<Operon::Scalar>;
using TMomentumUpdateRule       = Operon::UpdateRule::Momentum<Operon::Scalar>;
using TRmsPropUpdateRule        = Operon::UpdateRule::RmsProp<Operon::Scalar>;
using TAdaDeltaUpdateRule       = Operon::UpdateRule::AdaDelta<Operon::Scalar>;
using TAdaMaxUpdateRule         = Operon::UpdateRule::AdaMax<Operon::Scalar>;
using TAdamUpdateRule           = Operon::UpdateRule::Adam<Operon::Scalar>;
using TYamAdamUpdateRule        = Operon::UpdateRule::YamAdam<Operon::Scalar>;
using TAmsGradUpdateRule        = Operon::UpdateRule::AmsGrad<Operon::Scalar>;
using TYogiUpdateRule           = Operon::UpdateRule::Yogi<Operon::Scalar>;

namespace detail {
class Optimizer {
    std::unique_ptr<TOptimizerBase const> optimizer_;

public:
    auto SetBatchSize(std::size_t value) const { optimizer_->SetBatchSize(value); }
    [[nodiscard]] auto BatchSize() const { return optimizer_->BatchSize(); }

    auto SetIterations(std::size_t value) const { optimizer_->SetIterations(value); }
    [[nodiscard]] auto Iterations() const { return optimizer_->Iterations(); }

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

    auto Set(TOptimizerBase* optimizer) {
        optimizer_.reset(optimizer);
    }

    [[nodiscard]] auto Get() const { return optimizer_.get(); }
};
}  // namespace detail
