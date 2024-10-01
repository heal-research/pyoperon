// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include <stdexcept>
#include <operon/optimizer/optimizer.hpp>
#include <operon/operators/local_search.hpp>
#include <operon/optimizer/solvers/sgd.hpp>

#include "pyoperon/pyoperon.hpp"
#include "pyoperon/optimizer.hpp"

namespace detail {

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
        })
        .def("__call__", &Operon::CoefficientOptimizer::operator());

    nb::class_<Operon::OptimizerSummary>(m, "OptimizerSummary")
        .def_rw("InitialCost", &Operon::OptimizerSummary::InitialCost)
        .def_rw("FinalCost", &Operon::OptimizerSummary::FinalCost)
        .def_rw("Iterations", &Operon::OptimizerSummary::Iterations)
        .def_rw("FunctionEvaluations", &Operon::OptimizerSummary::FunctionEvaluations)
        .def_rw("JacobianEvaluations", &Operon::OptimizerSummary::JacobianEvaluations)
        .def_rw("Success", &Operon::OptimizerSummary::Success)
        .def_rw("InitialParameters", &Operon::OptimizerSummary::InitialParameters)
        .def_rw("FinalParameters", &Operon::OptimizerSummary::FinalParameters);


    nb::class_<detail::Optimizer>(m, "Optimizer");

    nb::class_<detail::LMOptimizer, detail::Optimizer>(m, "LMOptimizer")
        .def(nb::init<TDispatch const&, Problem const&, std::size_t, std::size_t>()
            , nb::arg("dtable")
            , nb::arg("problem")
            , nb::arg("max_iter") = 10
            , nb::arg("batch_size") = TDispatch::BatchSize<Operon::Scalar>
        );

    nb::class_<detail::LBFGSOptimizer, detail::Optimizer>(m, "LBFGSOptimizer")
        .def(nb::init<TDispatch const&, Problem const&, std::string const&, std::size_t, std::size_t>()
            , nb::arg("dtable")
            , nb::arg("problem")
            , nb::arg("likelihood") = "gaussian"
            , nb::arg("max_iter") = 10
            , nb::arg("batch_size") = TDispatch::BatchSize<Operon::Scalar>
        );

    nb::class_<detail::SGDOptimizer, detail::Optimizer>(m, "SGDOptimizer")
        .def(nb::init<TDispatch const&, Problem const&, LearningRateUpdateRule const&, std::string const&, std::size_t, std::size_t>()
            , nb::arg("dtable")
            , nb::arg("problem")
            , nb::arg("update_rule")
            , nb::arg("likelihood") = "gaussian"
            , nb::arg("max_iter") = 10
            , nb::arg("batch_size") = TDispatch::BatchSize<Operon::Scalar>
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

    nb::class_<TAdaMaxUpdateRule, TUpdateRule>(m, "AdaDeltaUpdateRule")
        .def(nb::init<Eigen::Index, Operon::Scalar, Operon::Scalar, Operon::Scalar>()
            , nb::arg("dimension") = 0
            , nb::arg("learning_rate") = 0.01
            , nb::arg("beta1") = 0.9
            , nb::arg("beta2") = 0.999
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