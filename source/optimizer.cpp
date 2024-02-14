// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

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
            Optimizer::Set(std::make_unique<TLBFGSOptimizerGauss>(dispatch, problem));
        } else if (likelihood == "poisson") {
            Optimizer::Set(std::make_unique<TLBFGSOptimizerPoisson>(dispatch, problem));
        } else if (likelihood == "poisson_log") {
            Optimizer::Set(std::make_unique<TLBFGSOptimizerPoissonLog>(dispatch, problem));
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


    py::class_<detail::Optimizer>(m, "Optimizer");

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

    // SGD update rules class definitions
    py::class_<TUpdateRule>(m, "UpdateRule"); // base class

    py::class_<TConstantUpdateRule, TUpdateRule>(m, "ConstantUpdateRule")
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