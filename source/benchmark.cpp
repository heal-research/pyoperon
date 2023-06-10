// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include <thread>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#if TF_MINOR_VERSION > 2
#include <taskflow/algorithm/reduce.hpp>
#endif

#include <operon/core/dataset.hpp>
#include <operon/random/random.hpp>
#include <operon/operators/creator.hpp>
#include <operon/operators/evaluator.hpp>

#include "pyoperon/pyoperon.hpp"

void InitBenchmark(py::module_ &m)
{
    // benchmark functionality
    m.def("Bench", [](int nTrees = 10000, int maxLength = 50, int maxDepth = 100, int nRows = 10000, int nCols = 10, int seed = 0, int nThreads = 0) {
        std::uniform_real_distribution<Operon::Scalar> dist(-1.f, +1.f);
        Eigen::Matrix<decltype(dist)::result_type, -1, -1> data(nRows, nCols);

        Operon::RandomGenerator rng(seed);
        for (auto& v : data.reshaped()) { v = dist(rng); }
        Operon::Dataset ds(data);

        auto const* targetName = "Y";
        auto inputs = ds.VariableHashes();
        auto target = ds.GetVariable(targetName);
        std::erase(inputs, target->Hash);

        Operon::PrimitiveSet pset;
        pset.SetConfig(Operon::PrimitiveSet::Arithmetic);

        std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);
        auto creator = Operon::BalancedTreeCreator { pset, inputs };

        if (nThreads == 0) { nThreads = std::thread::hardware_concurrency(); }

        Operon::Range range{0, ds.Rows<std::size_t>()};

        std::vector<Operon::Scalar> result(range.Size() * nTrees);
        std::vector<Operon::Tree> trees(nTrees);
        std::generate(trees.begin(), trees.end(), [&]() { return creator(rng, sizeDistribution(rng), 0, maxDepth); });
#ifdef _MSC_VER
        auto nTotal = std::reduce(trees.begin(), trees.end(), 0UL, [](size_t partial, const auto& t) { return partial + t.Length(); });
#else
        auto nTotal = std::transform_reduce(trees.begin(), trees.end(), 0UL, std::plus<> {}, [](auto& tree) { return tree.Length(); });
#endif
        Operon::Interpreter interpreter;
        tf::Executor executor(nThreads);
        std::vector<std::vector<Operon::Scalar>> values(executor.num_workers());
        for (auto& val : values) { val.resize(range.Size()); }
        tf::Taskflow taskflow;
        taskflow.for_each(trees.begin(), trees.end(), [&](auto const& tree) {
            auto& val = values[executor.this_worker_id()];
            interpreter.operator()<Operon::Scalar>(tree, ds, range, val);
        });
        executor.run(taskflow).wait();
        return nTotal * range.Size();
    }, py::call_guard<py::gil_scoped_release>(),
       py::arg("num_trees") = 10000,
       py::arg("max_length") = 50,
       py::arg("max_depth") = 100,
       py::arg("num_rows") = 10000,
       py::arg("num_cols") = 10,
       py::arg("random_state") = 0,
       py::arg("num_threads") = 0
    );
}
