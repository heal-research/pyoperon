// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "pyoperon/pyoperon.hpp"

#include <operon/algorithms/config.hpp>
#include <operon/core/version.hpp>
#include <operon/formatter/formatter.hpp>
#include <operon/parser/infix.hpp>

#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/unordered_map.h>

NB_MODULE(pyoperon, m)
{
    m.doc() = "Operon Python Module";
    m.attr("__version__") = 0.1;

    // binding code
    nb::bind_vector<std::vector<Operon::Variable>>(m, "VariableCollection");
    nb::bind_vector<std::vector<Operon::Individual>>(m, "IndividualCollection");

    InitAlgorithm(m);
    InitBenchmark(m);
    InitCreator(m);
    InitCrossover(m);
    InitDataset(m);
    InitEval(m);
    InitGenerator(m);
    InitInitializer(m);
    InitMutation(m);
    InitNode(m);
    InitNondominatedSorter(m);
    InitOptimizer(m);
    InitProblem(m);
    InitPset(m);
    InitReinserter(m);
    InitSelector(m);
    InitTree(m);

    // build information
    m.def("Version", &Operon::Version);

    // random numbers
    m.def("UniformInt", &Operon::Random::Uniform<Operon::RandomGenerator, int>);
    m.def("UniformReal", &Operon::Random::Uniform<Operon::RandomGenerator, double>);

    // mathematical constants
    auto math = m.def_submodule("Math");
    math.attr("Constants") = Operon::Math::Constants;

    // classes
    nb::class_<Operon::Individual>(m, "Individual")
        .def(nb::init<>())
        .def(nb::init<size_t>())
        .def("__getitem__", nb::overload_cast<size_t>(&Operon::Individual::operator[]))
        .def("__getitem__", nb::overload_cast<size_t>(&Operon::Individual::operator[], nb::const_))
        .def_rw("Genotype", &Operon::Individual::Genotype)
        .def("SetFitness", [](Operon::Individual& self, Operon::Scalar f, size_t i) { self[i] = f; })
        .def("GetFitness", [](Operon::Individual& self, size_t i) { return self[i]; })
        .def("__getstate__",
            [](Operon::Individual const& ind) {
                return std::make_tuple(ind.Genotype, ind.Fitness, ind.Rank, ind.Distance);
            })
        .def("__setstate__",
            [](Operon::Individual& ind, std::tuple<Operon::Tree, std::vector<Operon::Scalar>, std::size_t, Operon::Scalar> const& t) {
                if (std::tuple_size_v<std::remove_cvref_t<decltype(t)>> != 4) { throw std::runtime_error("Invalid state!"); }
                ind.Genotype = std::get<0>(t);
                ind.Fitness  = std::get<1>(t);
                ind.Rank     = std::get<2>(t);
                ind.Distance = std::get<3>(t);
            }
        );

    nb::class_<Operon::SingleObjectiveComparison>(m, "SingleObjectiveComparison")
        .def(nb::init<size_t>())
        .def("__call__", &Operon::SingleObjectiveComparison::operator());

    nb::class_<Operon::CrowdedComparison>(m, "CrowdedComparison")
        .def(nb::init<>())
        .def("__call__", &Operon::CrowdedComparison::operator());

    nb::class_<Operon::Variable>(m, "Variable")
        .def_rw("Name", &Operon::Variable::Name)
        .def_rw("Hash", &Operon::Variable::Hash)
        .def_rw("Index", &Operon::Variable::Index)
        .def("__getstate__",
            [](Operon::Variable const& variable) {
                return std::make_tuple(variable.Name, variable.Hash, variable.Index);
            })
        .def("__setstate__",
            [](Operon::Variable& v, std::tuple<std::string, Operon::Hash, int64_t> const& t) {
                if (std::tuple_size_v<std::remove_cvref_t<decltype(t)>> != 3) {
                    throw std::runtime_error("Invalid state!");
                }
                v.Name = std::get<0>(t);
                v.Hash = std::get<1>(t);
                v.Index = std::get<2>(t);
            }
        );

    nb::class_<Operon::Range>(m, "Range")
        .def(nb::init<size_t, size_t>())
        .def(nb::init<std::pair<size_t, size_t>>())
        .def_prop_ro("Start", &Operon::Range::Start)
        .def_prop_ro("End", &Operon::Range::End)
        .def_prop_ro("Size", &Operon::Range::Size);

    // random generators
    nb::class_<Operon::Random::RomuTrio>(m, "RomuTrio")
        .def(nb::init<uint64_t>())
        .def("__call__", &Operon::Random::RomuTrio::operator());

    nb::class_<Operon::Random::Sfc64>(m, "Sfc64")
        .def(nb::init<uint64_t>())
        .def("__call__", &Operon::Random::Sfc64::operator());

    // tree format
    nb::class_<Operon::TreeFormatter>(m, "TreeFormatter")
        .def("Format", [](Operon::Tree const& tree, Operon::Dataset const& dataset, int decimalPrecision) {
            return Operon::TreeFormatter::Format(tree, dataset, decimalPrecision);
        })
        .def("Format", [](Operon::Tree const& tree, std::unordered_map<Operon::Hash, std::string> const& variables, int decimalPrecision) {
            Operon::Map<Operon::Hash, std::string> map(variables.begin(), variables.end());
            return Operon::TreeFormatter::Format(tree, map, decimalPrecision);
        });

    nb::class_<Operon::InfixFormatter>(m, "InfixFormatter")
        .def("Format", [](Operon::Tree const& tree, Operon::Dataset const& dataset, int decimalPrecision) {
            return Operon::InfixFormatter::Format(tree, dataset, decimalPrecision);
        })
        .def("Format", [](Operon::Tree const& tree, std::unordered_map<Operon::Hash, std::string> const& variables, int decimalPrecision) {
            Operon::Map<Operon::Hash, std::string> map(variables.begin(), variables.end());
            return Operon::InfixFormatter::Format(tree, map, decimalPrecision);
        });

    nb::class_<Operon::InfixParser>(m, "InfixParser")
        .def_static("Parse", [](std::string const& expr, std::unordered_map<std::string, Operon::Hash> const& variables) {
            Operon::Map<std::string, Operon::Hash> map(variables.begin(), variables.end());
            return Operon::InfixParser::Parse(expr, map);
        });

    // genetic algorithm
    nb::class_<Operon::GeneticAlgorithmConfig>(m, "GeneticAlgorithmConfig")
        .def_rw("Generations", &Operon::GeneticAlgorithmConfig::Generations)
        .def_rw("Evaluations", &Operon::GeneticAlgorithmConfig::Evaluations)
        .def_rw("Iterations", &Operon::GeneticAlgorithmConfig::Iterations)
        .def_rw("PopulationSize", &Operon::GeneticAlgorithmConfig::PopulationSize)
        .def_rw("PoolSize", &Operon::GeneticAlgorithmConfig::PoolSize)
        .def_rw("CrossoverProbability", &Operon::GeneticAlgorithmConfig::CrossoverProbability)
        .def_rw("MutationProbability", &Operon::GeneticAlgorithmConfig::MutationProbability)
        .def_rw("LocalSearchProbability", &Operon::GeneticAlgorithmConfig::LocalSearchProbability)
        .def_rw("LamarckianProbability", &Operon::GeneticAlgorithmConfig::LamarckianProbability)
        .def_rw("Seed", &Operon::GeneticAlgorithmConfig::Seed)
        .def_rw("Epsilon", &Operon::GeneticAlgorithmConfig::Epsilon)
        .def_rw("TimeLimit", &Operon::GeneticAlgorithmConfig::TimeLimit)
        .def("__init__", [](Operon::GeneticAlgorithmConfig* config, size_t gen, size_t evals, size_t iter, size_t popsize, size_t poolsize, double pc, double pm, double ps, double plm, double epsilon, size_t seed, size_t timelimit) {
            config->Generations = gen;
            config->Evaluations = evals;
            config->Iterations = iter;
            config->PopulationSize = popsize;
            config->PoolSize = poolsize;
            config->CrossoverProbability = pc;
            config->MutationProbability = pm;
            config->LocalSearchProbability = ps;
            config->LamarckianProbability = plm;
            config->Epsilon = epsilon;
            config->Seed = seed;
            config->TimeLimit = timelimit;
        } , nb::arg("generations")
          , nb::arg("max_evaluations")
          , nb::arg("local_iterations")
          , nb::arg("population_size")
          , nb::arg("pool_size")
          , nb::arg("p_crossover")
          , nb::arg("p_mutation")
          , nb::arg("p_local") = 1.0
          , nb::arg("p_lamarck") = 1.0
          , nb::arg("epsilon") = 1e-5
          , nb::arg("seed") = 0
          , nb::arg("time_limit") = ~size_t{0});
}
