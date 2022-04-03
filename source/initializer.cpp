#include "pyoperon/pyoperon.hpp"

#include <random>

#include <operon/operators/initializer.hpp>
#include <operon/operators/creator.hpp>

using UniformLengthTreeInitializer = Operon::TreeInitializer<std::uniform_int_distribution<size_t>>;
using NormalDistributedCoefficientInitializer = Operon::CoefficientInitializer<std::normal_distribution<double>>;

void InitInitializer(py::module_ &m)
{
    py::class_<Operon::TreeInitializerBase> treeInitializerBase(m, "TreeInitializerBase");
    py::class_<Operon::CoefficientInitializerBase> coeffInitializerBase(m, "CoefficientInitializerBase");

    py::class_<UniformLengthTreeInitializer, Operon::TreeInitializerBase>(m, "UniformLengthTreeInitializer")
        .def(py::init<Operon::CreatorBase&>())
        .def(py::init<Operon::BalancedTreeCreator&>())
        .def(py::init<Operon::GrowTreeCreator&>())
        .def("__call__", [](UniformLengthTreeInitializer& self, Operon::RandomGenerator& random) {
                return self(random);
            })
        .def_property("MinDepth", &UniformLengthTreeInitializer::MinDepth, &UniformLengthTreeInitializer::SetMinDepth)
        .def_property("MaxDepth", &UniformLengthTreeInitializer::MaxDepth, &UniformLengthTreeInitializer::SetMaxDepth)
        .def("ParameterizeDistribution", [](UniformLengthTreeInitializer& self, size_t lower, size_t upper) {
                self.ParameterizeDistribution(lower, upper);
            });

    py::class_<NormalDistributedCoefficientInitializer, Operon::CoefficientInitializerBase>(m, "NormalCoefficientInitializer")
        .def(py::init<>())
        .def("__call__", [](NormalDistributedCoefficientInitializer& self, Operon::RandomGenerator& random, Operon::Tree& tree) {
                return self(random, tree);
            })
        .def("ParameterizeDistribution", [](NormalDistributedCoefficientInitializer& self, double mean, double stdev) {
                self.ParameterizeDistribution(mean, stdev);
            });

    using UniformIntCoefficientInitializer = Operon::CoefficientInitializer<std::uniform_int_distribution<int>>;
    using UniformRealCoefficientInitializer = Operon::CoefficientInitializer<std::uniform_real_distribution<double>>;

    py::class_<UniformIntCoefficientInitializer, Operon::CoefficientInitializerBase>(m, "UniformIntCoefficientAnalyzer")
        .def(py::init<>())
        .def("__call__", [](UniformIntCoefficientInitializer& self, Operon::RandomGenerator& random, Operon::Tree& tree) {
                return self(random, tree);
            })
        .def("ParameterizeDistribution", [](UniformIntCoefficientInitializer& self, int a, int b) {
                self.ParameterizeDistribution(a, b);
            });

    py::class_<UniformRealCoefficientInitializer, Operon::CoefficientInitializerBase>(m, "UniformRealCoefficientAnalyzer")
        .def(py::init<>())
        .def("__call__", [](UniformRealCoefficientInitializer& self, Operon::RandomGenerator& random, Operon::Tree& tree) {
                return self(random, tree);
            })
        .def("ParameterizeDistribution", [](UniformRealCoefficientInitializer& self, double a, double b) {
                self.ParameterizeDistribution(a, b);
            });
}

