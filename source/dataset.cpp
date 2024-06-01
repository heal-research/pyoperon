// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include <operon/core/dataset.hpp>
#include <utility>

#include "pyoperon/pyoperon.hpp"

namespace nb = nanobind;

template<typename T>
auto InitDataset(Operon::Dataset* ds, nb::ndarray<T, nb::f_contig> arr)
{
    auto const* data = arr.data();
    auto rows = arr.shape(0);
    auto cols = arr.shape(1);
    if constexpr (std::is_same_v<T, Operon::Scalar>) {
        new (ds) Operon::Dataset(data, rows, cols);
    } else {
#ifdef DEBUG
        fmt::print("warning: data types do not match, data will be copied\n");
#endif
        Eigen::Matrix<Operon::Scalar, -1, -1> values(rows, cols);
        values = Eigen::Map<Eigen::Matrix<T, -1, -1> const>(data, rows, cols).template cast<Operon::Scalar>();
        new (ds) Operon::Dataset(values);
    }
}

void InitDataset(nb::module_ &m)
{
    // dataset
    nb::class_<Operon::Dataset>(m, "Dataset")
        .def(nb::init<std::string const&, bool>(), nb::arg("filename"), nb::arg("has_header"))
        .def(nb::init<Operon::Dataset const&>())
        .def(nb::init<std::vector<std::string> const&, const std::vector<std::vector<Operon::Scalar>>&>())
        .def("__init__", [](Operon::Dataset* ds, nb::ndarray<float, nb::f_contig> array) { InitDataset(ds, array); }, nb::arg("data").noconvert())
        .def("__init__", [](Operon::Dataset* ds, nb::ndarray<double, nb::f_contig> array) { InitDataset(ds, array); }, nb::arg("data").noconvert())
        .def_prop_ro("Rows", &Operon::Dataset::Rows<int64_t>)
        .def_prop_ro("Cols", &Operon::Dataset::Cols<int64_t>)
        .def_prop_ro("Values", &Operon::Dataset::Values)
        .def_prop_rw("VariableNames", &Operon::Dataset::VariableNames, &Operon::Dataset::SetVariableNames)
        .def_prop_ro("VariableHashes", &Operon::Dataset::VariableHashes)
        .def("GetValues", [](Operon::Dataset const& self, std::string const& name) { return MakeView(self.GetValues(name)); })
        .def("GetValues", [](Operon::Dataset const& self, Operon::Hash hash) { return MakeView(self.GetValues(hash)); })
        .def("GetValues", [](Operon::Dataset const& self, int64_t index) { return MakeView(self.GetValues(index)); })
        .def("GetVariable", nb::overload_cast<const std::string&>(&Operon::Dataset::GetVariable, nb::const_))
        .def("GetVariable", nb::overload_cast<Operon::Hash>(&Operon::Dataset::GetVariable, nb::const_))
        .def_prop_ro("Variables", &Operon::Dataset::GetVariables)
        .def("Shuffle", &Operon::Dataset::Shuffle)
        .def("Normalize", &Operon::Dataset::Normalize)
        .def("Standardize", &Operon::Dataset::Standardize)
        ;
}
