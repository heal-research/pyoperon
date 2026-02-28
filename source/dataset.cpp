// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include <operon/core/dataset.hpp>
#include <utility>

#include "pyoperon/pyoperon.hpp"

namespace nb = nanobind;

namespace {
template<typename T>
auto InitDataset(Operon::Dataset* ds, nb::ndarray<T, nb::ndim<2>, nb::f_contig> const& arr)
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

// initialization function when the memory-order doesn't match (we do a manual copy)
template<typename T>
auto InitDataset(Operon::Dataset* ds, nb::ndarray<T, nb::ndim<2>, nb::c_contig> const& arr)
{
    auto rows = arr.shape(0);
    auto cols = arr.shape(1);

    Eigen::Matrix<Operon::Scalar, -1, -1> values(rows, cols);
    for (auto i = 0L; i < rows; ++i) {
        for (auto j = 0L; j < cols; ++j) {
            values(i, j) = arr(i, j);
        }
    }
    new (ds) Operon::Dataset(values);
}

// handles the case of a generic, unspecified ndarray
// (a copy will potentially be made)
auto InitDataset(Operon::Dataset* ds, nb::object arr)
{
#ifdef DEBUG
    fmt::print("received an nb::object parameter, trying to deduce type.\n");
#endif

    nb::ndarray<float,  nb::ndim<2>, nb::c_contig> a1;
    nb::ndarray<double, nb::ndim<2>, nb::c_contig> a2;
    nb::ndarray<float,  nb::ndim<2>, nb::f_contig> a3;
    nb::ndarray<double, nb::ndim<2>, nb::f_contig> a4;

    if (nb::try_cast(arr, a1)) { return InitDataset(ds, a1); }
    if (nb::try_cast(arr, a2)) { return InitDataset(ds, a2); }
    if (nb::try_cast(arr, a3)) { return InitDataset(ds, a3); }
    if (nb::try_cast(arr, a4)) { return InitDataset(ds, a4); }

    std::string repr = nb::repr(arr).c_str();
    throw std::runtime_error(fmt::format("dataset initialization failed: unable to convert object of type {} to ndarray.", repr));
}
} // namespace

void InitDataset(nb::module_ &m)
{
    // dataset
    nb::class_<Operon::Dataset>(m, "Dataset")
        .def(nb::init<std::string const&, bool>(), nb::arg("filename"), nb::arg("has_header"))
        .def(nb::init<Operon::Dataset const&>())
        .def(nb::init<std::vector<std::string> const&, const std::vector<std::vector<Operon::Scalar>>&>())
        .def("__init__", [](Operon::Dataset* ds, nb::ndarray<float, nb::ndim<2>, nb::f_contig> array) { InitDataset(ds, array); }, nb::arg("data").noconvert())
        .def("__init__", [](Operon::Dataset* ds, nb::ndarray<double, nb::ndim<2>, nb::f_contig> array) { InitDataset(ds, array); }, nb::arg("data").noconvert())
        .def("__init__", [](Operon::Dataset* ds, nb::ndarray<float, nb::ndim<2>, nb::c_contig> array) {
#ifdef DEBUG
            fmt::print("warning: unsupported memory layout, data will be copied\n");
#endif
            InitDataset(ds, array);
        }, nb::arg("data").noconvert())
        .def("__init__", [](Operon::Dataset* ds, nb::ndarray<double, nb::ndim<2>, nb::c_contig> array) -> void {
#ifdef DEBUG
            fmt::print("warning: unsupported memory layout, data will be copied\n");
#endif
            InitDataset(ds, array);
        }, nb::arg("data").noconvert())
        .def("__init__", [](Operon::Dataset* ds, nb::object array) -> void {
#ifdef DEBUG
            fmt::print("warning: unsupported memory layout, data will be copied\n");
#endif
            InitDataset(ds, array);
        }, nb::arg("data").noconvert())
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
