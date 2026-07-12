// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include <operon/core/dataset.hpp>
#include <utility>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "pyoperon/pyoperon.hpp"

namespace nb = nanobind;

namespace {

// f-contiguous (column-major) — same layout as Dataset's mdarray storage
template<typename T>
auto InitDataset(Operon::Dataset* ds, nb::ndarray<T, nb::ndim<2>, nb::f_contig> const& arr)
{
    auto rows = static_cast<int>(arr.shape(0));
    auto cols = static_cast<int>(arr.shape(1));
    if constexpr (std::is_same_v<T, Operon::Scalar>) {
        new (ds) Operon::Dataset(arr.data(), rows, cols);
    } else {
        // cast to Scalar column-by-column
        std::vector<std::vector<Operon::Scalar>> columns(cols, std::vector<Operon::Scalar>(rows));
        for (auto j = 0; j < cols; ++j) {
            for (auto i = 0; i < rows; ++i) {
                columns[j][i] = static_cast<Operon::Scalar>(arr(i, j));
            }
        }
        new (ds) Operon::Dataset(columns);
    }
}

// c-contiguous (row-major) — must transpose to column-major for Dataset storage
template<typename T>
auto InitDataset(Operon::Dataset* ds, nb::ndarray<T, nb::ndim<2>, nb::c_contig> const& arr)
{
    auto rows = static_cast<int>(arr.shape(0));
    auto cols = static_cast<int>(arr.shape(1));
    std::vector<std::vector<Operon::Scalar>> columns(cols, std::vector<Operon::Scalar>(rows));
    for (auto i = 0; i < rows; ++i) {
        for (auto j = 0; j < cols; ++j) {
            columns[j][i] = static_cast<Operon::Scalar>(arr(i, j));
        }
    }
    new (ds) Operon::Dataset(columns);
}

// handles the case of a generic, unspecified ndarray (a copy may be made)
auto InitDataset(Operon::Dataset* ds, nb::object arr)
{
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
            InitDataset(ds, array);
        }, nb::arg("data").noconvert())
        .def("__init__", [](Operon::Dataset* ds, nb::ndarray<double, nb::ndim<2>, nb::c_contig> array) -> void {
            InitDataset(ds, array);
        }, nb::arg("data").noconvert())
        .def("__init__", [](Operon::Dataset* ds, nb::object array) -> void {
            InitDataset(ds, array);
        }, nb::arg("data").noconvert())
        .def_prop_ro("Rows", &Operon::Dataset::Rows<int64_t>)
        .def_prop_ro("Cols", &Operon::Dataset::Cols<int64_t>)
        .def_prop_ro("Values", [](Operon::Dataset const& ds) {
            // Data() returns the SIMD-padded mdspan ((rows+7)&~7 rows for owning
            // datasets); expose the logical (rows, cols) shape via an explicit
            // column stride of paddedRows rather than the padded row count.
            auto view = ds.Data();
            size_t shape[2] = {static_cast<size_t>(ds.Rows()), static_cast<size_t>(view.extent(1))};
            int64_t strides[2] = {1, static_cast<int64_t>(view.extent(0))};
            return nb::ndarray<Operon::Scalar const, nb::numpy>(
                view.data_handle(), 2, shape, nb::find(ds), strides
            );
        })
        .def_prop_rw("VariableNames", &Operon::Dataset::VariableNames, &Operon::Dataset::SetVariableNames)
        .def_prop_ro("VariableHashes", &Operon::Dataset::VariableHashes)
        .def("GetValues", [](Operon::Dataset const& self, std::string const& name) { return MakeView(self.GetValues(name), nb::find(self)); })
        .def("GetValues", [](Operon::Dataset const& self, Operon::Hash hash) { return MakeView(self.GetValues(hash), nb::find(self)); })
        .def("GetValues", [](Operon::Dataset const& self, int64_t index) { return MakeView(self.GetValues(index), nb::find(self)); })
        .def("GetVariable", [](Operon::Dataset const& self, std::string const& name) -> std::optional<Operon::Variable> {
            auto res = self.GetVariable(name);
            return res.has_value() ? std::optional{std::move(*res)} : std::nullopt;
        })
        .def("GetVariable", [](Operon::Dataset const& self, Operon::Hash hash) -> std::optional<Operon::Variable> {
            auto res = self.GetVariable(hash);
            return res.has_value() ? std::optional{std::move(*res)} : std::nullopt;
        })
        .def_prop_ro("Variables", &Operon::Dataset::GetVariables)
        .def("SetWeights", [](Operon::Dataset& self, std::vector<Operon::Scalar> const& w) { self.SetWeights(w); })
        // Like GetValues/Values above, this is a view into weights_; a
        // subsequent SetWeights() reallocates the backing vector, so
        // re-fetch Weights after calling SetWeights rather than holding
        // onto a stale view.
        .def_prop_ro("Weights", [](Operon::Dataset const& self) -> nb::object {
            auto w = self.Weights();
            return w ? nb::cast(MakeView(*w, nb::find(self))) : nb::none();
        })
        .def("Shuffle", &Operon::Dataset::Shuffle)
        .def("Normalize", &Operon::Dataset::Normalize)
        .def("Standardize", &Operon::Dataset::Standardize)
        ;
}
