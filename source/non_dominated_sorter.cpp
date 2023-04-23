// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "pyoperon/pyoperon.hpp"

#include <pybind11/stl.h>
#include <operon/operators/non_dominated_sorter.hpp>

namespace py = pybind11;

void InitNondominatedSorter(py::module_ &m)
{
    py::class_<Operon::NondominatedSorterBase>(m, "NonDominatedSorterBase");

    py::class_<Operon::RankIntersectSorter, Operon::NondominatedSorterBase>(m, "RankSorter")
        .def(py::init<>())
        .def("Sort", [](Operon::RankIntersectSorter const& self, std::vector<std::vector<Operon::Scalar>> const& points) {
            auto sz = points.size();

            std::vector<Operon::Individual> vec;
            vec.reserve(sz);
            std::transform(points.begin(), points.end(), std::back_inserter(vec), [](std::vector<Operon::Scalar> const& a) {
                Operon::Individual ind(a.size());
                std::copy(a.begin(), a.end(), ind.Fitness.begin());
                return ind;
            });
            return self.Sort(vec, std::numeric_limits<Operon::Scalar>::epsilon());
        });
}
