#!/usr/bin/env bash

if [[ -z "${CONDA_PREFIX}" ]]; then
    INSTALL_PREFIX=$(python -c "import sysconfig; print(sysconfig.get_config_var('prefix'))")
else
    INSTALL_PREFIX="${CONDA_PREFIX}"
fi

if [[ -z "${INSTALL_PREFIX}" ]]; then
    echo "Error: could not determine install prefix."
    echo $INSTALL_PREFIX
    exit
fi

set -e

# aria-csv
git clone https://github.com/AriaFallah/csv-parser csv-parser
mkdir -p ${CONDA_PREFIX}/include/aria-csv
pushd csv-parser
git checkout 4965c9f320d157c15bc1f5a6243de116a4caf101
cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
popd
rm -rf csv-parser

## eve
git clone https://github.com/jfalcou/eve eve
mkdir -p ${CONDA_PREFIX}/include/eve
mkdir -p ${CONDA_PREFIX}/lib
pushd eve
git checkout 3d5821fe770a62c01328b78bb55880b39b8a0a26
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DEVE_BUILD_TEST=OFF \
    -DEVE_BUILD_BENCHMARKS=OFF \
    -DEVE_BUILD_DOCUMENTATION=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
popd
rm -rf eve

## vstat
git clone https://github.com/heal-research/vstat.git
pushd vstat
git switch cpp20-eve
git checkout 4ed22ae344c6a2a6e4522ad8b2c40070dd760600
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
popd
rm -rf vstat

## fast-float
git clone https://github.com/fastfloat/fast_float.git
pushd fast_float
git checkout 7a6fe5ee799bc5583b9f8ac62966b15d669bed0f
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DFASTLOAT_TEST=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
popd
rm -rf fast_float

## pratt-parser
git clone https://github.com/foolnotion/pratt-parser-calculator.git
pushd pratt-parser-calculator
git checkout 025ba103339bb69e3b719b62f3457d5cbb9644e6
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
popd
rm -rf pratt-parser-calculator

## unordered_dense
git clone https://github.com/martinus/unordered_dense.git
pushd unordered_dense
git checkout e88dd1ce6e9dc5b3fe84a7d93ac1d7f6f7653dbf
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
popd
rm -rf unordered_dense

## cpp-sort
git clone https://github.com/Morwenn/cpp-sort.git
pushd cpp-sort
git checkout 29b593a6f9de08281bc5863ca82f6daaf55906d4
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCPPSORT_BUILD_TESTING=OFF \
    -DCPPSORT_BUILD_EXAMPLES=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
popd
rm -rf cpp-sort

## fmt (conda only includes the shared library and we want the static)
git clone https://github.com/fmtlib/fmt.git
pushd fmt
mkdir build
pushd build
git checkout e57ca2e3685b160617d3d95fcd9e789c4e06ca88
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DFMT_TEST=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
make install
popd
popd
rm -rf fmt

## quickcpplib
git clone https://github.com/ned14/quickcpplib.git
pushd quickcpplib
git checkout 5f33a37e9686b87b10f560958e7f78aff64624e4
cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
    -DCMAKE_BUILD_TYPE=Release
cmake --install build
popd
rm -rf quickcpplib

## status-code
git clone https://github.com/ned14/status-code.git
pushd status-code
git checkout 6bd2d565fd4377e16614c6c5beb495c33bfa835b
cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
    -DCMAKE_BUILD_TYPE=Release
cmake --install build
popd
rm -rf status-code

## outcome
git clone https://github.com/ned14/outcome.git
pushd outcome
git checkout 11a18c85ca7ae16af34ea309da5a0fe90024e3c3
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DPROJECT_IS_DEPENDENCY=ON \
    -DOUTCOME_BUNDLE_EMBEDDED_QUICKCPPLIB=OFF \
    -Dquickcpplib_DIR=${CONDA_PREFIX}/quickcpplib \
    -DOUTCOME_BUNDLE_EMBEDDED_STATUS_CODE=OFF \
    -Dstatus-code_DIR=${CONDA_PREFXI}/status-code \
    -DOUTCOME_ENABLE_DEPENDENCY_SMOKE_TEST=OFF \
    -DCMAKE_DISABLE_FIND_PACKAGE_Git=ON \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
popd
rm -rf outcome

## lbfgs
git clone https://github.com/foolnotion/lbfgs.git
pushd lbfgs
git checkout 0ac2cb5b8ffea5e3e71f264d8e2d37d585449512
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTING=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
popd
rm -rf lbfgs

## operon
git clone https://github.com/heal-research/operon.git
pushd operon
git switch cpp20
git checkout a64e05159f6c12fac48f1338fcf5d97dc5de9724
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_CLI_PROGRAMS=OFF \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_CXX_FLAGS="-march=x86-64-v3 -fno-math-errno" \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
    -DCMAKE_PREFIX_PATH=${CONDA_PREFIX}/lib64/cmake
cmake --build build -j -t operon_operon
cmake --install build
popd
rm -rf operon

set +e
