#!/usr/bin/env bash

if [[ -n "${VIRTUAL_ENV}" ]]; then
    INSTALL_PREFIX="${VIRTUAL_ENV}"
elif [[ -n "${CONDA_PREFIX}" ]]; then
    INSTALL_PREFIX="${CONDA_PREFIX}"
else
    INSTALL_PREFIX=$(python -c "import sysconfig; print(sysconfig.get_config_var('prefix'))")
fi

if [[ -z "${INSTALL_PREFIX}" ]]; then
    echo "Error: could not determine install prefix."
    echo $INSTALL_PREFIX
    exit
fi

set -e

PLATFORM=linux

if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM=osx
fi

# aria-csv
git clone https://github.com/AriaFallah/csv-parser csv-parser
mkdir -p ${INSTALL_PREFIX}/include/aria-csv
pushd csv-parser
git checkout 4965c9f320d157c15bc1f5a6243de116a4caf101
cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}
cmake --install build
popd
rm -rf csv-parser

# eigen
git clone https://gitlab.com/libeigen/eigen.git
pushd eigen
git checkout 3.4.0
cmake -S . -B build \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}
cmake --build build
cmake --install build
popd
rm -rf eigen

## eve
git clone https://github.com/jfalcou/eve eve
mkdir -p ${INSTALL_PREFIX}/include/eve
mkdir -p ${INSTALL_PREFIX}/lib
pushd eve
git checkout 3d5821fe770a62c01328b78bb55880b39b8a0a26
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DEVE_BUILD_TEST=OFF \
    -DEVE_BUILD_BENCHMARKS=OFF \
    -DEVE_BUILD_DOCUMENTATION=OFF \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}
cmake --install build
popd
rm -rf eve

## fluky
git clone https://github.com/foolnotion/fluky fluky
pushd fluky
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}
cmake --install build
popd
rm -rf fluky

## vstat
git clone https://github.com/heal-research/vstat.git
pushd vstat
git checkout 428ec2385aebf44d9ba89064b2b2ef419fd6206a
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}
cmake --install build
popd
rm -rf vstat

## fast-float
git clone https://github.com/fastfloat/fast_float.git
pushd fast_float
git checkout f476bc713fda06fbd34dc621b466745a574b3d4c
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DFASTLOAT_TEST=OFF \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}
cmake --install build
popd
rm -rf fast_float

## pratt-parser
git clone https://github.com/foolnotion/pratt-parser-calculator.git
pushd pratt-parser-calculator
git checkout 5093c67e2e642178cce1bc455f7dee8720820642
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}
cmake --install build
popd
rm -rf pratt-parser-calculator

## unordered_dense
git clone https://github.com/martinus/unordered_dense.git
pushd unordered_dense
git checkout v4.5.0
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}
cmake --install build
popd
rm -rf unordered_dense

## cpp-sort
git clone https://github.com/Morwenn/cpp-sort.git
pushd cpp-sort
git checkout 31dd8e9574dfc21e87d36794521b9e0a0fd6f5f6
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCPPSORT_BUILD_TESTING=OFF \
    -DCPPSORT_BUILD_EXAMPLES=OFF \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}
cmake --install build
popd
rm -rf cpp-sort

## fmt (conda only includes the shared library and we want the static)
git clone https://github.com/fmtlib/fmt.git
pushd fmt
mkdir build
pushd build
git checkout 11.1.3
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DFMT_TEST=OFF \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}
make install
popd
popd
rm -rf fmt

## microsoft-gsl
git clone https://github.com/microsoft/GSL.git
pushd GSL
mkdir build
pushd build
git checkout 355982daf6c54ccb11bef8a1c511be2622dec402
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    -DGSL_INSTALL=1 \
    -DGSL_TEST=0
make install
popd
popd
rm -rf GSL

# span-lite
git clone https://github.com/martinmoene/span-lite.git
pushd span-lite
git checkout 50f55c59d1b66910837313c40d11328d03447a41
mkdir build
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}
cmake --install build
popd
rm -rf span-lite

# byte-lite
git clone https://github.com/martinmoene/byte-lite.git
pushd byte-lite
git checkout dd5b3827f7cd74c1f399d1ec2c063982d3442a99
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}
cmake --install build
popd
rm -rf byte-lite

## quickcpplib
git clone https://github.com/ned14/quickcpplib.git
pushd quickcpplib
git checkout 8d5ddc873686aeb036612a4200c3bd924150c23c
cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    -DCMAKE_BUILD_TYPE=Release
cmake --install build
popd
rm -rf quickcpplib

## status-code
git clone https://github.com/ned14/status-code.git
pushd status-code
git checkout 525e324b1b85fbd1bf74046d760068b7e27b8cda
cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    -DCMAKE_BUILD_TYPE=Release
cmake --install build
popd
rm -rf status-code

## outcome
git clone https://github.com/ned14/outcome.git
pushd outcome
git checkout 744da6b7536f2850df972ab01504e3c4d9530149
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DPROJECT_IS_DEPENDENCY=ON \
    -DOUTCOME_BUNDLE_EMBEDDED_QUICKCPPLIB=OFF \
    -Dquickcpplib_DIR=${INSTALL_PREFIX}/quickcpplib \
    -DOUTCOME_BUNDLE_EMBEDDED_STATUS_CODE=OFF \
    -Dstatus-code_DIR=${CONDA_PREFIX}/status-code \
    -DOUTCOME_ENABLE_DEPENDENCY_SMOKE_TEST=OFF \
    -DCMAKE_DISABLE_FIND_PACKAGE_Git=ON \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}
cmake --install build
popd
rm -rf outcome

## lbfgs
git clone https://github.com/foolnotion/lbfgs.git
pushd lbfgs
git checkout 9fc1df5e1826c5007a233001931ab5d56f516f11
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTING=OFF \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}/lib64/cmake
cmake --install build
popd
rm -rf lbfgs

# taskflow
git clone https://github.com/taskflow/taskflow.git
pushd taskflow
git checkout v3.9.0
mkdir build
cmake -S . -B build \
    -DTF_BUILD_EXAMPLES=OFF \
    -DTF_BUILD_TESTS=OFF \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}
cmake --install build
popd
rm -rf taskflow

# mdspan
git clone https://github.com/kokkos/mdspan.git
pushd mdspan
git checkout 0e6a69dfe045acbb623003588a4aff844ea4b276
mkdir build
cmake -S . -B build \
    -DCMAKE_CXX_STANDARD=20 \
    -DMDSPAN_CXX_STANDARD=20 \
    -DMDSPAN_ENABLE_TESTS=OFF \
    -DMDSPAN_ENABLE_BENCHMARKS=OFF \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}
cmake --install build
popd
rm -rf mdspan

# cpptrace
git clone https://github.com/jeremy-rifkin/cpptrace.git
pushd cpptrace
cmake -S . -B build \
       -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
       -DCPPTRACE_USE_EXTERNAL_ZSTD=0 \
       -DCPPTRACE_GET_SYMBOLS_WITH_LIBDWARF=0
cmake --build build
cmake --install build
popd
rm -rf cpptrace

# libassert
git clone https://github.com/jeremy-rifkin/libassert.git
pushd libassert
git checkout v2.1.4
mkdir build
cmake -S . -B build \
       -DCMAKE_BUILD_TYPE=Release \
       -DLIBASSERT_USE_EXTERNAL_CPPTRACE=1 \
       -DBUILD_SHARED_LIBS=OFF \
       -DCMAKE_POSITION_INDEPENDENT_CODE=1 \
       -DCMAKE_CXX_FLAGS="-fPIC" \
       -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}
cmake --build build
cmake --install build
popd
rm -rf libassert

## operon
[ -d operon ] && rm -rf operon
git clone https://github.com/heal-research/operon.git
pushd operon
git checkout 1dbe27597ebdfb7e4fe4b421ce2daa134b51f2fd
mkdir build
cmake -S . -B build --preset build-${PLATFORM} \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_CLI_PROGRAMS=OFF \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}/lib64/cmake
cmake --build build -j -t operon_operon
cmake --install build
popd
rm -rf operon

set +e
