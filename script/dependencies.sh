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

export CC=gcc
export CXX=g++

# aria-csv
git clone  https://github.com/AriaFallah/csv-parser csv-parser
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
git checkout e6a52275248f48800341c5764267925aeef87169
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
git checkout 97af7dc11f6899472e49296a409467f4f163b589
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
git checkout 8f10dcc28669abbf45d95fd3af4c00c617af489c
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

## xxhash_cpp
git clone https://github.com/RedSpah/xxhash_cpp.git
pushd xxhash_cpp
git checkout 2400ea5adc1156b586ee988ea7850be45d9011b5
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
popd
rm -rf xxhash_cpp

## fmt (conda only includes the shared library and we want the static)
git clone https://github.com/fmtlib/fmt.git
pushd fmt
git checkout a33701196adfad74917046096bf5a2aa0ab0bb50
mkdir build
pushd build
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

# operon
git clone https://github.com/heal-research/operon.git
pushd operon
git switch cpp20
git checkout b9383ae91425b6c9f83303b13fa2a5e8b8e00217
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
cmake --build build -j -t operon_operon -- VERBOSE=1
cmake --install build
popd
rm -rf operon