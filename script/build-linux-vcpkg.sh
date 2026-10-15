#!/usr/bin/env bash
# Trial: build a pyoperon manylinux wheel whose C++ dependencies come only from
# vcpkg.json (+ ports/, triplets/). Intended to run inside
# quay.io/pypa/manylinux_2_28_x86_64 as root.
#
#   VCPKG_ROOT   checked-out microsoft/vcpkg (bootstrapped here if needed)
#   PYVER        python minor version, default 3.13
set -euxo pipefail

PYVER="${PYVER:-3.13}"
: "${VCPKG_ROOT:?set VCPKG_ROOT to a microsoft/vcpkg checkout}"
export VCPKG_ROOT VCPKG_DISABLE_METRICS=1
git config --global --add safe.directory '*'
cd "$(dirname "$0")/.."
dnf install -y clang zip

[ -x "$VCPKG_ROOT/vcpkg" ] || "$VCPKG_ROOT/bootstrap-vcpkg.sh" -disableMetrics

PYBIN="$(ls -d /opt/python/cp${PYVER/./}-cp${PYVER/./}/bin)"
rm -rf build/ dist/ wheelhouse/ venv-vcpkg/ pyoperon/*.so
"$PYBIN/python" -m venv venv-vcpkg
source venv-vcpkg/bin/activate
pip install nanobind scikit-build-core auditwheel

# scikit-build-core runs CMake; the toolchain file makes CMake call
# `vcpkg install` in manifest mode (vcpkg.json) into ./vcpkg_installed.
time SKBUILD_CMAKE_ARGS="--preset=build-linux-vcpkg" \
    pip wheel . -w dist/ --no-build-isolation --no-deps -v

# infix-parser is always a shared library (see its port); make it findable.
LD_LIBRARY_PATH="$(pwd)/vcpkg_installed/x64-linux-clang/lib:${LD_LIBRARY_PATH:-}" \
    auditwheel repair dist/*.whl -w wheelhouse/
