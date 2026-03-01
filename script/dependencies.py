import argparse
import os
import subprocess
import sys
import platform
import time

from multiprocessing import cpu_count
from pathlib import Path


def check_installed(name):
    result = subprocess.run(
        ['cmake-package-check', name],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def run(cmd, *, verbose=False, label=""):
    if verbose:
        subprocess.run(cmd, check=True)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"\n  FAILED: {' '.join(cmd)}", flush=True)
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
            result.check_returncode()


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.1f}m"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build and install pyoperon C++ dependencies")
    parser.add_argument('-v', '--verbose', action='store_true', help="Show full cmake/build output")
    args = parser.parse_args()

    install_prefix = sys.prefix
    current_system = platform.system().lower()
    operon_build_preset = 'build-linux' if current_system == 'linux' else 'build-osx'

    default_cmake_args = [
        '-DCMAKE_C_COMPILER=clang', '-DCMAKE_CXX_COMPILER=clang++',
        '-S', '.', '-B', 'build', '-DCMAKE_BUILD_TYPE=Release',
        f'-DCMAKE_INSTALL_PREFIX={install_prefix}',
        '-DBUILD_EXAMPLES=OFF', '-DBUILD_TESTING=OFF',
    ]

    dependencies = [
        ('AriaCsvParser', 'https://github.com/AriaFallah/csv-parser', '4965c9f320d157c15bc1f5a6243de116a4caf101', default_cmake_args),
        ('Eigen3', 'https://gitlab.com/libeigen/eigen', '3.4.0', default_cmake_args),
        ('eve', 'https://github.com/jfalcou/eve', '2cb833a3e0abfe25b78ec6cff51a9b50a9da49a7', default_cmake_args),
        ('fluky', 'https://github.com/foolnotion/fluky', '19320e7499cf0958268dc11fec28a6e41ac332e4', default_cmake_args),
        ('vstat', 'https://github.com/heal-research/vstat', 'cd753d0467ccc96389fb969551eccceca97f38d1', default_cmake_args),
        ('FastFloat', 'https://github.com/fastfloat/fast_float', '50a80a73ab2ab256ba1c3bf86923ddd8b4202bc7', default_cmake_args + ['-DFASTFLOAT_TEST=OFF']),
        ('pratt-parser', 'https://github.com/foolnotion/pratt-parser-calculator', '2e0b13615c6ff1fb6381c0ac87796932b326bc89', default_cmake_args),
        ('unordered_dense', 'https://github.com/martinus/unordered_dense', '4.5.0', default_cmake_args),
        ('cpp-sort', 'https://github.com/Morwenn/cpp-sort', '1.17.0', default_cmake_args + ['-DBUILD_TESTING=0']),
        ('fmt', 'https://github.com/fmtlib/fmt', '12.1.0', default_cmake_args + ['-DCMAKE_POSITION_INDEPENDENT_CODE=ON', '-DFMT_TEST=OFF', '-DBUILD_SHARED_LIBS=OFF']),
        ('Microsoft.GSL', 'https://github.com/microsoft/GSL', 'v4.2.0', default_cmake_args + ['-DGSL_INSTALL=1', '-DGSL_TEST=0']),
        ('tl-expected', 'https://github.com/TartanLlama/expected', 'v1.3.1', default_cmake_args),
        ('lbfgs', 'https://github.com/foolnotion/lbfgs', 'a9b2a47da72a5544c8766d73bb1ef4e8d5550ca3', default_cmake_args),
        ('Taskflow', 'https://github.com/taskflow/taskflow', 'v3.9.0', default_cmake_args + ['-DTF_BUILD_EXAMPLES=OFF', '-DTF_BUILD_TESTS=OFF']),
        ('mdspan', 'https://github.com/kokkos/mdspan', '0e6a69dfe045acbb623003588a4aff844ea4b276', default_cmake_args + ['-DCMAKE_CXX_STANDARD=20', '-DMDSPAN_CXX_STANDARD=20', '-DMDSPAN_ENABLE_TESTS=OFF', '-DMDSPAN_ENABLE_BENCHMARKS=OFF']),
        ('cpptrace', 'https://github.com/jeremy-rifkin/cpptrace', 'v1.0.4', default_cmake_args + ['-DCPPTRACE_USE_EXTERNAL_ZSTD=0', '-DCPPTRACE_GET_SYMBOLS_WITH_LIBDWARF=0']),
        ('libassert', 'https://github.com/jeremy-rifkin/libassert', 'v2.2.1', default_cmake_args + ['-DLIBASSERT_USE_EXTERNAL_CPPTRACE=1', '-DBUILD_SHARED_LIBS=OFF', '-DCMAKE_POSITION_INDEPENDENT_CODE=1', '-DCMAKE_CXX_FLAGS=-fPIC']),
        ('xxHash', 'https://github.com/Cyan4973/xxHash', '7aee8d0a341bb574f7c139c769e1db115b42cc3c', default_cmake_args + ['-S', 'build/cmake']),
        ('operon', 'https://github.com/heal-research/operon', '5a1c93769ca89a34c9a4fdc8948eecb413ae1f15', default_cmake_args + ['--preset', operon_build_preset, '-DBUILD_CLI_PROGRAMS=OFF', '-DBUILD_SHARED_LIBS=OFF', '-DCMAKE_POSITION_INDEPENDENT_CODE=ON']),
    ]

    total = len(dependencies)
    working_directory = os.getcwd()
    installed = []
    skipped = []
    failed = []
    total_start = time.monotonic()

    print(f"Installing {total} dependencies into {install_prefix}\n", flush=True)

    for i, (name, url, rev, cmake_args) in enumerate(dependencies, 1):
        prefix = f"[{i}/{total}]"

        if check_installed(name):
            print(f"{prefix} {name} — already installed, skipping", flush=True)
            skipped.append(name)
            continue

        print(f"{prefix} {name} — ", end="", flush=True)
        dep_start = time.monotonic()

        try:
            repo = url.split('/')[-1].replace('.git', '')

            if not Path(repo).exists():
                run(['git', 'clone', '--quiet', url], verbose=args.verbose)

            os.chdir(repo)

            if rev:
                run(['git', 'checkout', '--quiet', rev], verbose=args.verbose)

            run(['cmake', *cmake_args], verbose=args.verbose)
            run(['cmake', '--build', 'build', '-j', str(cpu_count())], verbose=args.verbose)
            run(['cmake', '--install', 'build'], verbose=args.verbose)

            elapsed = time.monotonic() - dep_start
            print(f"done ({format_time(elapsed)})", flush=True)
            installed.append(name)

        except subprocess.CalledProcessError:
            elapsed = time.monotonic() - dep_start
            print(f"FAILED ({format_time(elapsed)})", flush=True)
            failed.append(name)
            break

        finally:
            os.chdir(working_directory)

    total_elapsed = time.monotonic() - total_start
    print(f"\n{'─' * 40}")
    print(f"Installed: {len(installed)}  Skipped: {len(skipped)}  Failed: {len(failed)}")
    print(f"Total time: {format_time(total_elapsed)}")

    if failed:
        print(f"\nFailed dependencies: {', '.join(failed)}")
        sys.exit(1)
