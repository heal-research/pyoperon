import logging
import os
import subprocess
import sys

from multiprocessing import cpu_count
from pathlib import Path


def check_installed(name):
    result = subprocess.run(['cmake-package-check', name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.NOTSET)

    install_prefix = sys.prefix
    print("INSTALL PREFIX:", install_prefix)

    default_cmake_args = [
        '-DCMAKE_C_COMPILER=clang', '-DCMAKE_CXX_COMPILER=clang++',
        '-S', '.', '-B', 'build', '-DCMAKE_BUILD_TYPE=Release', f'-DCMAKE_INSTALL_PREFIX={install_prefix}', '-DBUILD_EXAMPLES=OFF', '-DBUILD_TESTING=OFF'
    ]
    print('DEFAULT ARGS:', *default_cmake_args)

    dependencies = [
        ('AriaCsvParser', 'https://github.com/AriaFallah/csv-parser', '4965c9f320d157c15bc1f5a6243de116a4caf101', default_cmake_args),
        ('Eigen3', 'https://gitlab.com/libeigen/eigen', '3.4.0', default_cmake_args),
        ('eve', 'https://github.com/jfalcou/eve', '2cb833a3e0abfe25b78ec6cff51a9b50a9da49a7', default_cmake_args),
        ('fluky', 'https://github.com/foolnotion/fluky', '19320e7499cf0958268dc11fec28a6e41ac332e4', default_cmake_args),
        ('vstat', 'https://github.com/heal-research/vstat', '428ec2385aebf44d9ba89064b2b2ef419fd6206a', default_cmake_args),
        ('FastFloat', 'https://github.com/fastfloat/fast_float', '50a80a73ab2ab256ba1c3bf86923ddd8b4202bc7', default_cmake_args + ['-DFASTFLOAT_TEST=OFF']),
        ('pratt-parser', 'https://github.com/foolnotion/pratt-parser-calculator', '5093c67e2e642178cce1bc455f7dee8720820642', default_cmake_args),
        ('unordered_dense', 'https://github.com/martinus/unordered_dense', '4.5.0', default_cmake_args),
        ('cpp-sort', 'https://github.com/Morwenn/cpp-sort', '1.17.0', default_cmake_args + ['-DBUILD_TESTING=0']),
        ('fmt', 'https://github.com/fmtlib/fmt', '12.1.0', default_cmake_args + ['-DCMAKE_POSITION_INDEPENDENT_CODE=ON', '-DFMT_TEST=OFF', 'DBUILD_SHARED_LIBS=OFF']),
        ('Microsoft.GSL', 'https://github.com/microsoft/GSL', 'v4.2.0', default_cmake_args + ['DGSL_INSTALL=1', '-DGSL_TEST=0']),
        ('tl-expected', 'https://github.com/TartanLlama/expected', 'v1.3.1', default_cmake_args),
        ('lbfgs', 'https://github.com/foolnotion/lbfgs', '9e384ec6f597d4be4edc72fbf9dcadf70189df21', default_cmake_args),
        ('Taskflow', 'https://github.com/taskflow/taskflow', 'v3.9.0', default_cmake_args + ['-DTF_BUILD_EXAMPLES=OFF', '-DTF_BUILD_TESTS=OFF']),
        ('mdspan', 'https://github.com/kokkos/mdspan', '0e6a69dfe045acbb623003588a4aff844ea4b276', default_cmake_args + ['-DCMAKE_CXX_STANDARD=20', '-DMDSPAN_CXX_STANDARD=20', '-DMDSPAN_ENABLE_TESTS=OFF', '-DMDSPAN_ENABLE_BENCHMARKS=OFF']),
        ('cpptrace', 'https://github.com/jeremy-rifkin/cpptrace', 'v1.0.4', default_cmake_args + ['-DCPPTRACE_USE_EXTERNAL_ZSTD=0', '-DCPPTRACE_GET_SYMBOLS_WITH_LIBDWARF=0']),
        ('libassert', 'https://github.com/jeremy-rifkin/libassert', 'v2.2.1', default_cmake_args + ['-DLIBASSERT_USE_EXTERNAL_CPPTRACE=1', '-DBUILD_SHARED_LIBS=OFF', '-DCMAKE_POSITION_INDEPENDENT_CODE=1', '-DCMAKE_CXX_FLAGS=-fPIC']),
        ('xxHash', 'https://github.com/Cyan4973/xxHash', '7aee8d0a341bb574f7c139c769e1db115b42cc3c', default_cmake_args + ['-S', 'build/cmake']),
        ('operon', 'https://github.com/heal-research/operon', '6fd7ebcc3774acd91dd0c9b751de4bb6c05f6609', default_cmake_args + ['--preset', 'build-osx', '-DBUILD_CLI_PROGRAMS=OFF', '-DBUILD_SHARED_LIBS=OFF', '-DCMAKE_POSITION_INDEPENDENT_CODE=ON'])
    ]

    working_directory = os.getcwd()

    for name, url, rev, cmake_args in dependencies:
        assert name, 'name must not be empty'

        if check_installed(name):
            logger.warning(f'{name} already installed. skipping.')
            continue

        repo = url.split('/')[-1].replace('.git', '')

        if Path(repo).exists():
            logger.warning(f'directory {repo} already exists. not going to clone.')
        else:
            result = subprocess.run(['git', 'clone', url], check=True)

        # go inside and try to do the work
        os.chdir(repo)

        # might fail if not a git repo
        if rev:
            try:
                subprocess.run(['git', 'checkout', rev], check=True)
            except:
                logger.warning(f'{name} is not a git repository. trying anyway.')

        # build and install package
        subprocess.run(['cmake', *cmake_args], check=True) # generate build
        subprocess.run(['cmake', '--build', 'build', '-j', f'{cpu_count()}', '-v'], check=True) # build
        subprocess.run(['cmake', '--install', 'build'], check=True) # install

        # move out of repo
        os.chdir(working_directory)
