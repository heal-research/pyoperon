set(CMAKE_C_COMPILER clang CACHE STRING "" FORCE)
set(CMAKE_CXX_COMPILER clang++ CACHE STRING "" FORCE)

# Chainloading replaces vcpkg's default Linux toolchain; pull it in so that
# -fPIC and VCPKG_{C,CXX}_FLAGS are still applied. This file is include()d
# from <vcpkg-root>/scripts/buildsystems/vcpkg.cmake, before vcpkg's own root
# variables exist, so derive the scripts directory from the includer.
get_filename_component(_clang_linux_vcpkg_scripts "${CMAKE_PARENT_LIST_FILE}" DIRECTORY)
include("${_clang_linux_vcpkg_scripts}/../toolchains/linux.cmake")
