cmake_minimum_required(VERSION 3.20)

include(cmake/prelude.cmake)

project(
    pyoperon
    VERSION 0.5.0
    DESCRIPTION "Scikit-learn module and python bindings and for the Operon library"
    HOMEPAGE_URL "https://operongp.readthedocs.io/en/latest/"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

docs_early_return()

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

# ---- Find dependencies ----
find_package(AriaCsvParser REQUIRED)
find_package(Ceres CONFIG)
find_package(Eigen3 REQUIRED)
find_package(FastFloat REQUIRED)
find_package(Threads REQUIRED)
find_package(byte-lite REQUIRED)
find_package(fluky REQUIRED)
find_package(span-lite REQUIRED)
find_package(lbfgs REQUIRED)
find_package(libassert REQUIRED)
find_package(mdspan REQUIRED)
find_package(operon REQUIRED)
find_package(outcome REQUIRED)
find_package(pratt-parser REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
# Detect the installed nanobind package and import it into CMake
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)
find_package(unordered_dense REQUIRED)
find_package(xxHash)
if(NOT xxHash_FOUND)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(xxhash IMPORTED_TARGET xxhash)
    if (NOT xxhash_FOUND)
        pkg_check_modules(xxhash IMPORTED_TARGET libxxhash)
    endif()
endif()

# ---- Declare library ----

nanobind_add_module(
    pyoperon_pyoperon
    MODULE NB_SUPPRESS_WARNINGS
    source/algorithm.cpp
    source/benchmark.cpp
    source/creator.cpp
    source/crossover.cpp
    source/dataset.cpp
    source/evaluator.cpp
    source/generator.cpp
    source/initializer.cpp
    source/mutation.cpp
    source/node.cpp
    source/non_dominated_sorter.cpp
    source/optimizer.cpp
    source/problem.cpp
    source/pset.cpp
    source/pyoperon.cpp
    source/reinserter.cpp
    source/selection.cpp
    source/tree.cpp
)
add_library(pyoperon::pyoperon ALIAS pyoperon_pyoperon)

# when operon is statically linked, dependency information will not be available
# via any external means (e.g., ldd), therefore cmake treats all PRIVATE deps as PUBLIC
# all of the below will be added to INTERFACE_LINK_LIBRARIES, therefore we try to find them
find_package(Taskflow)
find_package(cpp-sort)
find_package(fmt)
find_package(libassert)
find_package(vstat)
find_package(eve)

include(GenerateExportHeader)
generate_export_header(
    pyoperon_pyoperon
    BASE_NAME pyoperon
    EXPORT_FILE_NAME export/pyoperon/pyoperon_export.hpp
    CUSTOM_CONTENT_FROM_VARIABLE pragma_suppress_c4251
)

set_target_properties(
    pyoperon_pyoperon PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    VERSION "${PROJECT_VERSION}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
    EXPORT_NAME pyoperon
    OUTPUT_NAME pyoperon
)

target_include_directories(
    pyoperon_pyoperon ${pyoperon_warning_guard}
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)

target_include_directories(
    pyoperon_pyoperon SYSTEM
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/export>"
)

target_compile_features(pyoperon_pyoperon PUBLIC cxx_std_20)

# this will link in operon's public dependencies: fmt, ceres, etc.
target_link_libraries(pyoperon_pyoperon PRIVATE operon::operon)

if (MSVC)
    target_compile_options(pyoperon_pyoperon PRIVATE "/std:c++latest")
endif()

if (APPLE)
    set_target_properties(pyoperon_pyoperon PROPERTIES VERSION "")
endif()

# ---- Install rules ----

if(NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
endif()

# ---- Developer mode ----

if(NOT pyoperon_DEVELOPER_MODE)
  return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
  message(
      AUTHOR_WARNING
      "Developer mode is intended for developers of pyoperon"
  )
endif()

include(cmake/dev-mode.cmake)
