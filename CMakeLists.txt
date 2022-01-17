cmake_minimum_required(VERSION 3.14)

include(cmake/prelude.cmake)

project(
    pyoperon
    VERSION 0.1.1
    DESCRIPTION "Scikit-learn module and python bindings and for the Operon library"
    HOMEPAGE_URL "https://operongp.readthedocs.io/en/latest/"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

docs_early_return()

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

# ---- Find depencendies ----
find_package(Ceres CONFIG)
find_package(Eigen3 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
find_package(fmt REQUIRED)
find_package(operon REQUIRED)
find_package(pratt-parser REQUIRED)
find_package(pybind11 REQUIRED)
find_package(robin_hood REQUIRED)
find_package(span-lite REQUIRED)

# ---- Declare library ----

pybind11_add_module(
    pyoperon_pyoperon
    MODULE
    source/algorithm.cpp
    source/creator.cpp
    source/crossover.cpp
    source/dataset.cpp
    source/eval.cpp
    source/generator.cpp
    source/initializer.cpp
    source/mutation.cpp
    source/node.cpp
    source/non_dominated_sorter.cpp
    source/problem.cpp
    source/pset.cpp
    source/pyoperon.cpp
    source/reinserter.cpp
    source/selection.cpp
    source/tree.cpp
)
add_library(pyoperon::pyoperon ALIAS pyoperon_pyoperon)

include(GenerateExportHeader)
generate_export_header(
    pyoperon_pyoperon
    BASE_NAME pyoperon
    EXPORT_FILE_NAME export/pyoperon/pyoperon_export.hpp
    CUSTOM_CONTENT_FROM_VARIABLE pragma_suppress_c4251
)

if(NOT BUILD_SHARED_LIBS)
  target_compile_definitions(pyoperon_pyoperon PUBLIC PYOPERON_STATIC_DEFINE)
endif()

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

target_compile_features(pyoperon_pyoperon PUBLIC cxx_std_17)

target_link_libraries(pyoperon_pyoperon PRIVATE
    operon::operon # this will link in operon's public dependencies: fmt, ceres, etc.
    Python3::Module
    Python3::Python
    Eigen3::Eigen
    pybind11::pybind11)

target_compile_options(pyoperon_pyoperon PRIVATE "-march=x86-64;-mavx2;-mfma;-fPIC")

target_link_options(pyoperon_pyoperon PRIVATE "-Wl,--no-undefined")

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
