cmake_minimum_required(VERSION 3.14)

project(pyoperonTests LANGUAGES CXX)

include(../cmake/project-is-top-level.cmake)
include(../cmake/windows-set-path.cmake)

if(PROJECT_IS_TOP_LEVEL)
  find_package(pyoperon REQUIRED)
  enable_testing()
endif()

add_executable(pyoperon_test source/pyoperon_test.cpp)
target_link_libraries(pyoperon_test PRIVATE pyoperon::pyoperon)
target_compile_features(pyoperon_test PRIVATE cxx_std_17)

add_test(NAME pyoperon_test COMMAND pyoperon_test)
windows_set_path(pyoperon_test pyoperon::pyoperon)
