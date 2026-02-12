if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_INSTALL_INCLUDEDIR include/pyoperon CACHE PATH "")
endif()

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# find_package(<package>) call for consumers to find this project
set(package pyoperon)

install(
    TARGETS pyoperon_pyoperon
    EXPORT pyoperonTargets
    RUNTIME COMPONENT pyoperon_Runtime
    LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/pyoperon"
    )

install(
    FILES "${CMAKE_SOURCE_DIR}/pyoperon/__init__.py"
    FILES "${CMAKE_SOURCE_DIR}/pyoperon/sklearn.py"
    DESTINATION "${CMAKE_INSTALL_PREFIX}/pyoperon"
    )

write_basic_package_version_file(
    "${package}ConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion
)

# Allow package maintainers to freely override the path for the configs
set(
    pyoperon_INSTALL_CMAKEDIR "${CMAKE_INSTALL_DATADIR}/${package}"
    CACHE PATH "CMake package config location relative to the install prefix"
)
mark_as_advanced(pyoperon_INSTALL_CMAKEDIR)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()
