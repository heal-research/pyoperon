if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_INSTALL_INCLUDEDIR include/pyoperon CACHE PATH "")
endif()

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# find_package(<package>) call for consumers to find this project
set(package pyoperon)

#install(
#    DIRECTORY
#    include/
#    "${PROJECT_BINARY_DIR}/export/"
#    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
#    COMPONENT pyoperon_Development
#)
install(
    TARGETS pyoperon_pyoperon
    EXPORT pyoperonTargets
    RUNTIME COMPONENT pyoperon_Runtime
    LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/operon"
    )

install(
    FILES "${CMAKE_SOURCE_DIR}/python/__init__.py"
    FILES "${CMAKE_SOURCE_DIR}/python/sklearn.py"
    DESTINATION "${CMAKE_INSTALL_PREFIX}/operon"
    )

#install(
#    TARGETS pyoperon_pyoperon
#    EXPORT pyoperonTargets
#    RUNTIME #
#    COMPONENT pyoperon_Runtime
#    LIBRARY #
#    DESTINATION "${CMAKE_INSTALL_PREFIX}/operon"
#    COMPONENT pyoperon_Runtime
#    NAMELINK_COMPONENT pyoperon_Development
#    ARCHIVE #
#    COMPONENT python
#    DESTINATION "${CMAKE_INSTALL_PREFIX}/operon"
#    #COMPONENT pyoperon_Development
#    #INCLUDES #
#    #DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
#)

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

#install(
#    FILES cmake/install-config.cmake
#    DESTINATION "${pyoperon_INSTALL_CMAKEDIR}"
#    RENAME "${package}Config.cmake"
#    COMPONENT pyoperon_Development
#)
#
#install(
#    FILES "${PROJECT_BINARY_DIR}/${package}ConfigVersion.cmake"
#    DESTINATION "${pyoperon_INSTALL_CMAKEDIR}"
#    COMPONENT pyoperon_Development
#)
#
#install(
#    EXPORT pyoperonTargets
#    NAMESPACE pyoperon::
#    DESTINATION "${pyoperon_INSTALL_CMAKEDIR}"
#    COMPONENT pyoperon_Development
#)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()
