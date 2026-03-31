vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO jfalcou/eve
    REF 2cb833a3e0abfe25b78ec6cff51a9b50a9da49a7
    SHA512 99bb231d5b131b73f287bbd7ef8ab1a3cc2b7f254f4c5505442e5f99a50fe280f300217587d62e899de6423c51445c72c9f15857a231b79c774f602f5628f67e
    HEAD_REF main
)

set(VCPKG_BUILD_TYPE release) # header-only port

vcpkg_cmake_configure(SOURCE_PATH "${SOURCE_PATH}")

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(CONFIG_PATH "lib/eve-${VERSION}")
if(NOT EXISTS "${CURRENT_PACKAGES_DIR}/share/eve/eve-config.cmake")
    message(FATAL_ERROR "CMake config is missing")
endif()

file(REMOVE_RECURSE
    "${CURRENT_PACKAGES_DIR}/lib"
    "${CURRENT_PACKAGES_DIR}/share/doc"
)

file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
file(INSTALL "${SOURCE_PATH}/LICENSE.md" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)
