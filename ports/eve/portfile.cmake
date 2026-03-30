vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO jfalcou/eve
    REF 2cb833a3e0abfe25b78ec6cff51a9b50a9da49a7
    SHA512 de2673bf72d8cf4178a0e8f2500fd74318128e5b88257c57151f7c553bba202a6387cfa5db74503a0ca5949a0eb959850266fcc755e1f623f3533785c34244f0
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
