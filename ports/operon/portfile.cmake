vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO heal-research/operon
    REF fba8f69d21e8a0ca0d76d99e1be7b8cbfe0edb35
    SHA512 24a106ed46b15c56f38f94f719c807c37ef34ffbdc242925ea9e6223cf4e27843396060503325cd2df1af2d6a0804fb17e1f30e70f1a21377699ef46ce8a9e7e
    HEAD_REF main
    PATCHES
        add-msvc-support.patch
)

include("${VCPKG_ROOT_DIR}/ports/vcpkg-cmake/vcpkg_cmake_build.cmake")
include("${VCPKG_ROOT_DIR}/ports/vcpkg-cmake/vcpkg_cmake_install.cmake")
include("${VCPKG_ROOT_DIR}/ports/vcpkg-cmake-config/vcpkg_cmake_config_fixup.cmake")

set(VCPKG_BUILD_TYPE release)

string(COMPARE EQUAL "${VCPKG_LIBRARY_LINKAGE}" "dynamic" OPERON_SHARED_LIBS)

vcpkg_configure_cmake(
  SOURCE_PATH "${SOURCE_PATH}"
  PREFER_NINJA
  OPTIONS
        -DBUILD_TESTING=OFF
        -DBUILD_EXAMPLES=OFF
        -DBUILD_SHARED_LIBS=${OPERON_SHARED_LIBS}
        -DBUILD_CLI_PROGRAMS=OFF
        -DUSE_SINGLE_PRECISION=ON
  MAYBE_UNUSED_VARIABLES
        BUILD_TESTING
)
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME operon DO_NOT_DELETE_PARENT_CONFIG_PATH)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include"
                    "${CURRENT_PACKAGES_DIR}/debug/share")

file(
  INSTALL "${SOURCE_PATH}/LICENSE"
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
  RENAME copyright)
