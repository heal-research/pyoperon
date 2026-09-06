vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO heal-research/operon
    REF 1871b61edb4dbd90c885c173e502c0c721f6b813
    SHA512 cada3dbb695d6fa5afe44b8666f2628ababc0c02965323ba71f6f505c5dacf92858a57d3793bef0cd1b88218b02bf228c45a81a027fb69b310e1f2166b095713
    HEAD_REF main
    PATCHES
        add-msvc-support.patch
)

set(VCPKG_BUILD_TYPE release)

string(COMPARE EQUAL "${VCPKG_LIBRARY_LINKAGE}" "dynamic" OPERON_SHARED_LIBS)

vcpkg_cmake_configure(
  SOURCE_PATH "${SOURCE_PATH}"
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
