vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO heal-research/operon
    REF d4b53a63e94b628b3fc89ca117aef46c3a7a443a
    SHA512 d76f2bdd60014df5675f566954d31ebfd93d1616e10712b337b2a80db8694d2ee72be08e8c7a8faead14ca48673e8839f0a3d4f69b818785d95b4870114cd633
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
