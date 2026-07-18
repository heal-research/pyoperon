vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO heal-research/operon
    REF 78409afd29e4a992b122412f3590a6ec2a776a2e
    SHA512 5f1ac42267220ad153c52fbe4e89c991811cbeb3729628176fcd108cfda99ca8bb33c2f225d2cd8351ca9d00d47d2896b8ca04ba69b638e7e0a1c91521db80af
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
