vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO greg7mdp/gtl
    REF fe33cb643009c5aaf0701a4272b5768b83a0729d
    SHA512 6967df8b977967a0c0dd6c26627208937325772874491891a33f0f25961eb55b877c65dfa972ddc707c40e950f000730e1dfddbfae663c42d06cdafa3d1b3e08
    HEAD_REF main
)

set(VCPKG_BUILD_TYPE release)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DGTL_BUILD_TESTS=OFF
        -DGTL_BUILD_EXAMPLES=OFF
        -DGTL_BUILD_BENCHMARKS=OFF
)
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME gtl CONFIG_PATH share/cmake/gtl)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include"
                    "${CURRENT_PACKAGES_DIR}/debug/share")

file(
  INSTALL "${SOURCE_PATH}/LICENSE"
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
  RENAME copyright)
