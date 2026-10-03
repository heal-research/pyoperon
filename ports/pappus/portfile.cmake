vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO heal-research/pappus
    REF 6b40caf098a7db6c3be8242851d3d91847bfffb1
    SHA512 c58d35de75b5b45b46b4abd70b8ec51d384e555c5ea80d031d806125724a261841c795f2fe50b47aa6ff455d0357af2f62c651a83b99dcd63016c12e090f82e8
    HEAD_REF master
)

set(VCPKG_BUILD_TYPE release)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME pappus CONFIG_PATH share/pappus)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include"
                    "${CURRENT_PACKAGES_DIR}/debug/share")

file(
  INSTALL "${SOURCE_PATH}/LICENSE"
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
  RENAME copyright)
