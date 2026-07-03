vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO heal-research/pappus
    REF 3fef62ae2c650407fbf4412f8949780f76b6bafd
    SHA512 7a388fb27421aadb9cd130cca763a0e12b7b30c3ccf07ee92686ade89f8ff9f3ca6f2e3e9cb5999a6d4faa642b30debed0a904a798589bb9e717031e7b729004
    HEAD_REF main
)

set(VCPKG_BUILD_TYPE release) # header-only port

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
