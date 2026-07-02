vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO foolnotion/fluky
    REF 27b7ea8e4091332770fa74f6ea09f89384c7245b
    SHA512 f6a60e797be1cb3b4a48b5a1e6a7aa9a83c57cc3056964c22d3b4930240f7df7cac3ccbe7c609db220ed01e7ad05a4b8f445e0edba0e1204ef91c98ec32ed844
    HEAD_REF main
    PATCHES
        fix-includes.patch
)

set(VCPKG_BUILD_TYPE release)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup()

file(REMOVE_RECURSE ${CURRENT_PACKAGES_DIR}/debug/include)

file(
  INSTALL "${SOURCE_PATH}/LICENSE"
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
  RENAME copyright)
