vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO foolnotion/fluky
    REF 19320e7499cf0958268dc11fec28a6e41ac332e4
    SHA512 466f40f5aaa9b4bce66a6daeb6141c17f68972b529ac71db96545d87b50701c8a2fa7112b2f4250b98e7bf516e738a5562cfa1a18dc69a803ca21a8494755ae5
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
