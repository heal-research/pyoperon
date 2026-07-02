vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO foolnotion/ndsort
    REF d94c58a1eb3e08e1cd026c565ab63276ea6bc62a
    SHA512 6289f389742813a3b072262a9d9b3034cdd8baefdd00dfbaf680ca5f8be27ee0e3ac8509f1248235f3c3a2e789c729927fc595a5089fa487a450fe8d225fe461
    HEAD_REF main
)

set(VCPKG_BUILD_TYPE release)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DBUILD_EXAMPLES=OFF
)
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME ndsort CONFIG_PATH lib/cmake/ndsort)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include"
                    "${CURRENT_PACKAGES_DIR}/debug/share")

file(
  INSTALL "${SOURCE_PATH}/LICENSE"
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
  RENAME copyright)
