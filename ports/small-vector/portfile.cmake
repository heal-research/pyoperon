vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO gharveymn/small_vector
    REF v0.10.2
    SHA512 726ba81479f6ba01d59f52849ce06dfc23693a13db7050679b2599557b9f2c201387f488fdc30f356203ceaaf37bd5d64f025d778db863b38da15fbde778e44b
    HEAD_REF main
)

set(VCPKG_BUILD_TYPE release) # header-only port

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DGCH_SMALL_VECTOR_ENABLE_TESTS=OFF
        -DGCH_SMALL_VECTOR_ENABLE_BENCHMARKS=OFF
)
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME small_vector CONFIG_PATH lib/cmake/small_vector)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include"
                    "${CURRENT_PACKAGES_DIR}/debug/share")

file(
  INSTALL "${SOURCE_PATH}/LICENSE"
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
  RENAME copyright)
