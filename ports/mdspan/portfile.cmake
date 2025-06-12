vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO kokkos/mdspan
    REF 0e6a69dfe045acbb623003588a4aff844ea4b276
    SHA512 17dbcc6fcd44e4c83b1c7f03d4eb3ca4349712f940dc059100dcb5c8fc2352b2b5526b98dc888eb99f88b393536aa86a859a42546408e38f3e653aab806d026a
    HEAD_REF stable
)

set(VCPKG_BUILD_TYPE release) # header-only port

vcpkg_cmake_configure(SOURCE_PATH "${SOURCE_PATH}")

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/mdspan)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/lib")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
