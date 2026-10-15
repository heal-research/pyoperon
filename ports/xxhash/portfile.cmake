vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO Cyan4973/xxHash
    REF 7aee8d0a341bb574f7c139c769e1db115b42cc3c
    SHA512 6b31b54ed502b5bfc616bada33214c504e29389be73ac72549cacb6532f8b7ad3f02a851f20a7ccf13bf3220ed7941beda5b1e6bfd0fccf2d828869ef385ba04
    HEAD_REF dev
)

vcpkg_check_features(OUT_FEATURE_OPTIONS FEATURE_OPTIONS
    FEATURES xxhsum XXHASH_BUILD_XXHSUM
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}/build/cmake"
    OPTIONS ${FEATURE_OPTIONS}
)

vcpkg_cmake_install()
vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/xxHash)

if("xxhsum" IN_LIST FEATURES)
    vcpkg_copy_tools(TOOL_NAMES xxhsum AUTO_CLEAN)
endif()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

vcpkg_fixup_pkgconfig()

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
