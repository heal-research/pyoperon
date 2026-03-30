vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO foolnotion/lbfgs
    REF a9b2a47da72a5544c8766d73bb1ef4e8d5550ca3
    SHA512 1108d91f0adc378d9f68333b13817861ce8bc4527eb379632aae05881f5de6dd6014b56d32df1d4d0302d46c3ed82ca983ee85b067def8212bee0691caadf317
    HEAD_REF main
)

set(VCPKG_POLICY_SKIP_COPYRIGHT_CHECK enabled) # foolnotion/lbfgs has no license file

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup()

file(REMOVE_RECURSE ${CURRENT_PACKAGES_DIR}/debug)
