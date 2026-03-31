vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO foolnotion/lbfgs
    REF a9b2a47da72a5544c8766d73bb1ef4e8d5550ca3
    SHA512 a4e46b123da11cc347cc54c78791105fa543e5a57f33b32dea208824414a24a0ec54887e95252eb2f3ad4eaef00023a7a7859fb38c5d374a975418033e3d0d50
    HEAD_REF main
)

set(VCPKG_BUILD_TYPE release) # header-only port
set(VCPKG_POLICY_SKIP_COPYRIGHT_CHECK enabled) # foolnotion/lbfgs has no license file

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup()

file(REMOVE_RECURSE ${CURRENT_PACKAGES_DIR}/debug)
