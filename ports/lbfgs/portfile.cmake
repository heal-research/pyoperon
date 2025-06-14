vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO foolnotion/lbfgs
    REF 0ac2cb5b8ffea5e3e71f264d8e2d37d585449512
    SHA512 bf7ae25f1a38eb5ddd4237e1f422b3690cdeaecc36f4480f1417f1e164a8b32c4fd7e3d8b33ec035fc3c655a93c48646ac27f40637ac74f2e930f4839ea163ee
    HEAD_REF main
)

set(VCPKG_POLICY_SKIP_COPYRIGHT_CHECK enabled) # foolnotion/lbfgs has no license file

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup()

file(REMOVE_RECURSE ${CURRENT_PACKAGES_DIR}/debug)
