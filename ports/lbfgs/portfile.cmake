vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO foolnotion/lbfgs
    REF 9e384ec6f597d4be4edc72fbf9dcadf70189df21
    SHA512 86863b2402318bad10b1a879efdb912feb805afb252421fbae323740e748fcdb9fed6371b34950cf918128c610af2216f43d59371dd5426dce7c0acd2d618e8a
    HEAD_REF main
)

set(VCPKG_POLICY_SKIP_COPYRIGHT_CHECK enabled) # foolnotion/lbfgs has no license file

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup()

file(REMOVE_RECURSE ${CURRENT_PACKAGES_DIR}/debug)
