vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO foolnotion/infix-parser
    REF b9270d14d03c4fc54816e4eb52352e4c39927ef2
    SHA512 80e8101c16ba7849a234b6361adc5127a5147f3a9cd0fe13a7ae7efbd236a1230e5123a1d718c510313e82966cd5888431b1f7eb19683df4d2eb1eb46178d3d9
    HEAD_REF main
)

set(VCPKG_BUILD_TYPE release)

# infix-parser links lexy/fmt/fast_float PRIVATE and never find_dependency()s
# them, assuming consumers never need those symbols directly. That only
# holds if infix-parser itself is a shared library (its private deps get
# resolved at its own link time); a static build leaks them into the
# installed link interface and breaks find_package(infix-parser) for any
# consumer that doesn't also have lexy on CMAKE_PREFIX_PATH. Force shared
# regardless of the triplet.
set(VCPKG_LIBRARY_LINKAGE dynamic)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DBUILD_EXAMPLES=OFF
        -DBUILD_SHARED_LIBS=ON
)
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME infix-parser CONFIG_PATH lib/cmake/infix-parser DO_NOT_DELETE_PARENT_CONFIG_PATH)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include"
                    "${CURRENT_PACKAGES_DIR}/debug/share")

file(
  INSTALL "${SOURCE_PATH}/LICENSE"
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
  RENAME copyright)
