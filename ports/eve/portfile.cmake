vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO jfalcou/eve
    REF 39a07c77527ded5aa00468d7a7daec2c7ca6caad
    SHA512 4f29e2b3129e9547baa707f70e57459f4701a983e279c9ddd68e7231e65c1eeffc788c95b5c68708cb36a2b2f524961e6eca4bdda03dbe32e77100b04552983c
    HEAD_REF main
    PATCHES
        fix-clangcl-aligned-alloc.patch
)

set(VCPKG_BUILD_TYPE release) # header-only port

vcpkg_cmake_configure(SOURCE_PATH "${SOURCE_PATH}")

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(CONFIG_PATH "lib/eve")
if(NOT EXISTS "${CURRENT_PACKAGES_DIR}/share/eve/eve-config.cmake")
    message(FATAL_ERROR "CMake config is missing")
endif()

file(REMOVE_RECURSE
    "${CURRENT_PACKAGES_DIR}/lib"
    "${CURRENT_PACKAGES_DIR}/share/doc"
)

file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
file(INSTALL "${SOURCE_PATH}/LICENSE.md" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)
