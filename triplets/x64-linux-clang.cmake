set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_CMAKE_SYSTEM_NAME Linux)
set(VCPKG_CHAINLOAD_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_DIR}/clang-linux-toolchain.cmake")

# Same codegen flags as the `build-linux` preset used for the wheel itself.
set(VCPKG_C_FLAGS "-fno-math-errno -march=x86-64-v3")
set(VCPKG_CXX_FLAGS "-fsized-deallocation -fno-math-errno -march=x86-64-v3")

# Only a release wheel is shipped; skip the debug build of every port.
set(VCPKG_BUILD_TYPE release)
