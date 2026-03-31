set(CMAKE_C_COMPILER "C:/Program Files/LLVM/bin/clang-cl.exe" CACHE STRING "" FORCE)
set(CMAKE_CXX_COMPILER "C:/Program Files/LLVM/bin/clang-cl.exe" CACHE STRING "" FORCE)

# cmake auto-detects llvm-mt.exe from the LLVM directory and uses it for manifest
# embedding via vs_link_exe. llvm-mt does not generate the manifest.rc file cmake
# expects, causing rc.exe to fail with "no such file or directory". Since all
# targets use static CRT (/MT), manifests are unnecessary — disable them.
set(CMAKE_EXE_LINKER_FLAGS_INIT "/MANIFEST:NO")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "/MANIFEST:NO")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "/MANIFEST:NO")
