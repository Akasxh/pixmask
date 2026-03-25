cmake_minimum_required(VERSION 3.15...3.27)

project(pixmask VERSION 0.1.0 LANGUAGES CXX)

if(NOT SKBUILD)
  message(WARNING
    "This CMake file is meant to be used via scikit-build-core.\n"
    "Dev workflow:\n"
    "  pip install nanobind scikit-build-core[pyproject]\n"
    "  pip install --no-build-isolation -ve .")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================
# Google Highway — FetchContent
# ============================================================
include(FetchContent)

set(HWY_ENABLE_TESTS    OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB  OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_INSTALL  OFF CACHE BOOL "" FORCE)

FetchContent_Declare(highway
  GIT_REPOSITORY https://github.com/google/highway.git
  GIT_TAG        1.2.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(highway)

# ============================================================
# Core C++ library
# ============================================================
# POSITION_INDEPENDENT_CODE required: this static lib gets linked into a
# shared Python extension module (.so). Without -fPIC, ld fails on Linux.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(pixmask_core STATIC
  src/cpp/src/arena.cpp
  src/cpp/src/validate.cpp
  src/cpp/src/decode.cpp
  src/cpp/src/bitdepth.cpp
  src/cpp/src/median.cpp
  src/cpp/src/jpeg_roundtrip.cpp
  src/cpp/src/pipeline.cpp
)

target_include_directories(pixmask_core
  PUBLIC  src/cpp/include
  PRIVATE src/cpp/third_party
  PRIVATE src/cpp/src          # for -inl.h (Highway per-target includes)
)

# stb_image: restrict to JPEG + PNG only at compile time (DECISIONS.md §2)
# STBI_ONLY_JPEG + STBI_ONLY_PNG cause stb_image.h to internally define
# all STBI_NO_* macros. Don't also define them here to avoid redefinition warnings.
target_compile_definitions(pixmask_core PRIVATE
  STBI_ONLY_JPEG
  STBI_ONLY_PNG
)

target_link_libraries(pixmask_core PUBLIC hwy)

# ============================================================
# SIMD flags per platform
# ============================================================
if(DEFINED PIXMASK_ARCH)
  if(PIXMASK_ARCH STREQUAL "x86_64")
    target_compile_options(pixmask_core PRIVATE -mavx2 -mfma)
  elseif(PIXMASK_ARCH MATCHES "aarch64|arm64")
    target_compile_options(pixmask_core PRIVATE -march=armv8-a+simd)
  endif()
endif()

# ============================================================
# Python extension module (only when building via scikit-build-core)
# ============================================================
if(SKBUILD)
  find_package(Python 3.9
    REQUIRED COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)

  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(
    pixmask_ext
    STABLE_ABI
    NB_STATIC
    LTO
    NB_DOMAIN pixmask
    src/cpp/bindings/module.cpp
  )

  target_include_directories(pixmask_ext PRIVATE src/cpp/include)
  target_link_libraries(pixmask_ext PRIVATE pixmask_core)

  install(TARGETS pixmask_ext LIBRARY DESTINATION pixmask)

  nanobind_add_stub(
    pixmask_ext_stub
    INSTALL_TIME
    MODULE pixmask_ext
    OUTPUT pixmask_ext.pyi
    PYTHON_PATH "."
    MARKER_FILE py.typed
  )
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/py.typed" DESTINATION pixmask)
endif()

# ============================================================
# C++ unit tests (doctest, vendored header)
# ============================================================
if(BUILD_TESTING AND NOT SKBUILD)
  enable_testing()

  foreach(test_src
      src/tests/cpp/test_decode.cpp
      src/tests/cpp/test_bitdepth.cpp
      src/tests/cpp/test_median.cpp
      src/tests/cpp/test_validate.cpp
      src/tests/cpp/test_jpeg.cpp
      src/tests/cpp/test_pipeline.cpp)

    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE pixmask_core)
    target_include_directories(${test_name} PRIVATE src/cpp/third_party)
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
endif()

# ============================================================
# Fuzz targets (libFuzzer; Clang only)
# ============================================================
option(PIXMASK_FUZZ "Build libFuzzer fuzz targets" OFF)

if(PIXMASK_FUZZ)
  set(FUZZ_FLAGS -fsanitize=fuzzer,address -g -O1)

  foreach(fuzz_src
      src/tests/cpp/fuzz/fuzz_decode.cpp
      src/tests/cpp/fuzz/fuzz_validate.cpp)

    get_filename_component(fuzz_name ${fuzz_src} NAME_WE)
    add_executable(${fuzz_name} ${fuzz_src})
    target_link_libraries(${fuzz_name} PRIVATE pixmask_core)
    target_compile_options(${fuzz_name} PRIVATE ${FUZZ_FLAGS})
    target_link_options(${fuzz_name} PRIVATE ${FUZZ_FLAGS})
  endforeach()
endif()

# ============================================================
# ASan / UBSan build type
# ============================================================
set(CMAKE_CXX_FLAGS_SANITIZE
    "-g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer -fno-optimize-sibling-calls"
    CACHE STRING "Flags for Sanitize build type" FORCE)
set(CMAKE_C_FLAGS_SANITIZE
    "${CMAKE_CXX_FLAGS_SANITIZE}"
    CACHE STRING "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_SANITIZE
    "-fsanitize=address,undefined"
    CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_SANITIZE
    "${CMAKE_EXE_LINKER_FLAGS_SANITIZE}"
    CACHE STRING "" FORCE)
