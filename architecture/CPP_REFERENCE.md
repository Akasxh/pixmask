# pixmask C++ Build System Reference

> Copy-pasteable patterns for CMakeLists.txt, pyproject.toml, Highway SIMD, nanobind, and stb_image.
> Grounded in DECISIONS.md + verified against upstream sources (March 2026).
> All snippets reflect the directory layout in DECISIONS.md §8.

---

## 1. CMakeLists.txt

### 1.1 Complete CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.15...3.27)

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

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
# Python (required for nanobind)
# ============================================================
find_package(Python 3.9
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# ============================================================
# nanobind — located from pip-installed package
# ============================================================
find_package(nanobind CONFIG REQUIRED)

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
  GIT_TAG        1.2.0          # pin to a release tag
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(highway)

# ============================================================
# Core C++ library
# ============================================================
add_library(pixmask_core STATIC
  src/cpp/src/validate.cpp
  src/cpp/src/decode.cpp
  src/cpp/src/bitdepth.cpp
  src/cpp/src/median.cpp
  src/cpp/src/jpeg_roundtrip.cpp
  src/cpp/src/pipeline.cpp
  src/cpp/src/pixmask.cpp
)

target_include_directories(pixmask_core
  PUBLIC  src/cpp/include
  PRIVATE src/cpp/third_party   # stb_image.h, stb_image_write.h, doctest.h
)

# stb_image: restrict to JPEG + PNG only at compile time (DECISIONS.md §2)
target_compile_definitions(pixmask_core PRIVATE
  STBI_ONLY_JPEG
  STBI_ONLY_PNG
  # Explicit belt-and-suspenders exclusions (forward-compat with new decoders)
  STBI_NO_GIF
  STBI_NO_BMP
  STBI_NO_PSD
  STBI_NO_TGA
  STBI_NO_HDR
  STBI_NO_PIC
  STBI_NO_PNM
)

target_link_libraries(pixmask_core PUBLIC hwy)

# ============================================================
# SIMD flags per platform
# ============================================================
# Highway handles target selection at runtime; these flags only affect
# which targets Highway will *compile in*. For maximum compatibility,
# do not set -march=native here — let Highway's own detection pick up
# available targets. Only set architecture baseline if you must.
#
# For wheel builds, scikit-build overrides inject PIXMASK_ARCH (see
# pyproject.toml §overrides). Here we translate that to flags.
if(DEFINED PIXMASK_ARCH)
  if(PIXMASK_ARCH STREQUAL "x86_64")
    # Compile in AVX2 + FMA support (still requires runtime detection)
    target_compile_options(pixmask_core PRIVATE -mavx2 -mfma)
  elseif(PIXMASK_ARCH MATCHES "aarch64|arm64")
    target_compile_options(pixmask_core PRIVATE -march=armv8-a+simd)
  endif()
endif()

# ============================================================
# Python extension module
# ============================================================
nanobind_add_module(
  pixmask_ext
  STABLE_ABI    # .abi3.so; valid for CPython 3.12+
  NB_STATIC     # embed libnanobind; correct for single-extension packages
  LTO           # LTO in release builds
  NB_DOMAIN pixmask

  src/cpp/bindings/module.cpp
)

target_include_directories(pixmask_ext PRIVATE src/cpp/include)
target_link_libraries(pixmask_ext PRIVATE pixmask_core)

install(TARGETS pixmask_ext LIBRARY DESTINATION pixmask)

# Type stub (runs at install time; safe with scikit-build-core)
nanobind_add_stub(
  pixmask_ext_stub
  INSTALL_TIME
  MODULE pixmask_ext
  OUTPUT pixmask_ext.pyi
  PYTHON_PATH "."
  MARKER_FILE py.typed
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/py.typed" DESTINATION pixmask)

# ============================================================
# C++ unit tests (doctest, vendored header)
# ============================================================
# Only built when invoked directly (not via scikit-build-core packaging).
if(BUILD_TESTING AND NOT SKBUILD)
  enable_testing()

  # One executable per test file keeps compile-test loops fast.
  foreach(test_src
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
# Build with: cmake -DPIXMASK_FUZZ=ON -DCMAKE_CXX_COMPILER=clang++
option(PIXMASK_FUZZ "Build libFuzzer fuzz targets" OFF)

if(PIXMASK_FUZZ)
  # libFuzzer requires -fsanitize=fuzzer on both compile and link
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
# Usage: cmake -DCMAKE_BUILD_TYPE=Sanitize ...
# (Defined as a custom config alongside Debug/Release/RelWithDebInfo)
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
```

### 1.2 How to invoke each build mode

```bash
# Normal dev install (editable, auto-rebuild on import)
pip install --no-build-isolation -ve . -Ceditable.rebuild=true

# C++ unit tests only (no Python packaging)
cmake -S . -B build -DBUILD_TESTING=ON -DSKBUILD=OFF
cmake --build build
ctest --test-dir build --output-on-failure

# ASan/UBSan
cmake -S . -B build-san -DCMAKE_BUILD_TYPE=Sanitize -DSKBUILD=OFF \
      -DBUILD_TESTING=ON -DCMAKE_CXX_COMPILER=clang++
cmake --build build-san
ctest --test-dir build-san --output-on-failure

# Fuzz targets (clang required)
cmake -S . -B build-fuzz -DPIXMASK_FUZZ=ON -DSKBUILD=OFF \
      -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug
cmake --build build-fuzz
./build-fuzz/fuzz_decode corpus/   # run fuzzer
```

---

## 2. pyproject.toml

```toml
[build-system]
requires = ["scikit-build-core>=0.10", "nanobind>=2.0.0"]
build-backend = "scikit_build_core.build"

[project]
name = "pixmask"
version = "0.1.0"
description = "Fast image sanitization for multimodal LLMs"
readme = "README.md"
requires-python = ">=3.9"
license = { file = "LICENSE" }
authors = [{ name = "Akash" }]
classifiers = [
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: C++",
    "Topic :: Security",
    "Topic :: Scientific/Engineering :: Image Processing",
]

# Runtime deps: only numpy (no OpenCV, no Pillow required)
dependencies = ["numpy>=1.24"]

[tool.scikit-build]
# Pin config format against future schema changes
minimum-version = "build-system.requires"

# Per-wheel-tag build cache; avoids rebuilds when switching Python versions
build-dir = "build/{wheel_tag}"

# .abi3.so wheels valid for CPython 3.12+ (matches STABLE_ABI in CMakeLists.txt)
wheel.py-api = "cp312"

# Explicit package listing; prevents accidentally picking up test files
wheel.packages = ["python/pixmask"]

# ============================================================
# Per-architecture SIMD injection
# ============================================================
[[tool.scikit-build.overrides]]
if.platform-machine = "^x86_64$"
cmake.args = ["-DPIXMASK_ARCH=x86_64"]

[[tool.scikit-build.overrides]]
if.platform-machine = "^aarch64$"
cmake.args = ["-DPIXMASK_ARCH=aarch64"]

[[tool.scikit-build.overrides]]
if.platform-machine = "^arm64$"
cmake.args = ["-DPIXMASK_ARCH=arm64"]

# ============================================================
# cibuildwheel
# ============================================================
[tool.cibuildwheel]
build-verbosity = 1
test-requires = ["pytest", "numpy"]
test-command = "pytest {project}/src/tests/python -x -q"
# Skip PyPy and 32-bit targets
skip = "pp* *-win32 *-manylinux_i686"

[tool.cibuildwheel.macos.environment]
MACOSX_DEPLOYMENT_TARGET = "10.14"

[[tool.cibuildwheel.overrides]]
select = "*-manylinux_x86_64 *-musllinux_x86_64"
environment = { CMAKE_ARGS = "-DPIXMASK_ARCH=x86_64" }

[[tool.cibuildwheel.overrides]]
select = "*-manylinux_aarch64 *-musllinux_aarch64"
environment = { CMAKE_ARGS = "-DPIXMASK_ARCH=aarch64" }

[[tool.cibuildwheel.overrides]]
select = "*-macosx_arm64"
environment.CMAKE_ARGS = "-DPIXMASK_ARCH=arm64"
inherit.environment = "append"
```

### 2.1 Dynamic versioning (for future use, after first tag)

Replace `version = "0.1.0"` with:

```toml
[build-system]
requires = ["scikit-build-core>=0.10", "nanobind>=2.0.0", "setuptools-scm>=8"]
build-backend = "scikit_build_core.build"

[project]
dynamic = ["version"]

[tool.scikit-build]
metadata.version.provider = "scikit_build_core.metadata.setuptools_scm"
sdist.include = ["python/pixmask/_version.py"]

[tool.setuptools_scm]
write_to = "python/pixmask/_version.py"
fallback_version = "0.0.0+unknown"
```

Then in `python/pixmask/__init__.py`:

```python
from pixmask._version import version as __version__
```

---

## 3. Highway SIMD Patterns

### 3.1 Include order and dispatch boilerplate

The `foreach_target.h` mechanism recompiles the `.cpp` file once per enabled SIMD target
by repeatedly `#include`-ing it with different `HWY_NAMESPACE` values. The rule is:

1. `#undef HWY_TARGET_INCLUDE` + `#define HWY_TARGET_INCLUDE "this_file.cpp"` — must be first.
2. `#include "hwy/foreach_target.h"` — triggers the re-include loop.
3. `#include "hwy/highway.h"` — included by each re-include pass.
4. Your implementation wrapped in `HWY_BEFORE_NAMESPACE()` / `HWY_AFTER_NAMESPACE()`.
5. `#if HWY_ONCE` section contains `HWY_EXPORT` and the dispatch wrapper — compiled exactly once.

### 3.2 Bit-depth reduction on uint8 pixels (Highway)

This is the pattern for Stage 3 (`bitdepth.cpp`). It demonstrates:
- Correct file structure for `foreach_target.h`
- `ShiftRight<N>` on uint8 (which Highway handles correctly across targets including AVX2,
  where there is no native 8-bit shift instruction — Highway uses a 16-bit shift + mask internally)
- Scalar tail loop for non-multiple-of-Lanes remainder

**`src/cpp/include/pixmask/bitdepth.h`**

```cpp
#pragma once
#include <cstddef>
#include <cstdint>

namespace pixmask {

// Reduce each uint8 pixel to `target_bits` significant bits.
// Equivalent to: pixel = (pixel >> (8 - target_bits)) << (8 - target_bits)
// target_bits must be in [1, 8].
void ReduceBitDepth(const uint8_t* HWY_RESTRICT src,
                    uint8_t*       HWY_RESTRICT dst,
                    size_t         n_pixels,
                    int            target_bits);

}  // namespace pixmask
```

**`src/cpp/src/bitdepth.cpp`**

```cpp
// MUST come before any other include that pulls in highway.h
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/cpp/src/bitdepth.cpp"
#include "hwy/foreach_target.h"  // re-includes this file per target
#include "hwy/highway.h"

#include "pixmask/bitdepth.h"

HWY_BEFORE_NAMESPACE();
namespace pixmask {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Processes a contiguous array of uint8 pixels in-place (src→dst).
// shift = 8 - target_bits.
// Highway's ShiftRight<N> on u8 vectors is portable across all targets:
// on AVX2 (no native epi8 shift), Highway uses the standard pattern of
// widening to 16-bit, shifting, masking, and narrowing back.
template <int kShift>
void ReduceBitDepthImpl(const uint8_t* HWY_RESTRICT src,
                        uint8_t*       HWY_RESTRICT dst,
                        size_t n) {
  const hn::ScalableTag<uint8_t> d;
  const size_t N = hn::Lanes(d);

  size_t i = 0;
  for (; i + N <= n; i += N) {
    auto v = hn::LoadU(d, src + i);
    // ShiftRight<kShift>: logical right shift by compile-time constant.
    // Then ShiftLeft<kShift> to zero the low bits and restore scale.
    // Combined: equivalent to (v >> kShift) << kShift = v & mask.
    // Using And+Set is equivalent and often one instruction:
    const uint8_t mask_val = static_cast<uint8_t>(0xFF << kShift);
    auto mask = hn::Set(d, mask_val);
    auto result = hn::And(v, mask);
    hn::StoreU(result, d, dst + i);
  }
  // Scalar tail
  const uint8_t smask = static_cast<uint8_t>(0xFF << kShift);
  for (; i < n; ++i) {
    dst[i] = src[i] & smask;
  }
}

// Public dispatch entry — called via HWY_DYNAMIC_DISPATCH
void ReduceBitDepthDispatch(const uint8_t* src, uint8_t* dst,
                             size_t n, int target_bits) {
  const int shift = 8 - target_bits;
  switch (shift) {
    case 0: /* 8-bit: no-op */ if (src != dst) std::memcpy(dst, src, n); break;
    case 1: ReduceBitDepthImpl<1>(src, dst, n); break;
    case 2: ReduceBitDepthImpl<2>(src, dst, n); break;
    case 3: ReduceBitDepthImpl<3>(src, dst, n); break;
    case 4: ReduceBitDepthImpl<4>(src, dst, n); break;
    case 5: ReduceBitDepthImpl<5>(src, dst, n); break;
    case 6: ReduceBitDepthImpl<6>(src, dst, n); break;
    case 7: ReduceBitDepthImpl<7>(src, dst, n); break;
    default: break;
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace pixmask
HWY_AFTER_NAMESPACE();

// ---- Compiled exactly once ----
#if HWY_ONCE
namespace pixmask {

HWY_EXPORT(ReduceBitDepthDispatch);

void ReduceBitDepth(const uint8_t* src, uint8_t* dst,
                    size_t n_pixels, int target_bits) {
  HWY_DYNAMIC_DISPATCH(ReduceBitDepthDispatch)(src, dst, n_pixels, target_bits);
}

}  // namespace pixmask
#endif  // HWY_ONCE
```

### 3.3 AVX2 no-native-8-bit-shift: what Highway does internally

AVX2 has no `_mm256_srli_epi8` instruction. The workaround that Highway (and raw intrinsic
code) applies is: operate at 16-bit width, mask out contamination from the adjacent byte.

Highway's `ShiftRight<N>(Vec256<uint8_t>)` on x86 compiles to this pattern internally:
```
// For ShiftRight<2>(v) on AVX2:
// 1. Shift the whole 256-bit register as 16-bit lanes: _mm256_srli_epi16(v, 2)
//    This shifts every 16-bit pair right by 2, including contaminating bits
//    from the high byte into the low byte of each 16-bit lane.
// 2. AND with mask 0x3F3F (= 0b0011111100111111) to clear the 2 contaminated
//    bits at the top of each low byte that leaked from the high byte.
//
// Resulting in correct logical right-shift-by-2 for all 32 uint8 lanes.
```

The `And(v, mask)` approach used in `ReduceBitDepthImpl` above is equivalent and preferred
for bit-depth reduction (single AND vs shift+shift+AND), since the goal is masking, not shifting.

### 3.4 Minimal Highway function that processes uint8_t* — template

Use this as the skeleton for any new Stage:

```cpp
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/cpp/src/YOUR_FILE.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

// Other headers go here (after the highway includes)
#include <cstdint>
#include <cstddef>
#include "pixmask/YOUR_HEADER.h"

HWY_BEFORE_NAMESPACE();
namespace pixmask {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

void YourOpDispatch(const uint8_t* HWY_RESTRICT src,
                    uint8_t*       HWY_RESTRICT dst,
                    size_t n) {
  const hn::ScalableTag<uint8_t> d;
  const size_t N = hn::Lanes(d);   // vector width in uint8 lanes; varies by target

  size_t i = 0;
  for (; i + N <= n; i += N) {
    auto v = hn::LoadU(d, src + i);
    // ... transform v ...
    hn::StoreU(v, d, dst + i);
  }
  // Scalar tail for remainder < N
  for (; i < n; ++i) {
    dst[i] = /* scalar equivalent */;
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace pixmask
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace pixmask {
HWY_EXPORT(YourOpDispatch);
void YourOp(const uint8_t* src, uint8_t* dst, size_t n) {
  HWY_DYNAMIC_DISPATCH(YourOpDispatch)(src, dst, n);
}
}  // namespace pixmask
#endif
```

### 3.5 Key Highway ops for uint8 pixel work

```cpp
namespace hn = hwy::HWY_NAMESPACE;
const hn::ScalableTag<uint8_t> d;

// Load / store (unaligned — safe everywhere)
auto v   = hn::LoadU(d, ptr);
hn::StoreU(v, d, ptr);

// Arithmetic (saturating by default for uint8)
auto mn  = hn::Min(a, b);
auto mx  = hn::Max(a, b);
auto add = hn::SaturatedAdd(a, b);
auto sub = hn::SaturatedSub(a, b);

// Bitwise
auto res = hn::And(a, b);
auto res = hn::Or(a, b);
auto res = hn::Xor(a, b);
auto res = hn::Not(a);

// Broadcast scalar into all lanes
auto m   = hn::Set(d, static_cast<uint8_t>(0xFC));

// Widen uint8 → uint16 for non-saturating arithmetic
const hn::RepartitionToWide<decltype(d)> d16;
auto lo  = hn::PromoteLowerTo(d16, v);   // low half
auto hi  = hn::PromoteUpperTo(d16, v);   // high half
// ... operate on lo/hi as uint16 ...
// Narrow back: (truncates, so ensure values fit in uint8)
auto result = hn::OrderedDemote2To(d, lo16_result, hi16_result);
```

---

## 4. nanobind Patterns

### 4.1 NB_MODULE entry point

**`src/cpp/bindings/module.cpp`**

```cpp
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "pixmask/pipeline.h"
#include "pixmask/pixmask.h"

namespace nb = nanobind;
using namespace nb::literals;

// Forward declarations — bind_* functions live in separate .cpp files
// if the module grows large, or inline here for v0.1.
void bind_sanitize(nb::module_& m);

NB_MODULE(pixmask_ext, m) {
    m.doc() = "pixmask C++ extension: fast image sanitization for multimodal LLMs";
    bind_sanitize(m);
}
```

### 4.2 Exposing a function that takes and returns a numpy ndarray

```cpp
// In bind_sanitize (called from NB_MODULE above, or inline in module.cpp)

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace nb::literals;

// Type alias for the input constraint:
// - C-contiguous (row-major)
// - uint8_t dtype
// - exactly 3 dimensions: H × W × C
// - CPU device
// Zero-copy: nanobind passes the numpy buffer pointer directly.
using RGBArray = nb::ndarray<uint8_t,
                              nb::shape<-1, -1, -1>,  // H, W, C all dynamic
                              nb::device::cpu,
                              nb::c_contig>;

void bind_sanitize(nb::module_& m) {
    m.def(
        "sanitize_array",
        [](RGBArray img,
           int bit_depth,
           int jpeg_quality_lo,
           int jpeg_quality_hi) -> nb::ndarray<nb::numpy, uint8_t, nb::ndim<3>>
        {
            const size_t H = img.shape(0);
            const size_t W = img.shape(1);
            const size_t C = img.shape(2);  // expect 3 or 4

            pixmask::SanitizeOptions opts;
            opts.bit_depth       = static_cast<uint8_t>(bit_depth);
            opts.jpeg_quality_lo = static_cast<uint8_t>(jpeg_quality_lo);
            opts.jpeg_quality_hi = static_cast<uint8_t>(jpeg_quality_hi);

            // Allocate output on the C++ heap
            const size_t n = H * W * C;
            uint8_t* out_buf = new uint8_t[n];

            {
                // Release the GIL for the duration of C++ processing.
                // nanobind guarantees the ndarray storage stays alive
                // while we hold the input array reference (img).
                nb::gil_scoped_release release;

                pixmask::ImageView view{
                    img.data(),
                    static_cast<uint32_t>(W),
                    static_cast<uint32_t>(H),
                    static_cast<uint32_t>(C),
                    static_cast<uint32_t>(W * C)  // stride
                };

                // Run the sanitization pipeline (all 6 stages)
                auto result = pixmask::sanitize_inplace(view, opts, out_buf);
                if (!result.success) {
                    delete[] out_buf;
                    // GIL is reacquired automatically on scope exit
                    throw std::runtime_error(result.error_message);
                }
            }
            // GIL reacquired here

            // Capsule owns out_buf; deleter runs when numpy array is GC'd
            nb::capsule owner(out_buf, [](void* p) noexcept {
                delete[] static_cast<uint8_t*>(p);
            });

            return nb::ndarray<nb::numpy, uint8_t, nb::ndim<3>>(
                out_buf, {H, W, C}, owner);
        },
        "img"_a,
        "bit_depth"_a         = 5,
        "jpeg_quality_lo"_a   = 70,
        "jpeg_quality_hi"_a   = 85,
        nb::call_guard<nb::gil_scoped_release>(),  // alternative: release GIL for whole call
        "Sanitize a uint8 HxWxC numpy array. Returns new array; zero-copy input."
    );
}
```

**Note on GIL release**: two valid patterns exist:

```cpp
// Pattern A: nb::call_guard on the whole function (simplest — use when
// no Python objects are touched inside the lambda)
m.def("foo", foo_cpp, nb::call_guard<nb::gil_scoped_release>());

// Pattern B: manual nb::gil_scoped_release scope inside the lambda
// (use when you need the GIL for part of the function, e.g., to raise
// a Python exception or create a new ndarray)
m.def("foo", [](RGBArray img) {
    {
        nb::gil_scoped_release release;
        // C++ work here; GIL released
    }
    // GIL reacquired — safe to construct nb::ndarray return value
    return make_output_array();
});
```

For pixmask's `sanitize`, Pattern B is required because constructing the output `nb::ndarray`
requires the GIL. Do not use `nb::call_guard` on the whole function in that case.

### 4.3 Zero-copy input path summary

```cpp
// Input: nanobind enforces C-contiguous + uint8 at the Python/C++ boundary.
// No copy is made. img.data() is the raw numpy buffer pointer.
uint8_t* ptr = img.data();   // direct pointer, valid for lifetime of img

// Output: C++ allocates the buffer; capsule transfers ownership to numpy.
uint8_t* out = new uint8_t[H * W * C];
nb::capsule owner(out, [](void* p) noexcept { delete[] (uint8_t*)p; });
return nb::ndarray<nb::numpy, uint8_t, nb::ndim<3>>(out, {H, W, C}, owner);
// When Python GC's the returned array, the capsule deleter frees out.
```

### 4.4 Handling non-contiguous or wrong-dtype input gracefully

```cpp
// nanobind throws a TypeError automatically if the constraints on RGBArray
// are not satisfied (wrong dtype, wrong contiguity, wrong device).
// To provide a cleaner message, use the unconstrained variant + manual check:

m.def("sanitize_array",
    [](nb::ndarray<> img) {
        if (img.dtype() != nb::dtype<uint8_t>())
            throw nb::type_error("expected uint8 array");
        if (img.ndim() != 3)
            throw nb::value_error("expected HxWxC array (3 dimensions)");
        if (!img.is_valid())
            throw nb::value_error("array is not valid");
        // ... proceed
    });
```

---

## 5. stb_image Patterns

### 5.1 Compile-time format restriction

These defines must appear **before** `#include "stb_image.h"` in the implementation `.cpp`
(or be set via CMake `target_compile_definitions`). Using `STBI_ONLY_*` is forward-safe:
new decoders added in future stb versions are automatically disabled.

```cpp
// In decode.cpp — before the stb include
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG

// Belt-and-suspenders: explicit exclusions for formats NOT in STBI_ONLY_*
// (handles any hypothetical future decoders not yet known at write time)
#define STBI_NO_GIF
#define STBI_NO_BMP
#define STBI_NO_PSD
#define STBI_NO_TGA
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_PNM

// Only define the implementation in ONE translation unit
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
```

For `stb_image_write.h` (JPEG encode in Stage 5):

```cpp
// In jpeg_roundtrip.cpp
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
```

### 5.2 Memory-safe stb_image usage

```cpp
#include "stb_image.h"
#include <cstdint>
#include <stdexcept>

// Decode from a memory buffer (use this, not stbi_load, to avoid file I/O)
DecodeResult decode_image(const uint8_t* input_bytes, size_t input_len,
                          int desired_channels /* 0 = as-is, 3 = RGB, 4 = RGBA */) {
    int width = 0, height = 0, channels_in_file = 0;

    // stbi_load_from_memory returns NULL on failure.
    // Ownership: caller must call stbi_image_free() on non-NULL return.
    uint8_t* pixels = stbi_load_from_memory(
        input_bytes,
        static_cast<int>(input_len),
        &width,
        &height,
        &channels_in_file,
        desired_channels
    );

    if (!pixels) {
        // stbi_failure_reason() returns a static string describing the error.
        // Never NULL when stbi_load* returns NULL.
        const char* reason = stbi_failure_reason();
        return DecodeResult{nullptr, 0, 0, 0, reason};
    }

    // Validate dimensions before proceeding (stb checks STBI_MAX_DIMENSIONS=2^24
    // but we enforce tighter limits per DECISIONS.md Stage 0)
    constexpr int kMaxDim = 8192;
    if (width <= 0 || height <= 0 ||
        width > kMaxDim || height > kMaxDim) {
        stbi_image_free(pixels);
        return DecodeResult{nullptr, 0, 0, 0, "dimension exceeds limit"};
    }

    const int actual_channels = (desired_channels > 0) ? desired_channels : channels_in_file;
    return DecodeResult{
        pixels,
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height),
        static_cast<uint32_t>(actual_channels),
        nullptr  // no error
    };
}

// ALWAYS call this; wraps free(). Do not call free() directly.
void free_decoded_image(uint8_t* pixels) {
    if (pixels) stbi_image_free(pixels);
}
```

### 5.3 JPEG encode to memory buffer (Stage 5: JPEG roundtrip)

`stbi_write_jpg_to_func` writes to an arbitrary callback instead of a file. Use it to
write directly into a `std::vector<uint8_t>` without a temporary file.

```cpp
#include "stb_image_write.h"
#include <vector>
#include <cstdint>

// The write callback appends bytes to a user-provided vector.
static void stbi_write_to_vector(void* context, void* data, int size) {
    auto* buf = static_cast<std::vector<uint8_t>*>(context);
    const auto* bytes = static_cast<const uint8_t*>(data);
    buf->insert(buf->end(), bytes, bytes + size);
}

// Encode `pixels` as JPEG at the given quality (1–100).
// Returns the JPEG bytes, or empty vector on failure.
std::vector<uint8_t> encode_jpeg(
    const uint8_t* pixels,
    int width, int height, int channels,
    int quality /* 70–85 per DECISIONS.md Stage 5 */)
{
    std::vector<uint8_t> output;
    output.reserve(static_cast<size_t>(width * height * channels / 4));  // rough guess

    const int ok = stbi_write_jpg_to_func(
        stbi_write_to_vector,
        &output,
        width,
        height,
        channels,
        pixels,
        quality
    );

    if (!ok) {
        return {};  // failure: return empty, caller checks
    }
    return output;
}

// Full JPEG roundtrip (Stage 5: encode then decode)
// Returns pixel buffer from decoded JPEG (owned by Arena or caller).
RoundtripResult jpeg_roundtrip(
    const uint8_t* pixels,
    int width, int height, int channels,
    int quality)
{
    // 1. Encode to JPEG bytes in memory
    std::vector<uint8_t> jpeg_bytes = encode_jpeg(pixels, width, height, channels, quality);
    if (jpeg_bytes.empty()) {
        return {nullptr, 0, 0, 0, "JPEG encode failed"};
    }

    // 2. Decode the JPEG back to pixels (strips all metadata; re-encodes from scratch)
    int out_w = 0, out_h = 0, out_ch = 0;
    uint8_t* out_pixels = stbi_load_from_memory(
        jpeg_bytes.data(),
        static_cast<int>(jpeg_bytes.size()),
        &out_w, &out_h, &out_ch,
        channels  // request same channel count as input
    );

    if (!out_pixels) {
        return {nullptr, 0, 0, 0, stbi_failure_reason()};
    }

    return {out_pixels,
            static_cast<uint32_t>(out_w),
            static_cast<uint32_t>(out_h),
            static_cast<uint32_t>(channels),
            nullptr};
}
```

### 5.4 stb_image error handling reference

| Scenario | Return value | Error retrieval |
|---|---|---|
| File not found | `NULL` | `stbi_failure_reason()` |
| Corrupt / truncated file | `NULL` | `stbi_failure_reason()` |
| Unsupported format | `NULL` | `stbi_failure_reason()` |
| Dimension overflow | `NULL` | `stbi_failure_reason()` |
| `malloc` failure | `NULL` | `stbi_failure_reason()` |
| Success | `uint8_t* != NULL` | n/a — check `width > 0 && height > 0` |

`stbi_failure_reason()` returns a static string; no need to free it. It is not thread-safe
(uses a static `stbi__g_failure_reason`). For multi-threaded use, read the reason string
immediately after the failed call and before any other stb operation on the same thread.

---

## 6. Libfuzz Target Pattern

```cpp
// src/tests/cpp/fuzz/fuzz_decode.cpp
// Build: cmake -DPIXMASK_FUZZ=ON -DCMAKE_CXX_COMPILER=clang++
// Run:   ./fuzz_decode -max_len=10000000 corpus/

#include <cstddef>
#include <cstdint>

// stb_image — same restrictions as production code
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_NO_GIF
#define STBI_NO_BMP
#define STBI_NO_TGA
#define STBI_FAILURE_USERMSG
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "pixmask/validate.h"   // Stage 0 validation gate

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // Stage 0: must pass validation before reaching the parser
    if (!pixmask::ValidateInput(data, size, pixmask::SanitizeOptions{})) {
        return 0;  // rejected by validator; not a parser bug
    }

    int w = 0, h = 0, ch = 0;
    uint8_t* pixels = stbi_load_from_memory(
        data, static_cast<int>(size),
        &w, &h, &ch,
        3  // request RGB
    );

    if (pixels) {
        // Ensure we can actually access the decoded buffer without crash.
        // ASan will catch out-of-bounds access here.
        volatile uint8_t sink = pixels[0] + pixels[static_cast<size_t>(w * h * 3) - 1];
        (void)sink;
        stbi_image_free(pixels);
    }
    // stbi_failure_reason() on NULL return is fine; we just discard it.

    return 0;  // always return 0 (non-zero values are reserved)
}
```

---

## 7. doctest Patterns

```cpp
// src/tests/cpp/test_bitdepth.cpp
// Compile: add to CMakeLists.txt add_executable + target_link_libraries(pixmask_core)
// Run:     ./test_bitdepth

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"   // vendored at src/cpp/third_party/doctest.h

#include "pixmask/bitdepth.h"
#include <algorithm>
#include <vector>

TEST_CASE("ReduceBitDepth: 5-bit leaves top 5 bits") {
    // All 256 distinct values → each should be rounded down to nearest 8
    std::vector<uint8_t> src(256), dst(256);
    std::iota(src.begin(), src.end(), 0);  // 0,1,2,...,255

    pixmask::ReduceBitDepth(src.data(), dst.data(), 256, 5);

    for (int i = 0; i < 256; ++i) {
        const uint8_t expected = static_cast<uint8_t>(i & 0xF8);  // 0xFF << 3
        CHECK(dst[i] == expected);
    }
}

TEST_CASE("ReduceBitDepth: edge values {0, 127, 128, 255}") {
    std::vector<uint8_t> src = {0, 127, 128, 255};
    std::vector<uint8_t> dst(4);

    pixmask::ReduceBitDepth(src.data(), dst.data(), 4, 5);

    // 5-bit mask = 0xF8 = 11111000
    CHECK(dst[0] == 0);          // 0 & 0xF8 = 0
    CHECK(dst[1] == 120);        // 127 & 0xF8 = 120
    CHECK(dst[2] == 128);        // 128 & 0xF8 = 128
    CHECK(dst[3] == 248);        // 255 & 0xF8 = 248
}

TEST_CASE("ReduceBitDepth: 8-bit is identity") {
    std::vector<uint8_t> src(256);
    std::iota(src.begin(), src.end(), 0);
    std::vector<uint8_t> dst(256);

    pixmask::ReduceBitDepth(src.data(), dst.data(), 256, 8);
    CHECK(src == dst);
}

TEST_CASE("ReduceBitDepth: handles non-multiple-of-Lanes length") {
    // Size 17 is deliberately not a multiple of any SIMD lane count
    std::vector<uint8_t> src(17, 0xFF), dst(17, 0);
    pixmask::ReduceBitDepth(src.data(), dst.data(), 17, 4);
    for (uint8_t v : dst) CHECK(v == 0xF0);
}
```

---

## 8. Risks and Known Issues

**Highway**
- `HWY_TARGET_INCLUDE` must be a path relative to the compiler's include search path (or an absolute path). If the `.cpp` file is not directly under the project root, adjust accordingly. The safest pattern: always use a path relative to the source root as it appears on the compile command line.
- `HWY_EXPORT` must appear in a non-`HWY_NAMESPACE` namespace scope, inside `#if HWY_ONCE`. Placing it inside `HWY_BEFORE_NAMESPACE` / `HWY_AFTER_NAMESPACE` causes ODR violations.
- Highway's `ShiftRight<N>` on `uint8_t` is correct and portable. The And+Set pattern in `ReduceBitDepthImpl` above is equivalent and marginally preferred for the bit-masking use case (one instruction on most targets).

**stb_image**
- GIF parser has open vulnerabilities (#1838 double-free, #1916 OOB read as of March 2026). `STBI_ONLY_JPEG` + `STBI_ONLY_PNG` eliminates this entirely — GIF code is not compiled in.
- `stbi_failure_reason()` uses a static variable; not safe to call from multiple threads simultaneously. Read it synchronously after each failed call.
- stb_image PNG #1860 (heap overflow in 16-bit format conversion) is open. Stage 0's `stbi_info_from_memory` pre-check can gate on bit depth if needed.

**nanobind**
- `NB_STATIC` (default) embeds libnanobind into the `.so`. Correct for single-extension packages. If a second extension is added (e.g., a `_gpu` module), switch both to `NB_SHARED` + ensure they share the same `NB_DOMAIN`.
- `STABLE_ABI` + `FREE_THREADED` cannot be combined. `STABLE_ABI` is silently ignored for free-threaded Python 3.13+ builds — this is correct behavior, not a bug.
- Creating an `nb::ndarray` return value requires holding the GIL. Do not call `new nb::ndarray<...>(...)` inside a `nb::gil_scoped_release` scope.

**scikit-build-core**
- `cmake.args` in `[[tool.scikit-build.overrides]]` replaces (not appends) the base `cmake.args`. If base args are needed plus arch-specific args, repeat all args in each override block.
- FetchContent for Highway during a wheel build increases build time significantly. Consider vendoring Highway source at a pinned commit in `src/cpp/third_party/highway/` and using `add_subdirectory` once build time becomes a CI concern.

**libFuzzer**
- Fuzz targets must be built with Clang; GCC does not support `-fsanitize=fuzzer`.
- `-fsanitize=fuzzer` implies `-fsanitize=address` in most configurations but specify both explicitly for clarity.
- Do not link fuzz targets into the normal test suite. Keep them gated behind `PIXMASK_FUZZ=ON`.

---

*Sources: DECISIONS.md, research/09_cpp_simd_optimization.md, research/10_nanobind_build_system.md, research/06_malformed_image_security.md, wjakob/nanobind_example (GitHub), nanobind docs (nanobind.readthedocs.io), Highway docs (github.com/google/highway), stb_image.h (nothings/stb), stb_image_write.h (nothings/stb), libFuzzer docs (llvm.org). March 2026.*
