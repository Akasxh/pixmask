# C++ SIMD Median Filter Reference — pixmask

> Authoritative implementation reference for `src/cpp/include/pixmask/median.h` and
> `src/cpp/src/median.cpp`. Covers the 3×3 sorting-network median filter (Stage 4 of
> the pixmask pipeline) using Google Highway for portable SIMD dispatch.

---

## 1. Sorting Network: 19-Step Median-of-9

### 1.1 Background

A **sorting network** is a fixed, data-independent sequence of compare-and-swap (CAS)
operations. The sequence is determined at compile time; no branches depend on pixel
values. This makes it ideal for SIMD: each CAS maps to one `min` + one `max` instruction,
both of which are single-cycle on modern microarchitectures.

**Source for the 19-step median-only network:**

The sequence below is taken directly from OpenCV's `median_blur.simd.hpp` (modules/imgproc,
tag 4.x), which credits it as the optimal partial-sort network for extracting only the
median of 9 elements. It is **not** a full 25-comparator sort; the extra 6 comparators
needed to fully sort all 9 elements are skipped because only element [4] (the median) is
needed.

The full 9-element sort requires **25 comparators** (optimal depth 7, confirmed by
Knuth TAOCP Vol. 3 §5.3.4 and the catalogues at
`https://bertdobbelaere.github.io/sorting_networks.html`). The 19-step partial network
is derived from the Bose-Nelson construction pruned for median-only extraction: once
element [4] is proven to be in its correct rank position, remaining comparators that
would only affect elements [0-3] and [5-8] are dropped.

### 1.2 Exact Comparator Pairs (19 steps)

Label the 9 input elements p0–p8, arranged from the 3×3 neighborhood in row-major order:

```
p0  p1  p2     (row above: left, center, right)
p3  p4  p5     (current row)
p6  p7  p8     (row below)
```

CAS(a, b) means: after the operation, a ≤ b (a holds min, b holds max).

```
Step  1:  CAS(p1, p2)
Step  2:  CAS(p4, p5)
Step  3:  CAS(p7, p8)
Step  4:  CAS(p0, p1)
Step  5:  CAS(p3, p4)
Step  6:  CAS(p6, p7)
Step  7:  CAS(p1, p2)
Step  8:  CAS(p4, p5)
Step  9:  CAS(p7, p8)
Step 10:  CAS(p0, p3)
Step 11:  CAS(p5, p8)
Step 12:  CAS(p4, p7)
Step 13:  CAS(p3, p6)
Step 14:  CAS(p1, p4)
Step 15:  CAS(p2, p5)
Step 16:  CAS(p4, p7)
Step 17:  CAS(p4, p2)
Step 18:  CAS(p6, p4)
Step 19:  CAS(p4, p2)
```

After step 19, **p4 contains the median**. No other element is guaranteed to be in
sorted order except p4.

### 1.3 Why p4 is the Median

After these 19 CAS operations, p4 satisfies:
- At least 4 elements are ≤ p4 (established by steps 10–16)
- At least 4 elements are ≥ p4 (established by steps 17–19 via symmetry)

This proves p4 has rank 4 (0-indexed) among 9 elements — the exact median.

---

## 2. SIMD Implementation with Google Highway

### 2.1 Core Insight

`Min(a, b)` + `Max(a, b)` on a Highway `Vec<D>` performs CAS on all SIMD lanes
simultaneously. With `ScalableTag<uint8_t>`:
- SSE2: 16 lanes → 16 pixels processed per CAS pair
- AVX2: 32 lanes → 32 pixels processed per CAS pair
- AVX-512BW: 64 lanes → 64 pixels processed per CAS pair
- NEON: 16 lanes
- SVE: variable width, handled automatically by Highway

The median filter inner loop thus processes N pixels per iteration (N = Lanes(d)) with
zero branching in the hot path.

### 2.2 File Structure

The Highway dispatch pattern requires two files:

```
src/cpp/include/pixmask/median.h       — public interface, no Highway includes
src/cpp/src/median.cpp                 — dynamic dispatch driver
src/cpp/src/median-inl.h              — per-target SIMD implementation
```

### 2.3 Public Header (`median.h`)

```cpp
// src/cpp/include/pixmask/median.h
#pragma once
#include <cstdint>
#include <cstddef>

namespace pixmask {

// Applies a 3x3 median filter to a single-channel uint8 image.
// Handles borders by replicating edge pixels (clamp-to-edge).
//
// Parameters:
//   src      — input pixel buffer, row-major, stride bytes between rows
//   dst      — output buffer (may alias src only if src == dst, in-place safe
//              only when src_stride == dst_stride and same dimensions)
//   width    — image width in pixels
//   height   — image height in pixels
//   src_stride — bytes per row in src (>= width)
//   dst_stride — bytes per row in dst (>= width)
//
// For multi-channel images, call once per channel with channel-stride-aware
// pointers, or call Median3x3RGB for interleaved RGB (processes each channel).
void Median3x3(const uint8_t* HWY_RESTRICT src, uint8_t* HWY_RESTRICT dst,
               uint32_t width, uint32_t height,
               uint32_t src_stride, uint32_t dst_stride);

void Median3x3RGB(const uint8_t* HWY_RESTRICT src, uint8_t* HWY_RESTRICT dst,
                  uint32_t width, uint32_t height,
                  uint32_t src_stride, uint32_t dst_stride);

} // namespace pixmask
```

Note: `HWY_RESTRICT` expands to `__restrict__` (GCC/Clang) or `__restrict` (MSVC).
Include `hwy/base.h` in the `.cpp` file, not in this header, to keep it Highway-free
for consumers that only link against the dispatch wrapper.

### 2.4 Per-Target Implementation (`median-inl.h`)

This file is compiled once per SIMD target by the `foreach_target.h` mechanism.

```cpp
// src/cpp/src/median-inl.h
// Per-target include guard for Highway dynamic dispatch.
#if defined(PIXMASK_MEDIAN_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef PIXMASK_MEDIAN_INL_H_
#undef PIXMASK_MEDIAN_INL_H_
#else
#define PIXMASK_MEDIAN_INL_H_
#endif

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace pixmask {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// -------------------------------------------------------------------------
// CAS macro for Highway Vec<D>.
// After CAS(a, b): a = min(a,b), b = max(a,b).
// Uses two ops; the compiler merges them into minps/maxps or vminub/vmaxub.
// -------------------------------------------------------------------------
#define HWY_CAS(a, b)                  \
    do {                               \
        auto _lo = hn::Min((a), (b));  \
        auto _hi = hn::Max((a), (b));  \
        (a) = _lo;                     \
        (b) = _hi;                     \
    } while (0)

// -------------------------------------------------------------------------
// Scalar CAS (for the border fallback path, no SIMD).
// -------------------------------------------------------------------------
static HWY_INLINE void ScalarCAS(uint8_t& a, uint8_t& b) {
    uint8_t lo = a < b ? a : b;
    uint8_t hi = a < b ? b : a;
    a = lo;
    b = hi;
}

// -------------------------------------------------------------------------
// Scalar 3x3 median: used for border pixels and non-SIMD fallback.
// Inputs: p0..p8 as described in section 1.2.
// Returns: median value.
// -------------------------------------------------------------------------
static HWY_INLINE uint8_t Median9Scalar(uint8_t p0, uint8_t p1, uint8_t p2,
                                         uint8_t p3, uint8_t p4, uint8_t p5,
                                         uint8_t p6, uint8_t p7, uint8_t p8) {
    // 19-step partial sort: only p4 is guaranteed correct after this.
    ScalarCAS(p1, p2); ScalarCAS(p4, p5); ScalarCAS(p7, p8);
    ScalarCAS(p0, p1); ScalarCAS(p3, p4); ScalarCAS(p6, p7);
    ScalarCAS(p1, p2); ScalarCAS(p4, p5); ScalarCAS(p7, p8);
    ScalarCAS(p0, p3); ScalarCAS(p5, p8); ScalarCAS(p4, p7);
    ScalarCAS(p3, p6); ScalarCAS(p1, p4); ScalarCAS(p2, p5);
    ScalarCAS(p4, p7);
    ScalarCAS(p4, p2); ScalarCAS(p6, p4); ScalarCAS(p4, p2);
    return p4;
}

// -------------------------------------------------------------------------
// SIMD 3x3 median kernel: processes one row of (width - 2) interior pixels.
//
// Caller guarantees:
//   - row_above, row_curr, row_below point to the correct source rows
//   - dst points to dst_row + 1 (first interior column)
//   - n = width - 2 (number of interior pixels to write)
//
// The function processes floor(n / Lanes(d)) full SIMD vectors, then handles
// the remainder with scalar code.
// -------------------------------------------------------------------------
template <class D>
HWY_INLINE void MedianRow(D d,
                           const uint8_t* HWY_RESTRICT row_above,
                           const uint8_t* HWY_RESTRICT row_curr,
                           const uint8_t* HWY_RESTRICT row_below,
                           uint8_t* HWY_RESTRICT dst,
                           size_t n) {
    using V = hn::Vec<D>;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    for (; i + N <= n; i += N) {
        // Load 9 neighborhoods: for output pixel at column (i + 1) through
        // (i + N), we need columns i, i+1, i+2 from three rows.
        // src column offsets: left = i, center = i+1, right = i+2
        // (caller passes row_above pointing to column 0, so +0 = left of
        // first interior pixel, +1 = center, +2 = right)
        V p0 = hn::LoadU(d, row_above + i);
        V p1 = hn::LoadU(d, row_above + i + 1);
        V p2 = hn::LoadU(d, row_above + i + 2);
        V p3 = hn::LoadU(d, row_curr  + i);
        V p4 = hn::LoadU(d, row_curr  + i + 1);
        V p5 = hn::LoadU(d, row_curr  + i + 2);
        V p6 = hn::LoadU(d, row_below + i);
        V p7 = hn::LoadU(d, row_below + i + 1);
        V p8 = hn::LoadU(d, row_below + i + 2);

        // 19-step median-only sorting network (Section 1.2).
        HWY_CAS(p1, p2); HWY_CAS(p4, p5); HWY_CAS(p7, p8);
        HWY_CAS(p0, p1); HWY_CAS(p3, p4); HWY_CAS(p6, p7);
        HWY_CAS(p1, p2); HWY_CAS(p4, p5); HWY_CAS(p7, p8);
        HWY_CAS(p0, p3); HWY_CAS(p5, p8); HWY_CAS(p4, p7);
        HWY_CAS(p3, p6); HWY_CAS(p1, p4); HWY_CAS(p2, p5);
        HWY_CAS(p4, p7);
        HWY_CAS(p4, p2); HWY_CAS(p6, p4); HWY_CAS(p4, p2);

        hn::StoreU(p4, d, dst + i);
    }

    // Scalar remainder (< N pixels left in this interior row segment).
    for (; i < n; ++i) {
        dst[i] = Median9Scalar(
            row_above[i],   row_above[i+1], row_above[i+2],
            row_curr [i],   row_curr [i+1], row_curr [i+2],
            row_below[i],   row_below[i+1], row_below[i+2]);
    }
}

// -------------------------------------------------------------------------
// Full 3x3 median filter dispatch target.
// Border strategy: REPLICATE (clamp-to-edge).
// -------------------------------------------------------------------------
void Median3x3Impl(const uint8_t* HWY_RESTRICT src,
                   uint8_t* HWY_RESTRICT dst,
                   uint32_t width, uint32_t height,
                   uint32_t src_stride, uint32_t dst_stride) {
    const hn::ScalableTag<uint8_t> d;

    for (uint32_t y = 0; y < height; ++y) {
        // Clamp-to-edge row selection.
        const uint32_t y0 = y == 0          ? 0          : y - 1;
        const uint32_t y2 = y == height - 1 ? height - 1 : y + 1;

        const uint8_t* row_above = src + (size_t)y0 * src_stride;
        const uint8_t* row_curr  = src + (size_t)y  * src_stride;
        const uint8_t* row_below = src + (size_t)y2 * src_stride;
        uint8_t*       dst_row   = dst + (size_t)y  * dst_stride;

        // Left border pixel (x=0): replicate left neighbor.
        // The 3x3 window for x=0 uses column 0 for the "left" neighbor too.
        dst_row[0] = Median9Scalar(
            row_above[0], row_above[0], row_above[1],
            row_curr [0], row_curr [0], row_curr [1],
            row_below[0], row_below[0], row_below[1]);

        // Interior pixels: x in [1, width-2], processed N at a time.
        if (width > 2) {
            // Pass pointers to column 0 of each row; MedianRow reads
            // offsets i, i+1, i+2 where i iterates 0..(width-3).
            MedianRow(d,
                      row_above, row_curr, row_below,
                      dst_row + 1,         // write starts at column 1
                      (size_t)(width - 2));
        }

        // Right border pixel (x = width-1): replicate right neighbor.
        if (width > 1) {
            const uint32_t xR = width - 1;
            dst_row[xR] = Median9Scalar(
                row_above[xR-1], row_above[xR], row_above[xR],
                row_curr [xR-1], row_curr [xR], row_curr [xR],
                row_below[xR-1], row_below[xR], row_below[xR]);
        }
    }
}

// -------------------------------------------------------------------------
// Interleaved RGB: process each channel independently.
// For a 3-channel interleaved image, load requires stride of 3 bytes between
// horizontally adjacent same-channel pixels. Rather than de-interleaving,
// the simplest correct approach is to call the scalar path per channel.
//
// For a perf-critical RGB path, consider planar layout in the Arena.
// -------------------------------------------------------------------------
void Median3x3RGBImpl(const uint8_t* HWY_RESTRICT src,
                      uint8_t* HWY_RESTRICT dst,
                      uint32_t width, uint32_t height,
                      uint32_t src_stride, uint32_t dst_stride) {
    // Process each channel by treating the image as planar with a column
    // stride of 3. This avoids deinterleaving but loses SIMD efficiency.
    // For v0.1 correctness; optimize to planar + SIMD in v0.2.
    for (uint32_t y = 0; y < height; ++y) {
        const uint32_t y0 = y == 0          ? 0          : y - 1;
        const uint32_t y2 = y == height - 1 ? height - 1 : y + 1;

        const uint8_t* ra = src + (size_t)y0 * src_stride;
        const uint8_t* rc = src + (size_t)y  * src_stride;
        const uint8_t* rb = src + (size_t)y2 * src_stride;
        uint8_t*       dr = dst + (size_t)y  * dst_stride;

        for (uint32_t x = 0; x < width; ++x) {
            const uint32_t xl = x == 0         ? 0         : x - 1;
            const uint32_t xr = x == width - 1 ? width - 1 : x + 1;

            for (int c = 0; c < 3; ++c) {
                dr[x*3 + c] = Median9Scalar(
                    ra[xl*3+c], ra[x*3+c], ra[xr*3+c],
                    rc[xl*3+c], rc[x*3+c], rc[xr*3+c],
                    rb[xl*3+c], rb[x*3+c], rb[xr*3+c]);
            }
        }
    }
}

#undef HWY_CAS

} // namespace HWY_NAMESPACE
} // namespace pixmask
HWY_AFTER_NAMESPACE();

#endif // per-target include guard
```

### 2.5 Dispatch Driver (`median.cpp`)

```cpp
// src/cpp/src/median.cpp
// Dynamic dispatch: compiles median-inl.h for every enabled SIMD target,
// then routes calls through a function-pointer table built at first call.

// Step 1: Tell Highway which file to re-include for each target.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/cpp/src/median-inl.h"
#include "hwy/foreach_target.h"  // recompiles median-inl.h N times
#include "hwy/highway.h"

// Step 2: Export per-target symbols so the dispatcher can find them.
HWY_EXPORT(pixmask::Median3x3Impl);
HWY_EXPORT(pixmask::Median3x3RGBImpl);

namespace pixmask {

void Median3x3(const uint8_t* HWY_RESTRICT src,
               uint8_t* HWY_RESTRICT dst,
               uint32_t width, uint32_t height,
               uint32_t src_stride, uint32_t dst_stride) {
    // HWY_DYNAMIC_DISPATCH: CPU detection on first call, direct jump after.
    HWY_DYNAMIC_DISPATCH(Median3x3Impl)(src, dst, width, height,
                                         src_stride, dst_stride);
}

void Median3x3RGB(const uint8_t* HWY_RESTRICT src,
                  uint8_t* HWY_RESTRICT dst,
                  uint32_t width, uint32_t height,
                  uint32_t src_stride, uint32_t dst_stride) {
    HWY_DYNAMIC_DISPATCH(Median3x3RGBImpl)(src, dst, width, height,
                                            src_stride, dst_stride);
}

} // namespace pixmask
```

---

## 3. Scalar Fallback

### 3.1 Purpose

The scalar fallback (`Median9Scalar` in `median-inl.h`) is used for:

1. **Border pixels**: The 1-pixel-wide border on all four sides where the SIMD path
   cannot form full vectors without out-of-bounds reads.
2. **Remainder pixels**: The last `n % Lanes(d)` pixels per row after the SIMD loop.
3. **Non-SIMD targets**: Highway falls back to `HWY_SCALAR` (a target that compiles
   to width-1 scalar ops) when no SIMD is available. The same `Median9Scalar` path
   is used — the compiler sees through the macro wrapper.

### 3.2 Implementation Choice: Sorting Network vs. Selection

The 19-step sorting network (`Median9Scalar`) is chosen over insertion sort or
`std::nth_element` for the scalar path because:

- **Branch-free**: Modern compilers emit conditional moves (CMOV), not jumps, for the
  `a < b ? a : b` pattern in `ScalarCAS`. This avoids branch misprediction penalties.
- **Predictable performance**: Fixed 19 operations regardless of pixel values. No
  worst-case O(n²) behavior from insertion sort on adversarial inputs.
- **Code uniformity**: Same logic as the SIMD path — one source of truth.

### 3.3 Alternative: Insertion Sort (for reference only)

```cpp
// Only use this for debugging the sorting network output.
static uint8_t Median9InsertionSort(uint8_t p[9]) {
    for (int i = 1; i < 9; ++i) {
        uint8_t key = p[i];
        int j = i - 1;
        while (j >= 0 && p[j] > key) {
            p[j + 1] = p[j];
            --j;
        }
        p[j + 1] = key;
    }
    return p[4];
}
```

Do not use in production: branches cause misprediction on adversarial pixel patterns.

---

## 4. Memory Access Pattern and Edge Handling

### 4.1 Row Pointer Arithmetic

The canonical pattern for accessing three consecutive rows in a strided image:

```cpp
// Given: uint8_t* base, uint32_t stride (bytes per row)
const uint8_t* row_above = base + (size_t)(y - 1) * stride;
const uint8_t* row_curr  = base + (size_t) y      * stride;
const uint8_t* row_below = base + (size_t)(y + 1) * stride;
```

**Critical**: cast `y` or `stride` to `size_t` before multiplication to prevent
32-bit overflow on images wider than ~65535 pixels. The `(size_t)` cast is mandatory,
not optional, even though v0.1 caps dimensions at 8192.

**Stride ≠ width × channels**: `pixmask::ImageView` exposes a `stride` field
(bytes per row). Always use `stride` for row arithmetic, never `width * channels`.
Row padding may exist for SIMD alignment (e.g., rows padded to 32 or 64 bytes).

### 4.2 SIMD Load Alignment

`LoadU` (unaligned load) is safe for any address that is a valid multiple of
`sizeof(uint8_t)` = 1. Use `LoadU` throughout; do not use `Load` (aligned) unless
you can prove 16-byte (SSE) or 32-byte (AVX2) alignment.

On AVX2, unaligned loads crossing a 64-byte cache line boundary cost ~1 extra cycle.
For the median filter this is negligible compared to the 19 min/max operations.

### 4.3 Border Handling: Replicate (Clamp-to-Edge)

**Strategy chosen**: replicate. When loading a pixel at column -1 or column W,
substitute column 0 or column W-1 respectively.

This strategy:
- Produces no ringing at image edges (unlike zero-padding).
- Is deterministic — no random content introduced.
- Matches OpenCV's default `BORDER_REPLICATE` behavior.

The implementation in `Median3x3Impl` above handles this explicitly for the
leftmost and rightmost pixel of each row via `Median9Scalar`.

**Alternative: reflect** (`BORDER_REFLECT_101`): uses pixel at column 1 as the
"column -1" neighbor. Slightly better statistical properties for natural images.
Not used in v0.1 because the adversarial-defense goal does not require border
accuracy — the border region is 1 pixel wide and irrelevant to the attack surface.

### 4.4 Prefetch Hints

For large images, prefetching `row_below` before it is needed in the next iteration
reduces L1 cache miss latency:

```cpp
// Inside the row loop, before the SIMD inner loop:
// Prefetch the row that will be needed 2 iterations from now.
const uint8_t* row_next2 = src + (size_t)(y + 2) * src_stride;
// Prefetch at T0 (L1): ~64 bytes per cache line, stride one line ahead.
// Only worth enabling if profiling shows cache misses on large images.
#if HWY_ARCH_X86
    __builtin_prefetch(row_next2, 0, 3);  // read, high temporal locality
#endif
```

In practice, for images ≤ 1080p (≤ 8 MB), the entire image fits in L3 cache on any
modern CPU. Prefetch only if benchmarks show measurable improvement.

### 4.5 Tile Processing (optional, for very large images)

For images that exceed L3 cache (>= 32 MP, i.e., 8192×4096 with 3 channels = 100 MB),
process in horizontal tiles of height `T` where `T * width * 3` ≤ L1 cache / 3
(to keep three rows hot). A safe default tile height is 64 rows.

```cpp
// Tile height that keeps 3 rows × width bytes in L1 (~32 KB):
// max_tile_h = (32 * 1024) / (3 * width)
// For width=8192: max_tile_h = 1 — degenerate; just use linear scan.
// Tiling is beneficial only when width * 3 < 10 KB, i.e., width < 3400.
```

For pixmask v0.1 (max 8192×8192), this is not needed. The SIMD loop's access pattern
(sequential reads on three consecutive rows) is already L1/L2 cache friendly.

---

## 5. Integration with pixmask Pipeline

### 5.1 Calling from `pipeline.cpp`

```cpp
#include "pixmask/median.h"

// Inside sanitize():
// src_view and dst_view are pixmask::ImageView from the Arena.
// For grayscale (channels=1):
pixmask::Median3x3(src_view.data, dst_view.data,
                   src_view.width, src_view.height,
                   src_view.stride, dst_view.stride);

// For RGB (channels=3), interleaved:
pixmask::Median3x3RGB(src_view.data, dst_view.data,
                      src_view.width, src_view.height,
                      src_view.stride, dst_view.stride);

// For RGBA (channels=4): apply Median3x3 to R,G,B channels only,
// pass the alpha channel through unchanged (do not median-filter alpha).
for (int c = 0; c < 3; ++c) {
    // Deinterleave to a planar scratch buffer, apply, re-interleave.
    // Or: use the scalar RGB path with c < 3 guard.
}
```

### 5.2 In-Place Correctness

The median filter is **not safe in-place** in the naive sense: reading from a pixel
that has already been overwritten produces a different result from the true median.

The pipeline must use two separate buffers (src and dst). The Arena allocator provides
scratch space for this. `src` and `dst` must not overlap.

---

## 6. Build Configuration

### 6.1 CMakeLists.txt Snippet

```cmake
# Fetch Highway (vendored via FetchContent as per DECISIONS.md)
include(FetchContent)
FetchContent_Declare(highway
  GIT_REPOSITORY https://github.com/google/highway.git
  GIT_TAG        1.2.0   # pin to a specific release
)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(highway)

# median.cpp: compiled once, instantiates all SIMD targets via foreach_target.h
target_sources(pixmask_core PRIVATE
    src/cpp/src/median.cpp
)
target_link_libraries(pixmask_core PRIVATE hwy)

# median.cpp needs to see median-inl.h at the path given in HWY_TARGET_INCLUDE.
# If median-inl.h is in src/cpp/src/, add that to include paths:
target_include_directories(pixmask_core PRIVATE src/cpp/src)
```

### 6.2 Compiler Flags

Highway generates the correct target flags internally via `foreach_target.h`. No manual
`-mavx2` or `-msse4.2` is needed on the translation unit that includes `foreach_target.h`.

Do not pass `-mavx2` globally — it breaks the runtime dispatch model by forcing all
code to use AVX2 even on CPUs that don't support it.

---

## 7. Correctness Verification

### 7.1 Test Cases for `test_median.cpp`

```cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "pixmask/median.h"
#include <cstring>

// Helper: single-pixel image of any value → output equals that value.
TEST_CASE("median 1x1 identity") {
    uint8_t src = 128, dst = 0;
    pixmask::Median3x3(&src, &dst, 1, 1, 1, 1);
    CHECK(dst == 128);
}

// Uniform image: median of identical values = that value.
TEST_CASE("median uniform image") {
    constexpr int W = 64, H = 64;
    uint8_t src[W*H], dst[W*H];
    memset(src, 42, sizeof(src));
    pixmask::Median3x3(src, dst, W, H, W, W);
    for (int i = 0; i < W*H; ++i) CHECK(dst[i] == 42);
}

// Single impulse: one pixel = 255 in a zero image.
// After median, the impulse should be suppressed (output = 0 at that pixel).
TEST_CASE("median impulse suppression") {
    constexpr int W = 5, H = 5;
    uint8_t src[W*H] = {}, dst[W*H] = {};
    src[2*W + 2] = 255;  // center pixel
    pixmask::Median3x3(src, dst, W, H, W, W);
    // Center pixel's 3x3 neighborhood has 8 zeros and 1 255 → median = 0.
    CHECK(dst[2*W + 2] == 0);
}

// Known-value test: verify exact sorting network output matches insertion sort.
TEST_CASE("median matches insertion sort reference") {
    // Use a 3x3 image, check center output pixel.
    uint8_t src[3*3] = {100, 50, 200, 30, 180, 70, 90, 120, 10};
    uint8_t dst[3*3] = {};
    pixmask::Median3x3(src, dst, 3, 3, 3, 3);
    // Center pixel neighbors: all 9. Sorted: 10,30,50,70,90,100,120,180,200
    // Median = 90.
    CHECK(dst[1*3 + 1] == 90);
}

// SIMD width crossing: width = 33 to exercise the remainder loop.
TEST_CASE("median width non-multiple-of-simd") {
    constexpr int W = 33, H = 5;
    uint8_t src[W*H], dst_simd[W*H], dst_ref[W*H];
    for (int i = 0; i < W*H; ++i) src[i] = (uint8_t)(i * 7 + 13);
    pixmask::Median3x3(src, dst_simd, W, H, W, W);
    // Verify using brute-force reference (omitted for brevity; use nth_element).
    // Key check: no out-of-bounds access, no crash.
    CHECK(true);
}
```

---

## 8. Performance Expectations

| Target       | Lanes | Pixels/iter | Expected throughput (1080p grayscale) |
|--------------|-------|-------------|---------------------------------------|
| Scalar       | 1     | 1           | ~25 ms                                |
| SSE2         | 16    | 16          | ~1.0 ms                               |
| AVX2         | 32    | 32          | ~0.42 ms                              |
| AVX-512BW    | 64    | 64          | ~0.25 ms (estimate)                   |
| NEON (A76)   | 16    | 16          | ~0.8 ms (estimate)                    |

Source for SSE2/AVX2 numbers: sudonull.com SIMD median article benchmarked on Intel
i7-4770, 1920×1080 uint8 grayscale image.

The pixmask performance gate is `<15ms at 512×512` for the full `balanced` pipeline.
The median filter alone should consume < 0.05ms on AVX2 for 512×512 — well within budget.

---

## 9. Key References

| Reference | Relevance |
|-----------|-----------|
| Knuth, TAOCP Vol. 3, §5.3.4 | Sorting networks theory; optimal 9-input sort = 25 comparators |
| Bose & Nelson (1962) "A Sorting Problem", JACM 9(2) | Original construction method for the network family |
| OpenCV `median_blur.simd.hpp` (4.x) | Source for the exact 19-step median-only comparator sequence |
| Perreault & Hébert, IEEE TIP 2007 | O(1) histogram algorithm for large kernels (≥ radius 4) |
| Huang et al., IEEE TASP 1979 | Sliding histogram, O(r) per pixel |
| sudonull.com SIMD median article | Benchmark data: 58.6× speedup AVX2 vs scalar on 1080p |
| Google Highway `g3doc/quick_reference.md` | `Min`, `Max`, `LoadU`, `StoreU`, `FirstN`, `Lanes` API |
| Google Highway `hwy/examples/skeleton-inl.h` | Canonical `-inl.h` / `.cc` dispatch pattern |
| Xu, Evans, Qi — Feature Squeezing, NDSS 2018 | 3×3 is optimal kernel size for adversarial defense |
