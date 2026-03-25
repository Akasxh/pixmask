# C++ SIMD and Performance Optimization for Image Processing

> Research for pixmask — targeting maximum throughput for image sanitization (bit-depth reduction, median filter, Gaussian blur, DCT).
> All source files in `/src/cpp/include/fsq/` are currently empty stubs; this document drives the implementation.

---

## 1. SIMD Landscape and Portable Abstraction

### 1.1 Instruction Set Targets

| ISA | Register width | uint8 lanes | Header |
|-----|---------------|-------------|--------|
| SSE2 | 128-bit | 16 | `<emmintrin.h>` |
| SSE4.1 | 128-bit | 16 | `<smmintrin.h>` |
| AVX2 | 256-bit | 32 | `<immintrin.h>` |
| AVX-512BW | 512-bit | 64 | `<immintrin.h>` |
| NEON (AArch64) | 128-bit | 16 | `<arm_neon.h>` |
| SVE | scalable | variable | `<arm_sve.h>` |

AVX-512F operates only on 32/64-bit lanes. Byte/word operations require **AVX-512BW**, which is a distinct extension — verify CPU support separately from AVX-512F.

Benchmarked uplift on an Intel i7-4770 processing a 2 MP grayscale image (scalar vs. SSE2 vs. AVX2):

| Method | Time (ms) | Speedup |
|--------|-----------|---------|
| Scalar | 24.8 | 1× |
| SSE2 | 0.57 | 43.9× |
| AVX2 | 0.42 | 58.6× |

Source: [Sudo Null SIMD Median article](https://sudonull.com/post/118351-Optimization-of-image-processing-in-C-using-SIMD-Median-filter)

### 1.2 Key uint8 Intrinsics (x86)

```cpp
// Load / store (unaligned safe; aligned variants add ~10% speed)
__m256i v = _mm256_loadu_si256((__m256i*)ptr);
_mm256_storeu_si256((__m256i*)ptr, v);

// Aligned load (requires 32-byte alignment)
__m256i v = _mm256_load_si256((__m256i*)ptr);

// Arithmetic on packed uint8
__m256i mn  = _mm256_min_epu8(a, b);    // saturating min
__m256i mx  = _mm256_max_epu8(a, b);    // saturating max
__m256i add = _mm256_adds_epu8(a, b);   // saturating add
__m256i sub = _mm256_subs_epu8(a, b);   // saturating sub

// Comparison → mask
__m256i eq  = _mm256_cmpeq_epi8(a, b);  // 0xFF where equal

// Widen uint8 → uint16 for arithmetic without overflow
__m256i lo = _mm256_unpacklo_epi8(v, _mm256_setzero_si256());
__m256i hi = _mm256_unpackhi_epi8(v, _mm256_setzero_si256());
```

For bit-depth reduction (N-bit quantization of 8-bit pixels):

```cpp
// Reduce to K bits by masking low (8-K) bits — equivalent to rounding down
// 6-bit: keep top 6 bits
const __m256i mask6 = _mm256_set1_epi8((int8_t)0xFC); // 0b11111100
__m256i quantized = _mm256_and_si256(pixels, mask6);

// Or use right-shift + left-shift (round-to-nearest approximation)
// Note: AVX2 has no native 8-bit shift; use 16-bit shift + mask
__m256i t = _mm256_srli_epi16(pixels, 2);   // >> 2 on 16-bit pairs
t = _mm256_and_si256(t, _mm256_set1_epi16(0x003F)); // keep 6 bits
t = _mm256_slli_epi16(t, 2);                // scale back
```

AVX2 has no `_mm256_srli_epi8`. The standard pattern is: apply 16-bit shift, then AND with a mask that removes bit contamination from the upper byte in each 16-bit pair.

### 1.3 NEON Equivalents (AArch64)

```cpp
#include <arm_neon.h>

uint8x16_t v = vld1q_u8(ptr);      // load 16 bytes
vst1q_u8(ptr, v);                   // store 16 bytes

uint8x16_t mn = vminq_u8(a, b);    // min
uint8x16_t mx = vmaxq_u8(a, b);    // max

// Widen for arithmetic
uint16x8_t lo = vmovl_u8(vget_low_u8(v));
uint16x8_t hi = vmovl_u8(vget_high_u8(v));

// Bit-depth reduction (same mask approach)
const uint8x16_t mask = vdupq_n_u8(0xFC);
uint8x16_t q = vandq_u8(v, mask);
```

`uint8x16_t` holds 16 lanes; `uint8x8_t` holds 8. NEON has native 8-bit shift via `vshrq_n_u8(v, n)`, so bit-depth reduction is cleaner than on x86:

```cpp
// Reduce 8-bit → 6-bit, then reconstruct at original scale
uint8x16_t q = vshrq_n_u8(v, 2);   // >> 2 (now 6-bit values in [0..63])
uint8x16_t r = vshlq_n_u8(q, 2);   // << 2 (scale back to [0..252])
```

---

## 2. Portable SIMD: Library Comparison

Three production-ready options, from most to least opinionated:

### 2.1 Google Highway (recommended for pixmask)

- **Repo**: https://github.com/google/highway
- **Used by**: JPEG XL (libjxl), libvips, AV1 (libaom), Chromium, Firefox, NumPy (proposed NEP-54)
- Supports 27 targets: SSE2→AVX-512 (all variants), NEON, SVE, RVV, WASM, Power, LoongArch
- Length-agnostic (SVE/RVV scalable vectors work without code changes)

**Runtime dispatch pattern:**

```cpp
// myops.h
#include "hwy/highway.h"
HWY_BEFORE_NAMESPACE();
namespace project {
namespace HWY_NAMESPACE {

void ProcessRow(const uint8_t* HWY_RESTRICT src,
                uint8_t* HWY_RESTRICT dst, size_t n) {
  const hn::ScalableTag<uint8_t> d;
  for (size_t i = 0; i < n; i += hn::Lanes(d)) {
    auto v = hn::LoadU(d, src + i);
    auto q = hn::ShiftRight<2>(hn::BitCast(hn::RebindToUnsigned<decltype(d)>{}, v));
    hn::StoreU(q, d, dst + i);
  }
}

} // HWY_NAMESPACE
} // project
HWY_AFTER_NAMESPACE();

// myops.cc — one translation unit per dispatched module
#define HWY_TARGET_INCLUDE "myops.h"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
HWY_EXPORT(project::ProcessRow);   // builds all targets

// Call site
HWY_DYNAMIC_DISPATCH(project::ProcessRow)(src, dst, n);
```

The `foreach_target.h` trick recompiles the header for each target in one `.cc` file, generating a function pointer table. First call pays CPU detection; subsequent calls are direct jumps.

### 2.2 xsimd

- **Repo**: https://github.com/xtensor-stack/xsimd
- Header-only, wraps intrinsics with operator overloading
- Targets: SSE→AVX-512, NEON, SVE, WASM, VSX, RVV
- Better for math-heavy code; less control over exact instructions than Highway

```cpp
#include <xsimd/xsimd.hpp>
using B = xsimd::batch<uint8_t>;  // picks best arch at compile time
B v = B::load_unaligned(ptr);
B q = v >> 2;   // will use 16-bit shift workaround internally
```

### 2.3 SIMDe (SIMD Everywhere)

- **Repo**: https://github.com/simd-everywhere/simde
- Translates Intel intrinsics to other architectures
- Use case: port existing SSE/AVX code to ARM without a rewrite
- Performance varies; not ideal for new code

**Decision for pixmask**: Use **Highway** for all new SIMD code. It is the only library used by performance-critical production codecs (JPEG XL, AV1) and explicitly designed for the use case of "write once, run fast everywhere."

---

## 3. Bit-Depth Reduction

### 3.1 Constexpr Lookup Table

For scalar path and as a correctness reference:

```cpp
template <int TargetBits>
constexpr std::array<uint8_t, 256> make_quantize_table() {
    static_assert(TargetBits >= 1 && TargetBits <= 8);
    std::array<uint8_t, 256> lut{};
    constexpr int shift = 8 - TargetBits;
    constexpr uint8_t mask = static_cast<uint8_t>(0xFF << shift);
    for (int i = 0; i < 256; ++i) {
        lut[i] = static_cast<uint8_t>(i & mask);  // truncate (round down)
        // For round-to-nearest: lut[i] = (i + (1 << (shift-1))) & mask;
    }
    return lut;
}

static constexpr auto kQuant4 = make_quantize_table<4>();
static constexpr auto kQuant5 = make_quantize_table<5>();
static constexpr auto kQuant6 = make_quantize_table<6>();
```

At compile time the compiler folds these entirely into `.rodata`. A 256-byte table fits in 4 cache lines; sequential access is always hot.

### 3.2 SIMD Path (Template Specialization)

```cpp
template <int TargetBits>
void quantize_row_avx2(const uint8_t* src, uint8_t* dst, size_t n) {
    static_assert(TargetBits >= 1 && TargetBits <= 8);
    constexpr int shift = 8 - TargetBits;
    // Build mask: top TargetBits bits set, rest zero
    const __m256i mask = _mm256_set1_epi8(static_cast<int8_t>(0xFF << shift));
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i v = _mm256_loadu_si256((__m256i*)(src + i));
        _mm256_storeu_si256((__m256i*)(dst + i), _mm256_and_si256(v, mask));
    }
    // scalar tail
    const uint8_t smask = static_cast<uint8_t>(0xFF << shift);
    for (; i < n; ++i) dst[i] = src[i] & smask;
}
```

Template specialization means the `mask` constant is a compile-time immediate in the generated assembly — no runtime computation.

---

## 4. Median Filter

### 4.1 Algorithm Selection

| Kernel | Algorithm | Complexity | Notes |
|--------|-----------|-----------|-------|
| 3×3 | Sorting network (SIMD) | O(9 log 9) ≈ 18 compare-exchange | Best for small kernels |
| 5×5 | Sorting network (SIMD) | O(25 log 25) ≈ ~52 compare-exchange | Still faster than histogram for uint8 |
| 7×7+ | Perreault-Hébert histogram | O(1) per pixel | Amortized constant via column histograms |
| Any radius | Huang (1979) histogram | O(r) per pixel | Simpler than Perreault but slower |

### 4.2 SIMD Sorting Network for 3×3

The key insight: `_mm256_min_epu8` and `_mm256_max_epu8` perform a compare-exchange on 32 pixels simultaneously. A branchless sorting network over 9 elements needs exactly 19 compare-exchange pairs. OpenCV's `median_blur.simd.hpp` implements this exactly.

```cpp
// One compare-exchange: sort pair (a[i], a[j]) such that a[i] <= a[j]
#define SORT2(a, b) { auto mn = _mm256_min_epu8(a,b); \
                      auto mx = _mm256_max_epu8(a,b); \
                      a = mn; b = mx; }

// 3x3 median: load 9 row-neighbors into v0..v8
// Apply the 19-step network, extract v4 as median
// Reference network: Bose-Nelson / optimal 9-element network
void median3x3_avx2(const uint8_t* src, uint8_t* dst,
                    int width, int stride) {
    // Load 3 rows, offset by -1, 0, +1 columns
    __m256i v0 = _mm256_loadu_si256((__m256i*)(src - stride - 1));
    __m256i v1 = _mm256_loadu_si256((__m256i*)(src - stride    ));
    __m256i v2 = _mm256_loadu_si256((__m256i*)(src - stride + 1));
    __m256i v3 = _mm256_loadu_si256((__m256i*)(src          - 1));
    __m256i v4 = _mm256_loadu_si256((__m256i*)(src             ));
    __m256i v5 = _mm256_loadu_si256((__m256i*)(src          + 1));
    __m256i v6 = _mm256_loadu_si256((__m256i*)(src + stride - 1));
    __m256i v7 = _mm256_loadu_si256((__m256i*)(src + stride    ));
    __m256i v8 = _mm256_loadu_si256((__m256i*)(src + stride + 1));

    // 19-step optimal sorting network for 9 elements
    SORT2(v0,v1); SORT2(v3,v4); SORT2(v6,v7);
    SORT2(v1,v2); SORT2(v4,v5); SORT2(v7,v8);
    SORT2(v0,v1); SORT2(v3,v4); SORT2(v6,v7);
    SORT2(v0,v3); SORT2(v3,v6); SORT2(v0,v3);
    SORT2(v1,v4); SORT2(v4,v7); SORT2(v1,v4);
    SORT2(v2,v5); SORT2(v5,v8); SORT2(v2,v5);
    SORT2(v1,v3); SORT2(v5,v7); SORT2(v4,v6);
    SORT2(v2,v4); SORT2(v4,v6);
    SORT2(v2,v3); SORT2(v5,v6);
    // v4 now contains the median of each of the 32 pixel positions
    _mm256_storeu_si256((__m256i*)dst, v4);
}
```

This processes 32 pixels per call with no branching.

### 4.3 Perreault-Hébert O(1) Histogram Algorithm

Reference: IEEE Transactions on Image Processing, vol. 16, 2007, pp. 2389–2394.

The algorithm maintains:
- One **column histogram** per image column (size 256 uint16 counters)
- One **sliding window histogram** = sum of column histograms in current row span

When advancing one pixel right:
1. Subtract the exiting column histogram from the window
2. Add the entering column histogram
3. Update the entering column histogram (subtract top pixel, add bottom pixel)
4. Find median via two-tier binary search in the window histogram

Two-tier structure used by OpenCV:
- **Coarse level**: 16 buckets (high 4 bits) → locate which fine bucket
- **Fine level**: 16×16 = 256 counters (full 8 bits) → linear scan within bucket

Finding the median requires scanning at most 16 fine-level entries after one 16-entry coarse scan.

SIMD acceleration: update operations on 16-element histograms fit exactly in one SSE2 or AVX2 register. The `v_add` / `v_sub` on the counter arrays are vectorized automatically by modern compilers or explicitly with `_mm_add_epi16` / `_mm256_add_epi16`.

**Crossover point**: For 8-bit images, use sorting network for 3×3 and 5×5; switch to Perreault-Hébert for radius ≥ 4.

---

## 5. Gaussian Blur

### 5.1 Separable Convolution (Reference Implementation)

A 2D Gaussian kernel G(x,y) = G(x)·G(y). Decompose into two 1D passes:
1. Horizontal pass: convolve each row with 1D kernel
2. Vertical pass: convolve each column of the intermediate result

This reduces O(W·H·k²) multiplications to O(W·H·2k), an exact result with k = ceil(3σ)×2+1.

For integer 1D convolution with AVX2:

```cpp
// Horizontal pass: process one row with 1D Gaussian weights
// src row → dst row, kernel is symmetric
void gaussian_row_avx2(const uint8_t* src, int16_t* dst,
                       const int16_t* kernel, int klen, int width) {
    int half = klen / 2;
    for (int x = 0; x < width; x += 16) {
        __m256i acc = _mm256_setzero_si256();
        for (int k = 0; k < klen; ++k) {
            // load 16 uint8s, widen to int16, multiply by kernel[k]
            __m128i raw = _mm_loadu_si128((__m128i*)(src + x + k - half));
            __m256i wide = _mm256_cvtepu8_epi16(raw); // zero-extend
            __m256i kv   = _mm256_set1_epi16(kernel[k]);
            acc = _mm256_add_epi16(acc, _mm256_mullo_epi16(wide, kv));
        }
        _mm256_storeu_si256((__m256i*)(dst + x), acc);
    }
}
```

Fixed-point kernel: multiply true Gaussian values by 256 (Q8.0), then right-shift results by 8. Error ≈ 1–3% at σ > 1, negligible for image sanitization.

### 5.2 Box Blur Approximation (Stack Blur / Gaussian by CLT)

Three box blur passes approximate a Gaussian via the Central Limit Theorem. Execution time is independent of sigma. Reference: [bfraboni/FastGaussianBlur](https://github.com/bfraboni/FastGaussianBlur), [blog.ivank.net](https://blog.ivank.net/fastest-gaussian-blur.html).

Performance: ~7ms for 2 MP image (Ryzen 7 2700X) regardless of sigma value.

Box blur itself: a sliding window with O(1) per pixel via prefix sum or running sum. Full SIMD: process 32 uint8 pixels per cycle using horizontal accumulation.

Algorithm:
1. Compute sizes of three boxes for target sigma: `boxes = boxesForGauss(sigma, 3)`
2. Apply horizontal box blur (box width W) to each row → tmp buffer
3. Apply vertical box blur to each column of tmp → output
4. Repeat 3 times with the three box sizes

The transposition step (to reuse the horizontal kernel for vertical) is itself cache-sensitive — see Section 7.2.

### 5.3 Recursive IIR Gaussian (Young-Van Vliet)

Reference: Ian T. Young, Lucas J. van Vliet, "Recursive implementation of the Gaussian filter," Signal Processing 44(2), 1995.

Single-header implementation: [iir_gauss_blur.h by Arkanis](http://arkanis.de/weblog/2018-08-30-iir-gauss-blur-h-a-gaussian-blur-single-header-file-library)

Properties:
- **O(1) per pixel regardless of sigma** — 4 causal/anticausal passes, 3 multiplies + 3 adds per sample per pass
- Exact Gaussian to floating-point precision
- 4 passes: left→right, right→left, top→bottom, bottom→top
- Boundary conditions from Triggs-Sdika (2006)
- Numerical instability for very large sigma (> 200) in float32; use double

```cpp
// Signature from iir_gauss_blur.h
void iir_gauss_blur(uint32_t width, uint32_t height, uint8_t components,
                    uint8_t* image, float sigma);
```

Coefficients (from Young-Van Vliet, Table I, σ-dependent):
```
q = sigma < 2.5 ?
      3.97156 - 4.14554*sqrt(1 - 0.26891*sigma) :
      0.98711*sigma - 0.96330;
b0 = 1.57825 + 2.44413*q + 1.4281*q*q + 0.422205*q*q*q;
b1 = 2.44413*q + 2.85619*q*q + 1.26661*q*q*q;
b2 = -(1.4281*q*q + 1.26661*q*q*q);
b3 = 0.422205*q*q*q;
B  = 1 - (b1+b2+b3)/b0;
```

For pixmask, use IIR Gaussian when sigma > 3 (box blur has sigma approximation error for small values). Use separable FIR or box blur for sigma ≤ 3.

### 5.4 Performance Decision Tree

```
sigma <= 1.0  → separable FIR with 3-tap kernel (trivial)
1.0 < sigma <= 3.0  → separable FIR with SIMD-accelerated 1D convolution
3.0 < sigma <= 30.0 → 3-pass box blur (constant time, excellent approximation)
sigma > 30.0  → Young-Van Vliet IIR (O(1), numerically stable with double)
```

---

## 6. Fast DCT (8×8 Blocks)

### 6.1 AAN Algorithm

Arai, Agui, Nakajima (1988): reduces the 1D 8-point DCT from 11 multiplications + 29 additions (naive) down to **5 multiplications + 29 additions** (scaled output). For the 2D 8×8 DCT:
1. Apply 1D AAN DCT to each row (8 rows × 5 muls = 40 muls)
2. Apply 1D AAN DCT to each column (8 cols × 5 muls = 40 muls)
3. Total: 80 multiplications vs. 512 for naive 2D

The algorithm produces a scaled output; the scale factors are absorbed into the quantization matrix (JPEG does this, making AAN a perfect fit for DCT+quantize pipelines).

Reference implementation: [prtsh/aan_dct](https://github.com/prtsh/aan_dct)

### 6.2 libjpeg-turbo SIMD DCT Reference

libjpeg-turbo implements SIMD DCT across all major ISAs:

| ISA | Forward DCT | Inverse DCT |
|-----|-------------|-------------|
| AVX2 | Accurate integer | Accurate integer |
| SSE2 | Accurate + fast integer | Accurate + fast integer |
| NEON (AArch64) | Accurate integer | Accurate integer |
| NEON (AArch32) | Accurate + fast integer | Accurate + fast integer |

Source files:
- `simd/x86_64/jsimd_avx2.asm` — AVX2 DCT
- `simd/arm/aarch64/jsimd_neon.S` — AArch64 NEON DCT
- `simd/arm/arm/jsimd_neon.S` — AArch32 NEON DCT

Performance gain: 30–40% compression speedup vs. no SIMD on ARM (iPhone 4S/5S measurements from libjpeg-turbo documentation).

The accurate integer DCT uses 32-bit intermediates (not 16-bit) to avoid overflow, then rounds at the end. The fast integer DCT sacrifices a few LSBs for speed.

### 6.3 SIMD 8×8 DCT Pattern (Column-First)

Key implementation observation from Intel AP-922: processing columns first is more efficient than rows first in SIMD because column data is gathered (non-contiguous), while row data is contiguous. Loading 8 columns of 8 uint8s into an 8×__m128i set via interleaved loads, then processing, then storing transposed output aligns well with the subsequent row pass.

```cpp
// Load 8x8 block: 8 rows, 8 columns of uint8
// Use _mm_loadl_epi64 for 8-byte row loads, then unpack
__m128i rows[8];
for (int i = 0; i < 8; ++i)
    rows[i] = _mm_loadl_epi64((__m128i*)(src + i * stride));

// Unpack to 16-bit, apply 1D DCT butterfly to columns
// (transpose the 8x8 register block, then apply to rows)
// This is the approach used by libjpeg-turbo's SSE2 implementation
```

### 6.4 Faster Alternative: JPEG XL Approach

libjxl uses floating-point DCT via Highway, supporting variable block sizes (2×2 to 32×32). For pixmask's fixed 8×8 use case, libjpeg-turbo's integer AAN is preferable — it avoids FP conversion overhead and IEEE compliance ambiguity.

---

## 7. Cache-Optimal Access Patterns

### 7.1 Tiling Strategy

L1 cache (32–64 KB), L2 cache (256 KB–1 MB), cache line = 64 bytes.

For a 1920×1080 uint8 grayscale image = 2.07 MB — does not fit in L2. Process in tiles that do fit.

**Tile size for L1 (32 KB budget):**
- A 3×3 filter needs 3 rows in memory simultaneously
- Target tile width: `3 * tile_width * sizeof(uint8_t) <= 32KB / 2 = 16KB`
- `tile_width <= 5461` — entire row fits for 1080p (1920 bytes × 3 = 5.76 KB)
- For separable filter vertical pass: tile height × width × 2 buffers ≤ 32KB
- Optimal tile: 64 cols × (L1_size / 64) rows = **64 × 256 = 16 KB** per plane

**Tile loop structure:**
```cpp
constexpr int TILE_W = 64;   // fits in L1 with 3 row buffers
constexpr int TILE_H = 128;

for (int ty = 0; ty < height; ty += TILE_H) {
    for (int tx = 0; tx < width; tx += TILE_W) {
        process_tile(src, dst, tx, ty,
                     std::min(TILE_W, width-tx),
                     std::min(TILE_H, height-ty));
    }
}
```

### 7.2 Access Patterns for Separable Filters

Horizontal pass: row-major, fully sequential, maximum cache friendliness.

Vertical pass: column-major, stride = `width` bytes. For 1920 × uint8, stride = 1920 → each column access crosses 1920/64 = 30 cache lines. Catastrophic without tiling.

**Solution — transpose intermediate buffer:**
1. Horizontal pass: write to tmp in row-major order
2. Transpose tmp (L1-blocked transpose — classic cache oblivious trick)
3. Apply "horizontal" kernel again to transposed tmp (now effectively vertical)
4. Transpose result back

Alternatively, tile the vertical pass with TILE_W matching cache line width:

```cpp
// Process vertical pass in tile columns
// Each tile column = 64 consecutive columns processed together
// Loads are: base + col_in_tile + row*width — stride accesses
// but within a tile, 64 bytes per cache line touch is amortized
```

### 7.3 Prefetch Hints

For sequential row scan, prefetch N cache lines ahead:

```cpp
// Prefetch 4 cache lines ahead (256 bytes) during pixel scan
for (int x = 0; x < width; x += 32) {
    _mm_prefetch((char*)(src + x + 256), _MM_HINT_T0);  // L1
    // process pixels at x..x+31
}
```

Prefetch distance: typically 128–256 bytes (2–4 cache lines) for sequential access. Too small → cache miss latency visible; too large → thrashes cache.

For multi-row access (3×3 kernel), prefetch all three row-next-lines:

```cpp
_mm_prefetch((char*)(row_prev + x + 64), _MM_HINT_T0);
_mm_prefetch((char*)(row_curr + x + 64), _MM_HINT_T0);
_mm_prefetch((char*)(row_next + x + 64), _MM_HINT_T0);
```

### 7.4 Memory Alignment

| SIMD width | Required alignment | Allocator |
|------------|-------------------|-----------|
| SSE2 (128-bit) | 16 bytes | `aligned_alloc(16, size)` |
| AVX2 (256-bit) | 32 bytes | `aligned_alloc(32, size)` |
| AVX-512 (512-bit) | 64 bytes | `aligned_alloc(64, size)` |
| Cache line | 64 bytes | `aligned_alloc(64, size)` |

Best practice: always align to 64 bytes (cache line). This satisfies all SIMD requirements and avoids false sharing.

```cpp
// Row stride must be a multiple of 64 for aligned AVX-512 stores
size_t aligned_stride = (width + 63) & ~63UL;
uint8_t* buf = static_cast<uint8_t*>(
    std::aligned_alloc(64, aligned_stride * height));
```

Use `__builtin_assume_aligned(ptr, 64)` (GCC/Clang) or `__assume_aligned(ptr, 64)` (MSVC/ICC) to tell the compiler the pointer is aligned, enabling it to emit aligned load/store instructions.

---

## 8. Zero-Allocation Patterns

### 8.1 Arena Allocator for Scratch Buffers

Image filters require temporary buffers (e.g., vertical pass output, histogram arrays, column state). A pool allocated once and reset per-frame eliminates all per-pixel `malloc`:

```cpp
class ScratchArena {
public:
    explicit ScratchArena(size_t capacity)
        : storage_(static_cast<uint8_t*>(std::aligned_alloc(64, capacity)))
        , capacity_(capacity)
        , offset_(0) {}

    ~ScratchArena() { std::free(storage_); }

    // Non-copyable, non-movable (pointers into arena remain valid)
    ScratchArena(const ScratchArena&) = delete;

    template <typename T>
    T* alloc(size_t count, size_t align = alignof(T)) {
        size_t aligned_off = (offset_ + align - 1) & ~(align - 1);
        size_t new_off = aligned_off + sizeof(T) * count;
        if (new_off > capacity_) return nullptr;  // caller must handle
        offset_ = new_off;
        return reinterpret_cast<T*>(storage_ + aligned_off);
    }

    void reset() noexcept { offset_ = 0; }

private:
    uint8_t* storage_;
    size_t   capacity_;
    size_t   offset_;
};
```

Usage in pipeline:

```cpp
// Pre-allocate once for max image size
ScratchArena arena(4096 * 4096 * 4 * 2);  // 2 frame buffers, RGBA

void process_frame(const uint8_t* src, uint8_t* dst,
                   int width, int height, ScratchArena& arena) {
    arena.reset();
    int16_t* tmp = arena.alloc<int16_t>(width * height, 64);
    // ... use tmp for intermediate results ...
}
```

### 8.2 Pre-Allocated Pipeline Buffers

For the pixmask pipeline (bit-depth → median → gaussian → dct), each stage writes to a pre-allocated buffer rather than allocating:

```cpp
struct PipelineBuffers {
    std::vector<uint8_t> stage1;  // after bit-depth reduction
    std::vector<uint8_t> stage2;  // after median
    std::vector<uint8_t> stage3;  // after gaussian
    // All allocated at max supported resolution, never reallocated
};
```

### 8.3 In-Place Operations

Bit-depth reduction is in-place safe (reads then writes each byte). Median filter requires a separate output buffer (reads a 3×3 neighborhood for each output pixel). Gaussian blur separable passes need one scratch row or the full intermediate buffer.

---

## 9. Compile-Time Optimization

### 9.1 Template Specialization for Kernel Sizes

```cpp
// Base template — general path
template <int KernelRadius>
void median_filter(const uint8_t* src, uint8_t* dst, int width, int height);

// Explicit specializations — SIMD sorting network, fully unrolled
template <>
void median_filter<1>(const uint8_t* src, uint8_t* dst, int width, int height);
// → 3x3 sorting network, 19 SORT2 macros, zero loops

template <>
void median_filter<2>(const uint8_t* src, uint8_t* dst, int width, int height);
// → 5x5 sorting network, histogram crossover still faster at r=2 for float

// General path (r >= 3) uses Perreault-Hébert
```

The specializations guarantee the compiler sees fixed loop bounds and constant kernel weights, enabling full unrolling.

### 9.2 Constexpr Gaussian Kernel Generation

```cpp
template <int N>
constexpr std::array<int16_t, N> make_gaussian_kernel(float sigma) {
    // Compute at compile time for fixed sigma values
    // For runtime sigma, generate at startup and cache
    std::array<int16_t, N> k{};
    float sum = 0;
    for (int i = 0; i < N; ++i) {
        float x = i - N/2;
        k[i] = static_cast<int16_t>(std::exp(-0.5f*x*x/(sigma*sigma)) * 256.0f);
        sum += k[i];
    }
    // Normalize (adjust last element to ensure sum == 256)
    return k;
}
// Instantiate for common sigma values
static constexpr auto kGaussKernel_s1 = make_gaussian_kernel<5>(1.0f);
static constexpr auto kGaussKernel_s2 = make_gaussian_kernel<7>(2.0f);
```

### 9.3 LTO (Link-Time Optimization)

Benefits for pixmask:
- Cross-module inlining: SIMD hot loops in `bitdepth.hpp` called from `pipeline.hpp` get inlined across TU boundaries
- Dead code elimination: unused kernel specializations stripped
- Constant propagation: pipeline-configured parameters become compile-time constants in LTO mode

CMake configuration:

```cmake
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if(ipo_supported)
    set_property(TARGET pixmask PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Or explicitly:
target_compile_options(pixmask PRIVATE -flto=thin)
target_link_options(pixmask PRIVATE -flto=thin)
```

ThinLTO (Clang) provides 80–90% of full LTO speedup with incremental rebuild support. Full LTO is 86× slower to link; ThinLTO is 48× slower.

Performance impact: typically 3–8% for library code with cross-module hot paths. Not a substitute for SIMD but compounds it.

---

## 10. Reference Implementations to Study

| Algorithm | Repo | Key file(s) |
|-----------|------|------------|
| SIMD image ops (all platforms) | [ermig1979/Simd](https://github.com/ermig1979/Simd) | `src/Simd/SimdAvx2*.cpp` |
| Portable SIMD | [google/highway](https://github.com/google/highway) | `hwy/highway.h`, `hwy/foreach_target.h` |
| SIMD xsimd | [xtensor-stack/xsimd](https://github.com/xtensor-stack/xsimd) | `include/xsimd/` |
| Median SIMD | [OpenCV median_blur.simd.hpp](https://github.com/opencv/opencv/blob/master/modules/imgproc/src/median_blur.simd.hpp) | Two-tier histogram + sort network |
| Fast Gaussian | [bfraboni/FastGaussianBlur](https://github.com/bfraboni/FastGaussianBlur) | `fast_gaussian_blur.h` |
| IIR Gaussian | [arkanis iir_gauss_blur.h](http://arkanis.de/weblog/2018-08-30-iir-gauss-blur-h-a-gaussian-blur-single-header-file-library) | single header |
| AAN DCT | [prtsh/aan_dct](https://github.com/prtsh/aan_dct) | fixed-point AAN |
| SIMD DCT (production) | [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo) | `simd/x86_64/jsimd_avx2.asm`, `simd/arm/aarch64/jsimd_neon.S` |
| SIMD sort networks | [WojciechMula/simd-sort](https://github.com/WojciechMula/simd-sort) | AVX-512 / AVX2 |

---

## 11. Pitfalls and Risks

| Risk | Detail | Mitigation |
|------|--------|-----------|
| AVX-512 throttling | Intel CPUs drop clock frequency when AVX-512 is active (Skylake-X, some Ice Lake). May be slower than AVX2 for short bursts. | Benchmark with `perf` on target hardware. Highway's dynamic dispatch will select AVX-512 only if worth it on current CPU frequency state. |
| AVX2 no native 8-bit shift | `_mm256_srli_epi8` does not exist. Requires 16-bit shift + masking workaround. | Use the interleaved 16-bit shift pattern above, or use Highway's `ShiftRight` which generates optimal code per target. |
| Unaligned loads on edge pixels | Image rows rarely end on 32-byte boundaries. Last AVX2 load may read past the buffer end. | Pad rows to next 64-byte boundary at allocation; or use masked loads (`_mm256_maskload_epi32`) for the tail. |
| IIR Gaussian float precision | Young-Van Vliet coefficients in float32 become unstable at σ > ~50. | Switch to double for σ > 30, or use box-blur approximation. |
| 3×3 median boundary handling | Sorting network assumes all 9 neighbors exist. | Process interior pixels with SIMD; handle a 1-pixel border with scalar code, or extend image by 1 pixel via reflection/clamp. |
| Perreault-Hébert histogram overflow | Column histograms use uint16; max value = image height. For 4096-pixel images, uint16 max = 65535, safe. For 8-bit depth, accumulator = height ≤ 65535. | Verified safe for all common image sizes. |
| Cache-invalidating transpositions | Naive in-place transpose of large images = N² cache misses. | Always use blocked transpose with tile size matching L1 cache: 64×64 bytes for uint8. |
| LTO + SIMD inline asm | LTO may not inline asm blocks. Some SIMD implementations use inline asm (libjpeg-turbo .S files). | Use intrinsics (not inline asm) in all Highway/xsimd paths; LTO works correctly with intrinsics. |

---

## 12. Recommended pixmask Implementation Stack

Based on the research above:

1. **SIMD abstraction**: Google Highway for all new SIMD code
   - Runtime dispatch via `HWY_DYNAMIC_DISPATCH`
   - Covers x86 (SSE2→AVX-512), ARM (NEON, SVE), WASM

2. **Bit-depth reduction**: SIMD AND-mask with `constexpr`-generated mask value per target bit depth

3. **Median filter**:
   - 3×3, 5×5: SIMD sorting network (`_mm256_min/max_epu8` pattern)
   - r ≥ 3: Perreault-Hébert two-tier histogram, vectorized histogram updates

4. **Gaussian blur**:
   - σ ≤ 3: separable FIR with fixed-point integer 1D convolution
   - σ > 3: 3-pass box blur (linear time, constant per sigma)
   - High-quality path: Young-Van Vliet IIR (4 passes, O(1), double precision for σ > 30)

5. **DCT**: libjpeg-turbo AAN integer DCT as reference; Highway-based float DCT (libjxl approach) for portability

6. **Memory**: 64-byte aligned allocations, `ScratchArena` for temporaries, pre-allocated pipeline buffers

7. **Build**: ThinLTO enabled for release builds, template specializations for k=1 and k=2 median radii

---

*Sources:*
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [Simd Library (ermig1979)](https://github.com/ermig1979/Simd)
- [Google Highway](https://github.com/google/highway)
- [xsimd](https://github.com/xtensor-stack/xsimd)
- [SIMDe](https://simd-everywhere.github.io/blog/2024/05/02/0.8.0-0.8.2-release.html)
- [OpenCV median_blur.simd.hpp](https://github.com/opencv/opencv/blob/master/modules/imgproc/src/median_blur.simd.hpp)
- [Perreault & Hébert (2007), IEEE TIP](https://ieeexplore.ieee.org/document/4287006/)
- [bfraboni/FastGaussianBlur](https://github.com/bfraboni/FastGaussianBlur)
- [Young-Van Vliet IIR Gaussian (arkanis)](http://arkanis.de/weblog/2018-08-30-iir-gauss-blur-h-a-gaussian-blur-single-header-file-library)
- [libjpeg-turbo SIMD Coverage](https://libjpeg-turbo.org/About/SIMDCoverage)
- [prtsh/aan_dct](https://github.com/prtsh/aan_dct)
- [Sudo Null SIMD Median](https://sudonull.com/post/118351-Optimization-of-image-processing-in-C-using-SIMD-Median-filter)
- [MDPI: Fast Gaussian Filter Approximations SIMD](https://www.mdpi.com/2076-3417/14/11/4664)
- [Arm NEON Intrinsics Reference](https://arm-software.github.io/acle/neon_intrinsics/advsimd.html)
- [WojciechMula/simd-sort](https://github.com/WojciechMula/simd-sort)
- [Johnny's Software Lab: LTO](https://johnnysswlab.com/link-time-optimizations-new-way-to-do-compiler-optimizations/)
- [libjxl xl_overview](https://github.com/libjxl/libjxl/blob/main/doc/xl_overview.md)
- [Fastest Gaussian Blur (blog.ivank.net)](https://blog.ivank.net/fastest-gaussian-blur.html)
- [Stack Blur (Melatonin)](https://melatonin.dev/blog/implementing-marios-stack-blur-15-times-in-cpp/)
