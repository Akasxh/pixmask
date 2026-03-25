# Spatial Smoothing Algorithms for Adversarial Defense

> Research for pixmask — a C++ image sanitization library targeting maximum throughput.
> Focus: implementation details, complexity, SIMD paths, and adversarial-defense parameter recommendations.

---

## 1. Median Filter

### 1.1 Naive Sliding Window — O(r²) per Pixel

For a k×k window (k = 2r+1), gather k² pixels into a buffer, partial-sort to find the median, emit.

```cpp
// Minimum working skeleton — uint8 grayscale
void median_naive(const uint8_t* src, uint8_t* dst,
                  int W, int H, int r) {
    const int k = 2*r + 1;
    std::vector<uint8_t> buf(k * k);
    for (int y = r; y < H-r; ++y) {
        for (int x = r; x < W-r; ++x) {
            int n = 0;
            for (int dy = -r; dy <= r; ++dy)
                for (int dx = -r; dx <= r; ++dx)
                    buf[n++] = src[(y+dy)*W + (x+dx)];
            std::nth_element(buf.begin(),
                             buf.begin() + n/2,
                             buf.begin() + n);
            dst[y*W + x] = buf[n/2];
        }
    }
}
```

Complexity: O(W·H·k²) time, O(k²) scratch space.
`std::nth_element` gives O(k²) average, which is fine for k ≤ 7.

**Sorting-network variant (branchless, cache-friendly):**
For 3×3 exactly 9 elements. Optimal sorting network requires 25 compare-and-swap (CAS) operations. Replace `nth_element` with a hand-unrolled sequence of `min/max` pairs:

```cpp
#define CAS(a,b) { uint8_t t = std::min(a,b); b = std::max(a,b); a = t; }
// Batcher odd-even or network from Knuth TAOCP Vol. 3
// 9-element optimal: 25 CAS steps, pick element [4] as median
```

For 5×5 (25 elements): 49 CAS steps for full sort; for just the median, 89 CAS gives exact median via Waksman's network — still 3× faster than `nth_element` due to no branch misprediction.

---

### 1.2 Huang's Histogram Algorithm — O(W·k) per Row

**Paper:** Huang, Yang, Tang — "A Fast Two-Dimensional Median Filtering Algorithm", IEEE TASP 1979.

**Key insight:** Maintain a 256-bin histogram of the current k×k window. When sliding one column right:
- Remove k pixels from the leftmost column (k decrements)
- Add k pixels from the new rightmost column (k increments)
- Find median by scanning histogram from 0 upward until cumulative count > k²/2

Complexity per row: O(W·k) updates + O(W·256) for median scan = **O(W·k)** dominant for k < 256.
Total: O(H·W·k) — linear in k, not k².

```cpp
struct HuangMedian {
    uint16_t hist[256] = {};
    int count = 0;      // total pixels in window
    int median_pos = 0; // cached: sum of hist[0..median_pos-1]
    uint8_t median_val = 0;

    void add(uint8_t v)    { ++hist[v]; ++count; }
    void remove(uint8_t v) { --hist[v]; --count; }

    // After adds/removes, update median_val
    void recompute() {
        int half = (count + 1) / 2;
        int cum = 0;
        for (int i = 0; i < 256; ++i) {
            cum += hist[i];
            if (cum >= half) { median_val = i; return; }
        }
    }
};
```

Limitation: The 256-scan for each pixel is slow for small windows. Profitable when k ≥ ~20. For k < 11, naive + sorting network is faster.

---

### 1.3 Perreault & Hébert — O(1) per Pixel (CTMF)

**Paper:** S. Perreault, P. Hébert — "Median Filtering in Constant Time", IEEE TIP 2007.
**Source:** `http://nomis80.org/ctmf.html` (C source with SSE2/MMX)
**GitHub mirror:** `https://github.com/TeraLogics/ConstantTimeMedianFiltering`

**Innovation over Huang:** Maintain one histogram per image column. When the kernel moves one pixel down, each column histogram is updated with O(1) work (add one row's pixel, remove one row's pixel). When moving right by one pixel, one column histogram is subtracted from the running sum and one is added.

**Two-tier (coarse/fine) histogram:** The critical implementation detail.

```c
typedef struct __attribute__((aligned(16))) {
    uint16_t coarse[16];   // 4-bit: 16 buckets, each = sum of 16 fine bins
    uint16_t fine[16][16]; // 8-bit: 256 total bins, indexed as [coarse_bucket][fine_bucket]
} Histogram;
```

The coarse level narrows the search to a 16-bin range (16 values) before scanning fine bins. This reduces median finding from O(256) to O(16 + 16) = O(32) — a 4× win on the scan step.

**SIMD acceleration in CTMF:** The column histogram add/subtract is done with SSE2 on the coarse array (16 × uint16 = 128 bits fits exactly in one XMM register). Addition of two column histograms is a single `_mm_add_epi16`.

**Stripe processing for cache efficiency:** The image is processed in horizontal stripes whose height equals the L1 cache / (image width × bytes per histogram). This keeps all active column histograms in cache.

**Complexity:** O(1) per pixel after O(W) setup per row. Total: O(W·H) regardless of radius r.

**When to prefer CTMF vs. naive:**
- CTMF wins for r ≥ 5 (roughly k ≥ 11) on 8-bit images.
- CTMF is restricted to **8-bit uint** images and **square** kernels.
- For 16-bit images, use the O(k log k) order-statistic tree (weight-balanced BST) as used in DIPlib.
- DIPlib switches at k = 11 between naive and tree.

---

### 1.4 SIMD Sorting Networks for Small Kernels (3×3, 5×5)

**Best reference:** sudonull.com article on SIMD median filter optimization.
Performance on 1920×1080 uint8:
- Scalar naive: 24.8 ms
- SSE2 (16 pixels/iter, sorting network): 0.565 ms → **43.9× speedup**
- AVX2 (32 pixels/iter): 0.424 ms → **58.6× speedup**

**AVX2 3×3 implementation strategy:**

Process 32 pixels simultaneously. For 3×3 = 9 elements, load 9 rows of 32 pixels each into `__m256i` registers. Run a 25-step CAS sorting network using `_mm256_min_epu8` / `_mm256_max_epu8`.

```cpp
// Branchless CAS for 8-bit SIMD
#define CAS256(a, b) {              \
    __m256i t = _mm256_min_epu8(a, b); \
    b = _mm256_max_epu8(a, b);      \
    a = t;                          \
}
// Load 9 rows of 32 pixels:
__m256i r0 = _mm256_loadu_si256(...); // (y-1, x-1..x+30)
// ... 8 more rows ...
// 25 CAS steps on r0..r8, then r4 is the median
```

**Key trick (branchless scalar fallback):**

```cpp
int d = a - b;
int m = ~(d >> 8);   // all-ones if a < b, all-zeros otherwise
b += d & m;
a -= d & m;
```

No branches, no CMOV dependency — pure ALU for 8-bit pixels.

**5×5 kernel (25 elements):**
Uses odd-even merge sort network for the first pass (sort each row of 5), then merge. General approach: 2-pass odd-even sort — 1st pass sorts each row, 2nd pass merges recursively. Total CAS: ~60–70 for median-only extraction.

---

### 1.5 Adversarial Defense: Optimal Kernel Size

**From Feature Squeezing (Xu et al., NDSS 2018):**
- 2×2 median: accuracy drops 94.84% → 89.29% on CIFAR-10 (too destructive)
- 3×3 median: best for L0 attacks (CW0, JSMA) — nearly perfect adversarial detection with minimal clean accuracy loss
- Recommendation: **3×3 is the sweet spot** for pixmask. Larger (5×5) crushes fine details needed by VLMs.

**Implementation recommendation for pixmask:**
- k ≤ 9 (r ≤ 4): use SIMD sorting network path
- k = 11–31: use CTMF (histogram, O(1))
- k > 31 or 16-bit: use Huang + optional tree

---

## 2. Gaussian Blur

### 2.1 Separable Convolution — O(W·H·k)

A 2D Gaussian is separable: `G(x,y,σ) = G(x,σ) · G(y,σ)`. Two 1D passes replace one 2D pass, reducing O(k²) to O(2k) per pixel.

```
horizontal pass: for each row, convolve with 1D Gaussian kernel
vertical pass:   for each column, convolve with 1D Gaussian kernel
```

Integer kernel coefficients (for σ=1.0, k=5): `{1, 4, 6, 4, 1}`, shift right 4 bits (divide by 16). For σ=1.4, k=7: `{1, 6, 15, 20, 15, 6, 1}`, divide by 64. These fit in 8-bit arithmetic with 16-bit accumulators.

**SIMD horizontal pass:** Load 16 (SSE) or 32 (AVX2) uint8 pixels, promote to uint16, multiply by kernel coefficients (broadcast scalar), accumulate, shift right. ~5× speedup over scalar.

**Complexity:** O(W·H·k) — scales linearly with kernel size.

---

### 2.2 Extended Box Blur (3-Pass Approximation)

**Based on:** Ivan Kutskir's fast Gaussian blur; bfraboni's FastGaussianBlur.
**Source:** `https://github.com/bfraboni/FastGaussianBlur`

**Core theorem:** By the central limit theorem, repeated convolution with a box filter converges to Gaussian. Three passes suffices for visual quality indistinguishable from true Gaussian (0.04% average error per pixel).

**Box size formula** (converts σ to box widths for n=3 passes):

```
wIdeal = sqrt((12*sigma*sigma / n) + 1)
wl = floor(wIdeal)
if (wl % 2 == 0) wl--         // must be odd
wu = wl + 2
m = round((12*sigma*sigma - n*wl*wl - 4*n*wl - 3*n) / (-4*wl - 4))
// sizes: m boxes of width wl, (n-m) boxes of width wu
```

Example: σ=3.0, n=3 → wl=5 (3 times) or wl=5, wu=7 mix.

**Sliding accumulator (O(1) per pixel):**

```cpp
void box_blur_h(uint8_t* src, uint8_t* dst, int W, int H, int r) {
    float iarr = 1.0f / (2*r + 1);
    for (int i = 0; i < H; ++i) {
        int ti = i*W, li = ti, ri = ti+r;
        float fv = src[ti], lv = src[ti + W-1], val = (r+1)*fv;
        for (int j = 0; j < r; ++j) val += src[ti+j];
        for (int j = 0; j <= r; ++j) {
            val += src[ri++] - fv;
            dst[ti++] = (uint8_t)(val * iarr + 0.5f);
        }
        for (int j = r+1; j < W-r; ++j) {
            val += src[ri++] - src[li++];
            dst[ti++] = (uint8_t)(val * iarr + 0.5f);
        }
        for (int j = W-r; j < W; ++j) {
            val += lv - src[li++];
            dst[ti++] = (uint8_t)(val * iarr + 0.5f);
        }
    }
}
```

Apply box_blur_h (horizontal), transpose image (block-wise for cache), box_blur_h again (now vertical), transpose back. Three times total.

**Performance:** 2 million pixels in ~7 ms on Ryzen 7 2700X with OpenMP.
**Key advantage:** Constant execution time regardless of σ. σ=5 as fast as σ=50.

---

### 2.3 IIR Gaussian — O(1) per Pixel, O(n) Total

#### Young & Van Vliet (1995) — Recommended

**Paper:** I. Young, L. Van Vliet — "Recursive implementation of the Gaussian filter", Signal Processing 1995.
**Gist:** `https://gist.github.com/da0fb075c8aca70f67ad`

A 3rd-order causal IIR filter approximates the Gaussian. Six multiply-adds per pixel per dimension, independent of σ.

**Coefficients** (σ ≥ 0.5, two regimes):

For σ < 3.556:
```
q = 2.5091 * (exp(0.0561 * sigma) - 1) / (exp(0.0561 * 3.556) - 1)
```
For σ ≥ 3.556:
```
q = 0.98711 * sigma - 0.96330
```

Then:
```
b0 = 1.57825 + 2.44413*q + 1.4281*q^2 + 0.422205*q^3
b1 = 2.44413*q + 2.85619*q^2 + 1.26661*q^3
b2 = -(1.4281*q^2 + 1.26661*q^3)
b3 = 0.422205*q^3
B  = 1 - (b1 + b2 + b3) / b0
```

**Causal pass (left to right):**
```
y[i] = B*x[i] + (b1*y[i-1] + b2*y[i-2] + b3*y[i-3]) / b0
```

**Anticausal pass (right to left, Triggs-Sdika boundary):**
Initialize the anticausal filter using the matrix M (3×3) computed from coefficients to avoid ringing at image boundaries. Without Triggs-Sdika correction the boundary artifacts are visible.

**Boundary conditions (Triggs & Sdika, 2006):**
Compute M matrix once per σ, use it to initialize the first 3 samples of the anticausal pass from the final 3 causal outputs.

**Key properties:**
- Works for any σ ≥ 0.5
- No kernel truncation artifacts
- Apply row-wise then column-wise (separable)
- Total: O(6 · W · H) multiply-adds for both dimensions

#### Deriche (1992)

**Paper:** R. Deriche — "Recursively implementing the Gaussian and its derivatives", INRIA TR 1992.

Similar 4th-order IIR approach but coefficients lack closed form — must be tabulated per σ. Prone to ringing near boundaries without careful initialization. **Prefer Van Vliet for new implementations.**

---

### 2.4 Stack Blur (Weighted Box Approximation)

**Author:** Mario Klingemann (2004), popularized in image editing tools.

Maintains a "stack" (conceptual pyramid weighting) so central pixels count more than edge pixels, giving a triangular weight profile instead of uniform box. The effect approaches Gaussian faster per pass than plain box blur.

**Mechanics:** Sliding circular buffer of size `2r+1`. Maintains two accumulators:
- `sumIn`: sum of incoming (right side) pixels
- `sumOut`: sum of outgoing (left side) pixels
- The stack sum = `sumIn + sumOut + center * (r+1)`

Each pixel: subtract `sumOut` from total, add `sumIn`, advance pointers. O(1) per pixel.

**Comparison to 3-pass box:** Stack blur requires only 1 pass horizontally + 1 vertically but gives a triangular convolution kernel. Two passes of Stack Blur ≈ one pass of 3-pass box blur in quality. However, it does not achieve the same 0.04% Gaussian approximation quality. **Prefer 3-pass box for defense use cases.**

---

### 2.5 Adversarial Defense: Optimal Sigma

**From Xu et al. (Feature Squeezing, NDSS 2018):** Gaussian smoothing tested with local smoothing; effective for L2 attacks (C&W).

**From Guo et al. (ICLR 2018):** TV minimization outperformed Gaussian alone; Gaussian used as baseline.

**From low-pass filtering study (PMC10675189):**
- Optimal σ depends on attack strength (FGSM intensity 5–15% of pixel range)
- For 5% FGSM on 224×224 ImageNet: σ ≈ 8–10 pixels → accuracy 0.913 vs 0.206 undefended
- Clean accuracy degradation is minimal with augmentation training at these σ values

**Practical recommendation for pixmask (without retraining):**
- **σ = 0.5** (k=3): minimal blur, minimal clean accuracy drop, good for low-epsilon L∞ attacks
- **σ = 1.0** (k=5): balanced — good against L2, C&W
- **σ = 2.0** (k=11): aggressive — needed for strong perturbations but may blur fine VLM-relevant details
- Use **3-pass box blur** for all cases (fastest, sigma-independent speed)

---

## 3. Bilateral Filter

### 3.1 Naive Bilateral — O(k²) per Pixel

The bilateral filter is a joint spatial-range filter:

```
BF[I]_p = (1/W_p) * Σ_{q∈S} G_σs(|p-q|) * G_σr(|I_p - I_q|) * I_q
```

Where:
- `G_σs`: spatial Gaussian (proximity weight)
- `G_σr`: range Gaussian (intensity similarity weight)
- `W_p`: normalization factor

**Why it preserves edges:** Pixels across an edge differ in intensity by > σr, so `G_σr` suppresses their contribution. Only same-region neighbors accumulate. Adversarial perturbations, being small in intensity (ε < 8/255 for typical L∞ attacks), are treated as low-intensity "noise" and smoothed away, while actual edges (contrast >> ε) are preserved.

```cpp
void bilateral_naive(const float* src, float* dst,
                     int W, int H, int r,
                     float sigma_s, float sigma_r) {
    float inv_ss2 = -0.5f / (sigma_s * sigma_s);
    float inv_sr2 = -0.5f / (sigma_r * sigma_r);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float Ip = src[y*W + x];
            float sum = 0, norm = 0;
            for (int dy = -r; dy <= r; ++dy) {
                int yy = std::clamp(y+dy, 0, H-1);
                for (int dx = -r; dx <= r; ++dx) {
                    int xx = std::clamp(x+dx, 0, W-1);
                    float Iq = src[yy*W + xx];
                    float ws = expf((dx*dx + dy*dy) * inv_ss2);
                    float wr = expf((Ip-Iq)*(Ip-Iq) * inv_sr2);
                    float w  = ws * wr;
                    sum  += w * Iq;
                    norm += w;
                }
            }
            dst[y*W + x] = sum / norm;
        }
    }
}
```

Complexity: O(W·H·k²). Very slow for large kernels. Practical window: `r = 3*sigma_s` → k ≈ 7 for σs=1 (k²=49), k≈25 for σs=4 (k²=625).

**Optimization:** Precompute spatial weights as a lookup table (k×k floats). Precompute range weights as a 256-entry LUT for 8-bit images (`range_lut[delta] = exp(-delta²/(2σr²))`). This avoids all `expf` calls in the inner loop.

---

### 3.2 Bilateral Grid (Paris & Durand 2006) — O(n/σs²·σr) per Pixel

**Paper:** S. Paris, F. Durand — "A Fast Approximation of the Bilateral Filter Using a Signal Processing Approach", ECCV 2006 / IJCV 2009.

**Core idea:** Lift the 2D image into a 3D grid `(x, y, intensity)`. In this space, the bilateral filter becomes a linear Gaussian convolution. Sample back down (slice) to recover the filtered result.

**Algorithm:**
1. **Create grid:** Size `(W/σs, H/σs, 256/σr)`. Each grid cell accumulates weighted pixel values.
2. **Splat:** For each pixel `(x,y)` with intensity `I`, add `I` and `1.0` to grid cell `(x/σs, y/σs, I/σr)`.
3. **Blur grid:** Apply 3D Gaussian blur (separable) on the small grid.
4. **Slice:** For each output pixel, trilinearly interpolate at `(x/σs, y/σs, I/σr)` to get `sum` and `weight`. Output = `sum / weight`.

**Complexity:** Grid size is `(W·H/σs²) · (256/σr)`. For σs=8, σr=0.1 (range 0–1): grid is ~40× smaller than image. The dominant cost is splatting and slicing, O(W·H), not the grid blur.

**Typical parameters for adversarial defense:** σs = 3–5 pixels, σr = 0.1–0.2 (in [0,1] normalized range, = 25–50 in [0,255]).

---

### 3.3 O(1) Trigonometric Bilateral Filter (Chaudhury 2011)

**Paper:** K. Chaudhury, D. Sage, M. Unser — "Fast O(1) Bilateral Filtering Using Trigonometric Range Kernels", IEEE TIP 2011.

**Key property (shiftability):** `cos(ω(I_p - I_q)) = cos(ωI_p)cos(ωI_q) + sin(ωI_p)sin(ωI_q)`.

This decomposes the range kernel into a product of terms each depending on only one pixel. The bilateral filter then separates into standard (fast, O(1)) Gaussian spatial convolutions:

```
BF[I]_p ≈ Σ_{k=0}^{K} c_k · [cos(ω_k I_p) · (G_σs * cos(ω_k I)) +
                                sin(ω_k I_p) · (G_σs * sin(ω_k I))]
           / [c_0 + Σ_{k=1}^{K} c_k · cos(ω_k I_p) · (G_σs * cos(ω_k I_p))]
```

Where K = 3–5 terms suffice for good Gaussian range approximation. Each `G_σs * f(I)` is a standard Gaussian blur — computable with box blur or IIR in O(n) time.

**Complexity:** O(K · n) total. For K=4: 4× the cost of a single Gaussian blur. Fast for any σs (uses IIR) but σr tightly controls how many terms K are needed (large σr → small K).

**C++ implementation:** Available via EPFL bigwww group; also in CImg (`CImg::blur_bilateral`).

---

### 3.4 Adversarial Defense Value

Bilateral filtering provides **edge-aware noise removal** — exactly what adversarial perturbations look like: small-amplitude, high-frequency noise not correlated with image structure. Studies show bilateral filtering removes >90% of adversarial examples from multiple attack types in black-box settings.

**For pixmask:** Recommend bilateral grid implementation (Paris-Durand) for images where VLM comprehension depends on fine edge detail. Slightly slower than Gaussian but preserves text legibility, object boundaries etc.

Recommended parameters: **σs = 3, σr = 0.1** (normalized [0,1] range = ~25/255). Window r = 3·σs = 9 (k=19).

---

## 4. Non-Local Means (NLM)

### 4.1 Algorithm

**Paper:** A. Buades, B. Coll, J.M. Morel — "A non-local algorithm for image denoising", CVPR 2005.

Weighted average of all pixels in the image (or search window), where weight depends on patch similarity:

```
NLM[I]_p = (1/Z_p) Σ_{q∈Search} exp(-||P_p - P_q||² / h²) · I_q
```

Where:
- `P_p`: patch of size f×f centered at p (standard: f=7)
- Search window: (2s+1)×(2s+1) centered at p (standard: s=10, so 21×21)
- h: filter strength (h = σ for white Gaussian noise; h ≈ 0.75σ–σ in practice)
- Z_p: normalization

**Complexity (naive):** O(W·H · (2s+1)² · f²). For defaults: O(W·H · 441 · 49) ≈ O(21,000 · W·H). 50–100× slower than Gaussian blur.

---

### 4.2 Fast NLM with Integral Images

**Paper:** L. Condat — "A Simple Trick to Speed Up and Improve the Non-Local Means", 2010.

Key trick: For each displacement vector (dx, dy), precompute the array of squared differences between all pairs of pixels at that offset using a **2D summed area table** (integral image). Then the patch distance for any (p, q=p+(dx,dy)) is just a rectangular query O(1).

```
D(p, p+v) = SAT of (I - shift(I, v))²  queried over f×f patch
```

Total complexity: O((2s+1)² · W·H) = O(441 · W·H) for standard parameters. ~50× speedup, still O(s² · n) but f² factor eliminated.

**Implementation steps:**
```
For each offset (dx, dy) in [-s,s]×[-s,s]:
    1. Compute diff_sq[y][x] = (I[y][x] - I[y+dy][x+dx])^2
    2. Build SAT of diff_sq
    3. For each pixel p, query SAT for f×f neighborhood around p
       → adds to weight: w = exp(-SAT_query / (h² · f²))
       → adds to weighted_sum: w * I[p+(dx,dy)]
```

**Reported speedup:** 20×–200× over naive NLM depending on search window size.

---

### 4.3 Patch-Based vs Pixel-Based

- **Pixel-based NLM:** Use 1×1 "patches" (just pixel values). Faster but poor denoising.
- **Patch-based NLM (standard):** 7×7 patches give best signal-to-noise ratio. The larger the patch, the more robust to noise but the higher the computation.
- **Block-matching (BM3D variant):** Groups similar 3D patches and applies 3D transform — beyond scope here.

---

### 4.4 When NLM Outperforms Gaussian/Median

NLM is better when:
1. Image has **repetitive texture** — NLM finds many similar patches, averages precisely
2. **High perturbation strength** — Gaussian blurs edges; NLM reconstructs from similar clean-looking regions
3. **Adversarial examples in natural scenes** where textures repeat

NLM is overkill (and slower) when:
- Low epsilon attacks (ε < 4/255) — Gaussian or median is sufficient
- Real-time requirements — NLM is 20–1000× slower than Gaussian

**For pixmask:** NLM should be an optional high-quality mode. Default pipeline uses Gaussian or bilateral. Offer NLM with reduced search window (s=5, 11×11) for speed/quality tradeoff.

---

## 5. Total Variation (TV) Denoising

### 5.1 ROF Model

**Paper:** L. Rudin, S. Osher, E. Fatemi — "Nonlinear total variation based noise removal algorithms", Physica D 1992.

Minimize:
```
u* = argmin_u { λ·TV(u) + ½‖u - f‖₂² }

TV(u) = Σ_{p} √(|∇_x u_p|² + |∇_y u_p|²)  (isotropic)
      = Σ_{p} |∇_x u_p| + |∇_y u_p|         (anisotropic, faster)
```

Where f is the noisy image, λ controls smoothness vs. fidelity.

**Why TV works for adversarial defense:** Adversarial perturbations increase total variation significantly (they add oscillatory noise across every pixel). TV minimization drives u toward piecewise-smooth solution, discarding these high-frequency components.

---

### 5.2 Chambolle's Algorithm (2004)

**Paper:** A. Chambolle — "An Algorithm for Total Variation Minimization and Applications", JMIV 2004.

Introduces a dual variable `p = (p1, p2) ∈ ℝ^{W×H×2}` (vector field) and solves the dual problem via projected gradient descent.

**Update equations:**

```
p^{n+1} = proj_P(p^n + τ·∇(div(p^n) - f/λ))

u* = f - λ·div(p*)
```

Where projection onto `P = {p : |p_i| ≤ 1 ∀i}`:
```
proj_P(q)_i = q_i / max(1, |q_i|)
```

**Step size:** τ ≤ 1/8 (for stability; τ = 0.248 or 0.25 in practice).

**Discrete gradient (forward differences):**
```
(∇u)_x[i,j] = u[i, j+1] - u[i, j]  (clamp at border)
(∇u)_y[i,j] = u[i+1, j] - u[i, j]
```

**Discrete divergence (backward differences):**
```
(div p)[i,j] = p_x[i,j] - p_x[i, j-1] + p_y[i,j] - p_y[i-1, j]
```

**Convergence:** Typically 50–200 iterations. Convergence check: `max|p^{n+1} - p^n| < 1e-2`.

**C++ per-iteration cost:** O(W·H) — 4 additions per pixel for gradient, 4 additions for divergence, normalization.

```cpp
// One Chambolle iteration (isotropic)
void chambolle_iter(float* px, float* py, const float* f,
                    float lambda, float tau, int W, int H) {
    // 1. Compute u = f - lambda * div(p)
    // 2. Compute grad(u), update p
    // 3. Project p onto |p| <= 1
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            // divergence
            float divp = px[y*W+x] - (x>0 ? px[y*W+x-1] : 0)
                       + py[y*W+x] - (y>0 ? py[(y-1)*W+x] : 0);
            float u_xy = f[y*W+x] - lambda * divp;
            // gradient of u
            float gx = (x<W-1 ? u_xy - f[y*W+x+1] : 0) / lambda; // simplified
            float gy = (y<H-1 ? u_xy - f[(y+1)*W+x] : 0) / lambda;
            float np_x = px[y*W+x] + tau * gx;
            float np_y = py[y*W+x] + tau * gy;
            float norm = std::max(1.0f, sqrtf(np_x*np_x + np_y*np_y));
            px[y*W+x] = np_x / norm;
            py[y*W+x] = np_y / norm;
        }
    }
}
```

---

### 5.3 Split Bregman Method (Goldstein & Osher 2009)

**Paper:** T. Goldstein, S. Osher — "The Split Bregman Method for L1-Regularized Problems", SIAM J. Imaging Sciences 2009.

Introduces auxiliary variable `d ≈ ∇u` to decouple the L1 TV term from the L2 fidelity term:

```
min_{u,d} { λ‖d‖₁ + ½‖u - f‖₂² }  s.t. d = ∇u
```

**Iteration:**
```
u^{k+1} = argmin_u { ½‖u-f‖₂² + μ/2 ‖∇u - d^k + b^k‖₂² }
         → Solved with FFT: (I + μΔ)u = f + μ div(d^k - b^k)
           where Δ is the Laplacian, solved in O(n log n)

d^{k+1} = shrink(∇u^{k+1} + b^k, λ/μ)
         → Soft thresholding: shrink(x, γ) = x/|x| * max(|x|-γ, 0)

b^{k+1} = b^k + ∇u^{k+1} - d^{k+1}
```

**u-update via FFT (constant time per iteration):**
In the frequency domain, `(I + μΔ)` is diagonal: `(1 + μ(ω_x² + ω_y²)) * Û = RHS`.
Each u-update costs 2 FFTs + pointwise division. For W×H = 1024×1024: ~3ms on modern hardware.

**Convergence:** Typically 10–30 iterations to satisfactory quality. Faster convergence than Chambolle's projection (which may need 100–300 iterations for strict convergence).

**Parameter guidance:**
- λ: TV regularization weight. For adversarial defense: λ = 0.1–0.3
- μ: ADMM penalty. Set μ = 0.5–2.0 × λ
- Iterations: 20–50 for visual quality adequate for VLM inference

---

### 5.4 Chambolle-Pock (Primal-Dual, 2011)

More general than Chambolle 2004 or Split Bregman; handles the exact ROF functional with provable O(1/N) convergence.

**Updates (from extracted algorithm):**
```
p^{n+1} = proj_P(p^n + σ·∇ū^n)           // dual step
u^{n+1} = (u^n + τ·div(p^{n+1}) + τ/λ·f) / (1 + τ/λ)  // primal step
ū^{n+1} = u^{n+1} + θ·(u^{n+1} - u^n)   // extrapolation θ=1
```

Step size constraint: `σ·τ ≤ 1/L²` where L² = 8 (spectral norm of gradient operator for 2D).
Typical: τ = 0.01, σ = 12.5 (satisfying σ·τ = 0.125 < 1/8).

**C++ implementation notes:**
- Maintain two arrays: `px[W*H]`, `py[W*H]` for dual variable
- Main loop: O(W·H) per iteration with only adds/multiplies
- No FFT needed (unlike Split Bregman u-update)
- 50–100 iterations typical for convergence

**For pixmask:** Chambolle-Pock is the cleanest C++ implementation choice — no FFT dependency, simple pixelwise operations, amenable to SIMD.

---

### 5.5 Adversarial Defense Value of TV

**From Guo et al. (ICLR 2018):** TV minimization + image quilting = best input-transformation defense tested. TV eliminates 60% of strong gray-box attacks and 90% of black-box attacks. Outperforms pure Gaussian/median smoothing because:
1. TV actively minimizes oscillatory content (what adversarial perturbations are)
2. Edges are preserved (piecewise smooth model)
3. Non-differentiable operator — gradient masking frustrates white-box attacks

**Recommended λ for pixmask:** 0.05–0.15. Higher λ produces oil-painting effect that may confuse VLMs. Test with clean image quality metrics (SSIM > 0.9 is a good threshold).

**Speed concern:** Chambolle-Pock at 50 iterations × O(W·H) = ~50× slower than single Gaussian blur. For 1920×1080: ~50ms vs ~1ms. Acceptable for a defense pipeline; not for real-time video.

---

## 6. Implementation Recommendations for pixmask

### Algorithm Selection Matrix

| Use Case | Algorithm | Complexity | Latency (1080p) |
|---|---|---|---|
| Fast, light defense | 3-pass box blur (σ≤3) | O(n) | < 2 ms |
| IIR Gaussian any σ | Van Vliet 3rd-order IIR | O(n) | < 2 ms |
| Median (small kernel) | SIMD sorting network k=3 | O(n) | < 1 ms |
| Median (any k, 8-bit) | CTMF (Perreault-Hébert) | O(n) | 5–10 ms |
| Edge-preserving | Bilateral (naive, σs≤4) | O(n·k²) | 20–50 ms |
| Edge-preserving fast | Bilateral grid (Paris-Durand) | O(n) | 5–10 ms |
| Patch-based quality | NLM (integral image accel.) | O(s²·n) | 200–500 ms |
| Best TV defense | Chambolle-Pock (50 iters) | O(n·iters) | 50–100 ms |
| Fastest TV | Split Bregman + FFT (20 iters) | O(n·log(n)·iters) | 30–60 ms |

### C++ Header Structure Recommendation

```
include/fsq/filters/
  gaussian.hpp         -- box_blur_3pass, iir_gaussian_van_vliet
  median.hpp           -- median_simd_3x3, median_ctmf
  bilateral.hpp        -- bilateral_naive, bilateral_grid
  nlm.hpp              -- nlm_fast (integral image)
  tv.hpp               -- tv_chambolle_pock, tv_split_bregman
```

### Key Dependencies to Avoid

- **No OpenCV** — use raw `uint8_t*` arrays with stride
- **No FFTW** for TV — Chambolle-Pock avoids FFT; use Split Bregman only if FFT latency is acceptable
- **AVX2 required** for SIMD paths; runtime dispatch with CPUID for SSE2 fallback
- **Use `immintrin.h`** for all SIMD intrinsics

### Pipeline Order

```
Bit-depth reduction → Median (3×3) → Gaussian (σ=1.0) → [optional: bilateral or TV]
```

This order matches Feature Squeezing: bit-depth squashes absolute perturbation range, median kills salt-and-pepper L0 components, Gaussian removes residual L2 content.

---

## References

| Paper | Key Contribution |
|---|---|
| Huang et al., IEEE TASP 1979 | Histogram-based O(k) median |
| Perreault & Hébert, IEEE TIP 2007 | O(1) constant-time median (CTMF) |
| Young & Van Vliet, Signal Processing 1995 | IIR Gaussian, closed-form coefficients |
| Deriche, INRIA TR 1992 | Recursive Gaussian (tabulated coefficients) |
| Triggs & Sdika, IEEE TSP 2006 | Boundary conditions for Van Vliet IIR |
| Rudin, Osher, Fatemi, Physica D 1992 | ROF total variation model |
| Chambolle, JMIV 2004 | Dual projection TV algorithm |
| Goldstein & Osher, SIAM 2009 | Split Bregman TV, FFT u-update |
| Chambolle & Pock, JMIV 2011 | Primal-dual TV, O(1/N) convergence |
| Paris & Durand, ECCV 2006 | Bilateral grid O(1) approximation |
| Chaudhury et al., IEEE TIP 2011 | O(1) trigonometric bilateral filter |
| Buades et al., CVPR 2005 | Non-local means algorithm |
| Xu et al., NDSS 2018 | Feature squeezing defense, optimal kernel sizes |
| Guo et al., ICLR 2018 | TV and quilting as best input-transform defenses |
| Ivan Kutskir / bfraboni | Fast 3-pass box blur C++ implementation |
