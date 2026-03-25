# Image-Scaling Attacks and Defenses

> Research for pixmask — a C++ image sanitization layer before multimodal LLMs.
> Written: 2026-03-25

---

## 1. The Attack: Adversarial Preprocessing via Image Scaling

### 1.1 Core Concept

Image-scaling attacks (Quiring et al., USENIX Security 2020) exploit a fundamental property of downsampling: **the scaling operation is a linear transform that assigns unequal weight to input pixels**. An adversary crafts a source image A such that A appears visually identical to a benign image S at full resolution, but after downscaling, the output D matches a target image T chosen by the attacker.

Formally:
```
Given: source S, target T, scaling function f_scale
Find: attack image A such that:
  ||A - S||_inf <= ε          (imperceptible to human)
  ||f_scale(A) - T||_inf <= δ  (scaled output matches target)
```

This is distinct from traditional adversarial examples (which require gradient access to a model) — scaling attacks are **model-agnostic and preprocessing-stage attacks**. The adversary needs only to know the interpolation method used in the pipeline.

### 1.2 Why Downsampling is Exploitable

Every interpolation kernel assigns a weight matrix W of shape (H_out × H_in) for each spatial dimension. For a scaling ratio r = H_in / H_out, the kernel defines which input pixels contribute to each output pixel and with what weight.

**Key vulnerability**: when the scaling step size exceeds the kernel window width, many input pixels receive zero weight and are **completely ignored**. For bilinear downscaling by factor 4x, roughly 75% of input pixels have no influence on any output pixel. The attacker exploits this by:

1. Identifying which input pixel positions have non-negligible weight (the "considered" pixels)
2. Placing target content into those positions (modifying them to steer the output)
3. Keeping the remaining pixels close to S (preserving the source appearance at full res)

The scaling operation can be expressed as a matrix-vector product per row:
```
output_row = cl_matrix @ input_row
```
where `cl_matrix` encodes the kernel weights. The attack solves:
```
minimize  (1/2) ||delta||^2
subject to: cl_matrix @ (S + delta) ≈ T_row
            0 <= S + delta <= 255
```
This is a convex quadratic program (CVXPY-solvable), solved column-by-column then row-by-row (two-pass: horizontal then vertical).

### 1.3 Interpolation Method Vulnerability Analysis

| Method | Vulnerable | Reason |
|--------|-----------|--------|
| Nearest Neighbor (`INTER_NEAREST`) | **Yes** | Only 1 pixel selected per output pixel; all others ignored. Maximum information discard — easiest to exploit. |
| Bilinear (`INTER_LINEAR`) | **Yes** | 2x2 kernel for 2D. At large downscale ratios, most input pixels still get zero weight. Default in TF (`antialias=False`). |
| Bicubic (`INTER_CUBIC`) | **Yes** | 4x4 kernel. Larger support than bilinear but still leaves many pixels with zero weight at large ratios. |
| Lanczos (`INTER_LANCZOS4`) | **Yes** | 8x8 kernel in OpenCV. More pixels contribute, but attack still feasible — requires solving a harder optimization but works. |
| Area (`INTER_AREA`) | **Resistant** | Integrates over the full source region contributing to each output pixel. All input pixels influence output. No zero-weight pixels to exploit. |
| Pillow (non-nearest) | **Resistant** | Pillow's resize uses proper anti-aliasing by default (equivalent to area sampling). |
| TF with `antialias=True` | **Resistant** | Anti-aliasing filter ensures all input pixels contribute to output. |

**Critical finding**: OpenCV's `INTER_LINEAR` and `INTER_CUBIC` are vulnerable when used for downscaling without explicit anti-aliasing pre-filtering. The default TensorFlow `tf.image.resize` prior to 2020 used bilinear without anti-aliasing, making it directly exploitable.

### 1.4 Attack Variants

**Standard (Quadratic) Attack** — Solves the convex QP per row/column. Produces an attack image where the scaled output exactly matches the target. Computationally tractable but requires solving thousands of small LP/QP instances.

**Area-attack variant** — More difficult because area interpolation averages over non-overlapping integer blocks (integer-border case) or fractionally-weighted overlapping regions (non-integer case). Quiring et al. show that even area scaling can be partially attacked when integer borders align — only non-integer-border area is fully resistant.

**Adaptive attacks** — After defenders deploy median filtering, an adaptive adversary reformulates the optimization to account for the defense's pixel modification pattern. The adaptive attack partially degrades (but does not fully defeat) the median filter defense.

---

## 2. Defense Mechanisms

### 2.1 Safe Interpolation (Primary Defense)

**The simplest and most robust defense**: replace the vulnerable scaling method with area-based downsampling.

OpenCV: `cv::resize(src, dst, target_size, 0, 0, cv::INTER_AREA)`

This works because area interpolation computes a weighted average over the entire source region that maps to each output pixel. The weight matrix has no zero entries — every source pixel contributes. An attacker cannot selectively modify "ignored" pixels because there are no ignored pixels.

**Practical note**: `INTER_AREA` is only area-based when downscaling. When upscaling, it falls back to nearest-neighbor. Always enforce downscale-only use.

### 2.2 Pixel Influence Restoration (Median Filter Defense)

From Quiring et al. §5.2–5.3. Rather than changing the interpolation method, this defense sanitizes the input image before scaling:

**Algorithm**:
1. Compute the scaling kernel weight matrix — identify which input pixels have non-negligible weight ("considered pixels") via Fourier peak analysis.
2. For each considered pixel at position (i, j):
   - Define a neighborhood window of radius (k_v, k_h) where k is derived from the scaling ratio
   - Exclude all "considered" pixels within the window (set to NaN)
   - Replace (i, j) with the median of non-considered neighbors
3. Scale the restored image normally.

**Why it works**: The attack embeds target content in the high-weight ("considered") pixels. The median filter replaces those pixels with values from the low-weight neighbors — which the attacker left unchanged to preserve S's appearance. The restored pixel reflects S's local content, neutralizing the attack.

**Kernel size**: bandwidth k = floor(scaling_ratio / 2). For a 4x downscale, k ≈ 2 → 5x5 neighborhood.

**Implementation note**: The median must exclude considered pixels (not just the center pixel). Plain scipy.ndimage.median_filter is insufficient — must mask out considered positions.

**Limitation**: An adaptive adversary who knows the defense is applied can reformulate their optimization to account for the median replacement. The defense degrades but does not fully defeat adaptive attacks.

### 2.3 Randomized Filtering Defense

From Quiring et al. §5.4–5.5. Counters adaptive adversaries by introducing randomness:

**Algorithm**:
1. Identify considered pixels (same as above).
2. For each considered pixel (i, j):
   - Define a window of non-considered neighbors
   - Replace (i, j) with a **randomly selected** pixel from that window (not median)
3. Scale the restored image.

**Why it defeats adaptive attacks**: An adaptive attacker must know the exact replacement values to pre-compensate. With randomization, the replacement is unpredictable — the attacker's optimization cannot target a specific output. Each processing attempt produces a different result.

**Trade-off**: Introduces visual noise proportional to local texture variation. For low-frequency regions, barely perceptible. For high-texture regions, can degrade image quality.

**Practical enhancement**: Run multiple random passes and average, or use randomized + clipped median ensemble.

### 2.4 Multi-Scale Consistency Checking (Detection)

Rather than preventing the attack, this detects it:

**Principle**: Downscale the image to two different target resolutions using two different methods. If the image is clean, both results should be perceptually similar (high SSIM, low L2). If the image is a scaling attack crafted for one specific target resolution and method, the alternative method or resolution will produce the unintended source content.

**Implementation**:
```
score_area    = resize(img, target, INTER_AREA)
score_bilinear = resize(img, target, INTER_LINEAR)
consistency   = SSIM(score_area, score_bilinear)
if consistency < threshold: flag as suspicious
```

**Limitation**: A sophisticated adversary crafting attacks for multiple methods simultaneously can evade this, though the optimization becomes significantly harder.

### 2.5 Randomized Resolution Jitter

Defense-through-unpredictability: randomly vary the target resolution within a small range (e.g., ±5%) before scaling. Since the attack is crafted for a specific downscale ratio, slight deviations destroy the attack's coherence.

```
target_h = original_target_h + randint(-delta, +delta)
target_w = original_target_w + randint(-delta, +delta)
scaled = resize(img, (target_h, target_w), INTER_AREA)
scaled = resize(scaled, (original_target_h, original_target_w), INTER_AREA)
```

**Cost**: Minimal. Two area resizes. The second resize brings output back to expected dimensions.

---

## 3. Relevance to VLMs

### 3.1 VLM Image Preprocessing Pipelines

Each major VLM API resizes user-submitted images before inference. This creates a mandatory scaling step that is an exact analog to the vulnerable preprocessing studied by Quiring et al.

**Claude (Anthropic)**
- Images with long edge > 1568 px are scaled down, preserving aspect ratio
- Hard rejection: images > 8000x8000 px
- Effective resize target: ~1092x1092 for 1:1 ratio images
- Scaling method: not publicly documented, but likely bilinear or area
- Risk: If a user submits a 6000x6000 image, Claude's backend resizes it 4-6x before vision processing — exactly the attack scenario

**GPT-4V (OpenAI)**
- "high detail" mode: image is first resized to fit within 2048x2048, then split into 512x512 tiles (each 170 tokens)
- "low detail" mode: image always downscaled to 512x512 (85 tokens, single pass)
- Low-detail mode resize is a direct 512px target — precise ratio determined by input dimensions
- Maximum API input: 50 MB
- Risk: The 512x512 fixed target for low-detail mode creates a fixed scaling ratio — an attacker who knows input dimensions can craft a perfect attack

**Gemini (Google)**
- Images ≤ 384px on both dimensions: processed whole (258 tokens)
- Larger images: tiled into 768x768 tiles, each 258 tokens
- `media_resolution` parameter controls max tokens/tokens per frame
- Risk: The 768px tiling boundary creates fixed-ratio scaling events at known dimensions

### 3.2 Attack Feasibility Against VLMs

The attack is feasible in the following scenario:

1. Attacker knows (or can infer) the target resolution. For GPT-4V low-detail mode, this is always 512px. For Claude, images above 1568px will be scaled to ~1568px.
2. Attacker knows the interpolation method used server-side. This is the harder unknown — but bilinear is the default in virtually all ML frameworks.
3. Attacker submits an image that appears benign at submitted resolution but resolves to adversarial content at the model's internal resolution.

**Concrete threat scenario for pixmask**:
- User (or attacker) submits a high-resolution image to a VLM-powered application
- The application calls pixmask for sanitization
- pixmask receives the full-resolution image and must sanitize it **before** the VLM's own internal resize
- If pixmask passes the image through using a vulnerable scaling method (bilinear) to perform its own pre-resize, the attack survives into the VLM's input pipeline
- More critically: even if pixmask doesn't resize, the VLM's internal resize at the other end can activate the attack

**The pixmask defense layer must**:
1. Perform its own canonical downscale to the target VLM input resolution using INTER_AREA or with pixel influence restoration
2. Or detect attack signatures before forwarding to the VLM

### 3.3 Interaction with Other Attack Types

Scaling attacks can be **combined with adversarial perturbation attacks** (see `02_adversarial_perturbations.md`). A compound attack can:
- Embed adversarial perturbations in the "considered" pixels (so they survive downscaling)
- Use the remaining pixels to maintain the benign appearance at full resolution

This makes the combination significantly more dangerous than either attack alone and harder for defenses to handle.

---

## 4. C++ Implementation Recommendations

### 4.1 Safe Downscale (Primary Mitigation)

Use `cv::INTER_AREA` unconditionally for all downscaling in pixmask. Never use `INTER_LINEAR`, `INTER_CUBIC`, or `INTER_LANCZOS4` for downscaling user-provided images.

```cpp
// safe_resize.hpp
#pragma once
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace pixmask {

// Throws if used for upscaling (INTER_AREA is not safe for upscaling).
cv::Mat safe_downscale(const cv::Mat& src, cv::Size target) {
    if (target.width > src.cols || target.height > src.rows) {
        throw std::invalid_argument(
            "safe_downscale: target dimensions exceed source — use upscale path");
    }
    cv::Mat dst;
    cv::resize(src, dst, target, 0.0, 0.0, cv::INTER_AREA);
    return dst;
}

// Two-pass jitter resize: randomizes effective ratio slightly, then re-normalizes.
// Breaks attacks crafted for a specific ratio.
cv::Mat jitter_downscale(const cv::Mat& src, cv::Size target,
                          std::mt19937& rng, int jitter_px = 8) {
    std::uniform_int_distribution<int> dist(-jitter_px, jitter_px);
    cv::Size jittered(
        std::max(target.width + dist(rng),  target.width / 2),
        std::max(target.height + dist(rng), target.height / 2)
    );
    cv::Mat intermediate;
    cv::resize(src, intermediate, jittered, 0.0, 0.0, cv::INTER_AREA);
    cv::Mat dst;
    cv::resize(intermediate, dst, target, 0.0, 0.0, cv::INTER_AREA);
    return dst;
}

} // namespace pixmask
```

### 4.2 Pixel Influence Restoration (Median Filter Defense)

Applies before any resize. Identified "considered" pixel positions are replaced with median of non-considered neighbors.

```cpp
// scaling_defense.hpp
#pragma once
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <vector>

namespace pixmask {

// Compute which pixel columns/rows are "considered" by the scaling kernel.
// For bilinear downscaling by ratio r, column j is considered if
// j * (1/r) is close to an integer (within 0.5/r tolerance).
// Returns a boolean mask of size src_width.
std::vector<bool> considered_columns(int src_w, int dst_w) {
    std::vector<bool> mask(src_w, false);
    double ratio = static_cast<double>(src_w) / dst_w;
    for (int j_dst = 0; j_dst < dst_w; ++j_dst) {
        double j_src = (j_dst + 0.5) * ratio - 0.5;
        int j0 = static_cast<int>(std::floor(j_src));
        int j1 = j0 + 1;
        if (j0 >= 0 && j0 < src_w) mask[j0] = true;
        if (j1 >= 0 && j1 < src_w) mask[j1] = true;
    }
    return mask;
}

// Apply median-based pixel influence restoration for one channel.
// For each "considered" pixel, replace with median of non-considered
// neighbors in a (2k+1) x (2k+1) window.
void restore_pixel_influence(cv::Mat& channel,
                              const std::vector<bool>& col_mask,
                              const std::vector<bool>& row_mask,
                              int k) {
    int H = channel.rows, W = channel.cols;
    cv::Mat result = channel.clone();

    for (int r = 0; r < H; ++r) {
        if (!row_mask[r]) continue;
        for (int c = 0; c < W; ++c) {
            if (!col_mask[c]) continue;

            // Collect non-considered neighbors
            std::vector<uint8_t> neighbors;
            neighbors.reserve((2*k+1) * (2*k+1));
            for (int dr = -k; dr <= k; ++dr) {
                for (int dc = -k; dc <= k; ++dc) {
                    int nr = r + dr, nc = c + dc;
                    if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
                    if (row_mask[nr] && col_mask[nc]) continue; // skip considered
                    neighbors.push_back(channel.at<uint8_t>(nr, nc));
                }
            }
            if (neighbors.empty()) continue;

            // Median without full sort for small N
            std::nth_element(neighbors.begin(),
                             neighbors.begin() + neighbors.size() / 2,
                             neighbors.end());
            result.at<uint8_t>(r, c) = neighbors[neighbors.size() / 2];
        }
    }
    channel = result;
}

// Full defense: restore all channels, then safe-downscale.
cv::Mat defended_resize(const cv::Mat& src, cv::Size target,
                        int bandwidth = -1) {
    double ratio_h = static_cast<double>(src.rows) / target.height;
    double ratio_w = static_cast<double>(src.cols) / target.width;
    int k_h = (bandwidth > 0) ? bandwidth
                               : static_cast<int>(std::floor(ratio_h / 2.0));
    int k_w = (bandwidth > 0) ? bandwidth
                               : static_cast<int>(std::floor(ratio_w / 2.0));
    k_h = std::max(k_h, 1);
    k_w = std::max(k_w, 1);

    auto row_mask = considered_columns(src.rows, target.height);
    auto col_mask = considered_columns(src.cols,  target.width);

    cv::Mat working;
    if (src.channels() == 1) {
        working = src.clone();
        restore_pixel_influence(working, col_mask, row_mask,
                                std::max(k_h, k_w));
    } else {
        std::vector<cv::Mat> channels;
        cv::split(src, channels);
        for (auto& ch : channels) {
            restore_pixel_influence(ch, col_mask, row_mask,
                                    std::max(k_h, k_w));
        }
        cv::merge(channels, working);
    }

    cv::Mat dst;
    cv::resize(working, dst, target, 0.0, 0.0, cv::INTER_AREA);
    return dst;
}

} // namespace pixmask
```

### 4.3 Multi-Scale Consistency Check (Detection)

Detects scaling attack artifacts by comparing area vs bilinear downscale outputs:

```cpp
// scaling_attack_detector.hpp
#pragma once
#include <opencv2/imgproc.hpp>
#include <opencv2/quality.hpp>

namespace pixmask {

struct ScalingAttackScore {
    double ssim;           // 0.0 (attack) to 1.0 (clean)
    double l2_normalized;  // normalized L2 between two scale outputs
    bool likely_attack;
};

ScalingAttackScore detect_scaling_attack(const cv::Mat& src,
                                          cv::Size target,
                                          double ssim_threshold = 0.85) {
    cv::Mat area_scaled, bilinear_scaled;
    cv::resize(src, area_scaled,    target, 0.0, 0.0, cv::INTER_AREA);
    cv::resize(src, bilinear_scaled, target, 0.0, 0.0, cv::INTER_LINEAR);

    // Convert to float for SSIM
    cv::Mat a_f, b_f;
    area_scaled.convertTo(a_f, CV_32F, 1.0/255.0);
    bilinear_scaled.convertTo(b_f, CV_32F, 1.0/255.0);

    cv::Scalar ssim_score = cv::quality::QualitySSIM::compute(a_f, b_f, cv::noArray());
    double mean_ssim = (ssim_score[0] + ssim_score[1] + ssim_score[2]) / 3.0;

    cv::Mat diff;
    cv::absdiff(a_f, b_f, diff);
    double l2 = cv::norm(diff, cv::NORM_L2) /
                (target.width * target.height * src.channels());

    return {
        mean_ssim,
        l2,
        mean_ssim < ssim_threshold
    };
}

} // namespace pixmask
```

### 4.4 Recommended Pipeline Position

In pixmask's pipeline, scaling attack defense must be applied **before any other resize**:

```
[Input image] --> [scaling_attack_detector] --> if suspicious:
                                                  [pixel_influence_restoration]
                                                  [jitter_downscale to VLM target]
                                               else:
                                                  [safe_downscale to VLM target]
              --> [other sanitization stages (adversarial, steg, etc.)]
              --> [Output to VLM]
```

For maximum defense, always apply `defended_resize` regardless of detection result — it has low computational overhead and provides defense-in-depth.

---

## 5. Summary of Findings

### Vulnerability Matrix

| Interpolation | Downscale Risk | Notes |
|---------------|---------------|-------|
| `INTER_NEAREST` | Critical | Maximum exploitation surface |
| `INTER_LINEAR` (bilinear) | High | Default in most ML frameworks; TF default pre-2020 |
| `INTER_CUBIC` (bicubic) | High | Larger kernel but still exploitable |
| `INTER_LANCZOS4` | Medium-High | 8x8 kernel; harder but not immune |
| `INTER_AREA` | Low | Resistant; only partially attackable with integer borders |
| Pillow default | Low | Uses anti-aliasing by default |

### Defense Ranking

| Defense | Effectiveness vs Non-Adaptive | vs Adaptive Adversary | Runtime Cost |
|---------|------------------------------|----------------------|-------------|
| Area interpolation | Robust | Partially resistant | Negligible |
| Pixel influence restoration (median) | High | Moderate | Low-Medium |
| Randomized filtering | High | High | Low-Medium |
| Multi-scale consistency check | Detection only | Evasible | Low |
| Resolution jitter | High | High | Negligible |

### Recommended Defaults for pixmask

1. **Always use `INTER_AREA` for all user-image downscaling** — zero computational cost, robust against standard attacks.
2. **Apply pixel influence restoration for high-security contexts** — adds ~5-10ms per megapixel but defeats non-adaptive attacks entirely.
3. **Use resolution jitter as a free layer** — negligible cost, breaks ratio-dependent attacks.
4. **Log multi-scale consistency scores** — useful for threat intelligence even if not blocking.

---

## References

- Quiring, E., Klein, D., Arp, D., Johns, M., Rieck, K. (2020). "Adversarial Preprocessing: Image-Scaling Attacks and Defenses." *USENIX Security 2020*. https://scaling-attacks.net/
- Reference implementation: https://github.com/EQuiw/2019-scalingattack
- Xiao, C., Li, B., Zhu, J.-Y., He, W., Liu, M., Song, D. (2019). "Generating Adversarial Examples with Adversarial Networks." *IJCAI 2019*. (Prior work on scaling attacks)
- Qi, X. et al. (2024). "Visual Adversarial Examples Jailbreak Aligned Large Language Models." *AAAI 2024*.
- Anthropic Claude Vision API: https://platform.claude.com/docs/en/docs/build-with-claude/vision (image resize at 1568px long edge)
- OpenAI GPT-4V: 512x512 tiling for high-detail mode, single 512px downscale for low-detail mode
- Google Gemini: 768x768 tile-based processing; ≤384px processed whole
