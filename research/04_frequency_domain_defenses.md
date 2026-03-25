# Frequency-Domain Defenses Against Adversarial Images

> Research for pixmask — C++ image sanitization layer before multimodal LLMs.
> Primary sources: Prakash et al. CVPR 2018 (PDF), Guo et al. ICLR 2018 (PDF), Wang et al. CVPR 2020 (PDF), FFTW 3.3.10 docs.

---

## 1. DCT-Based Defenses

### 1.1 JPEG Compression as a Defense

JPEG compression is the oldest and most battle-tested frequency-domain defense. The mechanism is block DCT → quantize → entropy-code → decode; the quantization step is where adversarial signal is destroyed.

**Standard pipeline (8x8 block DCT):**
1. Convert image to YCbCr
2. Partition each channel into non-overlapping 8×8 blocks
3. Apply 2D DCT-II to each block — produces 64 coefficients per block
4. Divide each coefficient by the corresponding entry in the quantization matrix `Q`, round to nearest integer
5. Entropy-code (Huffman/arithmetic) and write to file
6. On decode: multiply rounded coefficients back by `Q`, then apply IDCT

The luminance quantization matrix at quality 50 (IJG standard) has values ranging from 2 (DC) to 61 (highest AC frequency corner). At quality 75 the matrix is divided by ~1.5; at quality 10 it is multiplied by ~5. High-frequency DCT coefficients (bottom-right corner of the 8×8 block) get the largest quantization divisors, so they round to zero first as quality decreases.

**Quality factor → quantization scale (IJG formula):**
```
if quality < 50:  scale = 5000 / quality
else:             scale = 200 - 2 * quality
Q_effective[i] = clamp(round(Q_base[i] * scale / 100), 1, 255)
```

At quality 75: scale = 50 → coefficients divided by ~0.5× their base value (mild).
At quality 10: scale = 500 → coefficients divided by 5× (aggressive zeroing of high AC).

**Guo et al. (ICLR 2018) findings — extracted from paper PDF:**
- Tested JPEG at quality level 75 (out of 100) against FGSM, I-FGSM, DeepFool, CW-L2 on ImageNet/ResNet-50.
- JPEG at Q=75 is a weak defense in the gray-box setting: "bit-depth reduction and JPEG compression are weak defenses in such a gray-box setting." Adversaries can adapt to the differentiable approximation of JPEG.
- JPEG is more useful as part of an ensemble (with TVM and quilting) than standalone.
- In the black-box (adversary unaware of defense) setting, JPEG provides meaningful protection.
- Bit-depth reduction to 3 bits and JPEG Q=75 have similar effect: both collapse small adversarial variations onto quantized values.

**Prakash et al. (CVPR 2018) comparison — extracted from paper PDF:**
- JPEG alone (no pixel deflection): 49.1% accuracy against FGSM, 31.2% against I-FGSM, 61.1% against DFool, 95.4% against C&W (at |L2|~0.04).
- JPEG is most effective against C&W (which produces smooth, low-amplitude perturbations) but fails against iterative attacks (I-FGSM).
- Wavelet denoising alone: 40.6% FGSM, 31.2% I-FGSM — comparable to JPEG.
- Pixel Deflection + Wavelet Denoising substantially outperforms both: 79.7% FGSM, 82.4% I-FGSM.
- "We have observed that for the perturbations added by well-known attacks, wavelet denoising yields superior results as compared to block DCT."

### 1.2 Selective DCT Coefficient Zeroing

Rather than using JPEG's fixed quantization tables, a targeted defense can zero high-frequency DCT coefficients directly.

**Block-level approach:**
- Process each 8×8 (or 16×16) block with 2D DCT
- Zero all coefficients beyond a diagonal threshold: if `i + j > threshold`, set `coeff[i][j] = 0`
- Apply IDCT — equivalent to a spatial low-pass filter with block-periodic artifacts

**Threshold selection:**
- `threshold = 4` (out of 14 max): keeps only the DC and very low AC components, strong defense but visible blurring
- `threshold = 8`: moderate — preserves edges, removes fine texture noise
- `threshold = 11-12`: minimal, removes only the highest-frequency content

The zigzag scan order in JPEG maps these to the first N coefficients: keeping coefficients 0–15 (of 64) is roughly equivalent to quality factor ~30.

**Limitation:** Block DCT introduces 8×8 grid artifacts. For adversarial perturbations that are not grid-aligned, some energy leaks back into lower-frequency coefficients due to the block boundary discontinuity (Gibbs phenomenon). Overlapping blocks (MDCT, wavelets) avoid this.

---

## 2. FFT / DFT Low-Pass Filtering

### 2.1 Circular Low-Pass Mask

The DFT of an H×W image produces a H×W complex spectrum. After `fftshift`, the DC component is at center `(H/2, W/2)`. A circular low-pass mask zeroes all frequencies with radius > cutoff from center:

```
for i in [0, H):
  for j in [0, W):
    d = sqrt((i - H/2)^2 + (j - W/2)^2)
    if d > cutoff:
      F[i][j] = 0
```

**Cutoff frequency selection:**
- The spectrum of natural images follows an approximate 1/f^2 power law: most energy is at low frequencies.
- Adversarial perturbations (FGSM, PGD) inject energy across the full spectrum, but the visible constraint (small L-inf norm) means the adversarial signal is spread broadly.
- Wang et al. (CVPR 2020) use radius parameter `r` with values {4, 8, 12, 16} on CIFAR-10 (32×32 images). At `r=4`, virtually all adversarial energy is removed but so is most texture information; at `r=16` texture is preserved but so is much adversarial signal.
- For 224×224 ImageNet images, cutoff radius of ~20–30 pixels (out of 112 max) is a reasonable starting point for aggressive filtering. This corresponds to retaining spatial frequencies up to ~0.09–0.13 cycles/pixel.

**Wang et al. (CVPR 2020) key finding:**
> "For these examples, the prediction outcomes are almost entirely determined by the high-frequency components of the image, which are barely perceivable to human."

CNNs trained on natural labels learn to exploit HFC. When only low-frequency components are fed (low-pass filtered images), classification accuracy drops even though the images look identical to humans. This confirms that adversarial perturbations target the HFC that CNNs rely on.

**Experimental results (Wang et al., CIFAR-10, ResNet-18, Table 1):**

| Radius `r` (LFC) | Train Acc | Test Acc |
|---|---|---|
| 4 | 96.7% | 61.7% |
| 8 | 97.9% | 71.5% |
| 12 | 97.9% | 75.2% |
| 16 | 98.4% | 77.1% |

Low-pass filtering (keeping only LFC) degrades accuracy even on clean images, demonstrating the fundamental robustness-accuracy tradeoff.

### 2.2 Defense Implications

- A cutoff that removes HFC sufficient to defeat adversarial attacks will also degrade useful texture features for classification.
- FFT low-pass is better suited as a preprocessing step for VLMs/multimodal models (which rely more on semantic content) than for image classifiers tuned on texture statistics.
- Per-channel FFT on YCbCr: filter Y more aggressively than CbCr (human perception is less sensitive to chroma detail).

---

## 3. Wavelet-Based Defenses

### 3.1 Wavelet Soft-Thresholding

Primary source: Prakash et al. CVPR 2018 (arXiv:1801.08926), extracted from PDF.

**Full algorithm (from paper Section 8):**

```
(a) Convert image from RGB to YCbCr
(b) Apply discrete wavelet transform (DWT) to each channel
    -- Wavelet family: db1 (Daubechies-1 = Haar); db2 also tested with similar results
(c) Soft-threshold all wavelet coefficients using BayesShrink
(d) Apply inverse DWT on thrunken coefficients
(e) Convert image back to RGB
```

**Why YCbCr before wavelet:** YCbCr has perceptually meaningful separation; the luminance channel (Y) carries most adversarial signal. The color space transform also decouples luma noise from chroma noise, allowing per-channel threshold tuning.

### 3.2 Wavelet Families

Tested in the paper: db1 (Haar), db2. Similar results were obtained with both. Haar (db1) is the simplest and fastest — piecewise constant basis. db2 has a slightly smoother basis (support length 4 vs 2).

**Summary of wavelet family tradeoffs for adversarial defense:**

| Family | Support | Regularity | Notes |
|---|---|---|---|
| Haar (db1) | 2 | C^0 | Fastest; block artifacts at low thresholds |
| Daubechies db2 | 4 | C^0.34 | Slightly smoother, marginal improvement |
| Daubechies db4 | 8 | C^1.3 | Better PSNR on natural images; more expensive |
| Symlet sym4 | 8 | C^1.3 | Near-symmetric, less ringing than db4 |
| Coiflet coif1 | 6 | C^0.7 | Good for natural images; used in JPEG2000 |

For adversarial defense, the paper found db1 sufficient. The thresholding operation, not the wavelet family, dominates the result. Higher-order wavelets become relevant if reconstruction quality (PSNR) is the primary concern.

### 3.3 Thresholding Strategies

**Hard thresholding (JPEG-style):**
```
Q(x_hat) = x_hat * (|x_hat| > T)
```
Sets sub-threshold coefficients to zero. Produces ringing/block artifacts. The paper explicitly notes: "hard thresholding results in over-blurring of the input image, while soft thresholding maintains better PSNR."

**Soft thresholding (used in this defense):**
```
Q(x_hat) = sign(x_hat) * max(0, |x_hat| - T)
```
Shrinks all coefficients by T, zeroing those below threshold. Avoids the discontinuity of hard thresholding. Better PSNR and avoids extraneous noise reintroduction.

**Threshold determination — BayesShrink (from paper eq. 6):**
```
T_Bayes ≈ sigma_noise^2 / sigma_x
```
where `sigma_noise` is the estimated noise variance (hyperparameter σ) and `sigma_x` is the estimated signal variance in each wavelet sub-band. BayesShrink adapts the threshold per sub-band by modeling coefficients as a Generalized Gaussian Distribution (GGD).

Compare to VisuShrink (universal threshold):
```
T_Visu = sigma * sqrt(2 * log(N))
```
where N is number of pixels. VisuShrink is a single global threshold; BayesShrink is per-sub-band.

**Ablation on threshold type (from paper Table in Section 10.3, sigma=0.04):**

| Method | Clean | FGSM | I-FGSM | DFool | C&W |
|---|---|---|---|---|---|
| Hard thresh. | 39.5% | 35.9% | - | - | - |
| VisuShrink | 96.1% | - | - | - | - |
| SURE thresh. | 92.1% | - | - | - | - |
| BayesShrink | 98.9% | 81.5% | 83.7% | 90.3% | 98.0% |

BayesShrink dominates substantially. Hard thresholding is essentially unusable (destroys image).

### 3.4 Pixel Deflection + Wavelet Denoising (Full Defense)

**Algorithm 1 (Pixel Deflection) — directly from paper:**
```
Input:  Image I, neighborhood size r, deflections K
Output: Image I' (same dimensions)

for i = 0 to K:
    p_i ~ U(I)               # uniform random pixel
    n_i ~ U(R_r_p ∩ I)       # uniform random pixel in r×r neighborhood of p
    I'[p_i] = I[n_i]
```

**Hyperparameters (validated on 300 ImageNet images):**
- Window size `r = 10` (10×10 neighborhood)
- Deflections `K = 100`
- BayesShrink `sigma = 0.04`
- Sampling: uniform within square neighborhood (Gaussian sampling was inferior)
- Replacement: random pixel value from neighborhood (mean/min/max all inferior)

**Results — Table 3 (sigma=0.04, window=10, deflections=100) on ResNet-50, VGG-19, Inception-v3:**

| Attack | |L2| | No Defense | Single | Ens-10 |
|---|---|---|---|---|
| Clean | 0.00 | 100% | 98.3% | 98.9% |
| FGSM | 0.05 | 20.0% | 79.9% | 81.5% |
| I-FGSM | 0.03 | 14.1% | 83.7% | 83.7% |
| DeepFool | 0.02 | 26.3% | 86.3% | 90.3% |
| JSMA | 0.02 | 25.5% | 91.5% | 97.0% |
| L-BFGS | 0.02 | 12.1% | 88.0% | 91.6% |
| C&W | 0.04 | 4.8% | 92.7% | 98.0% |

**Comparison vs. other defenses (Destruction Rate, Table 4):**

| Defense | FGSM | I-FGSM | DFool | C&W |
|---|---|---|---|---|
| Feature Squeezing (best combo) | 0.434 | 0.644 | 0.786 | 0.915 |
| Quilting + TVM (Guo et al.) | 0.629 | 0.882 | 0.883 | 0.859 |
| PD (pixel deflection only) | 0.735 | 0.880 | 0.914 | 0.931 |
| PD + R-CAM | 0.746 | 0.912 | 0.911 | 0.952 |
| **PD + R-CAM + DWT** | **0.769** | **0.927** | **0.948** | **0.981** |

PD + Robust CAM weighting + DWT soft-threshold achieves the best results across all four attacks.

---

## 4. Spectral Analysis of Adversarial Perturbations

### 4.1 Core Finding (Wang et al. CVPR 2020)

**Paper:** "High-Frequency Component Helps Explain the Generalization of Convolutional Neural Networks", Haohan Wang, Xindi Wu, Zeyi Huang, Eric Xing, CVPR 2020.

**Decomposition method:** Circular mask in Fourier domain with radius `r` (equation from paper):
```
z = F(x)                          # Fourier transform
zl(i,j) = z(i,j) if d((i,j),(cH,cW)) <= r else 0    # low-freq
zh(i,j) = 0       if d((i,j),(cH,cW)) <= r else z(i,j)  # high-freq
xl = F^{-1}(zl), xh = F^{-1}(zh)
```
(complex parts discarded after IFFT)

**Key experimental result (CIFAR-10, Table 1):**
- CNN trained and tested on HFC (xh) alone: train 97-99%, test 9-20% — models overfit to HFC
- CNN trained and tested on LFC (xl) alone: train 97-98%, test 61-77% — more robust generalization
- Adversarial perturbations exploit HFC: models whose predictions are determined by xh are highly adversarially vulnerable

**Adversarial robustness implication (Table 2, ResNet-18, CIFAR-10):**
- Vanilla model: FGSM accuracy 10.7% (ε=0.03), PGD 0.3%
- Adversarial training: FGSM 43.5%, PGD 40.3%
- Adversarial training smooths first-layer kernels, making them less responsive to HFC

**Defense from frequency perspective:** "smooth the convolution kernel" by mixing adjacent weights (eq. 3 in paper). With ρ=1.0, improves FGSM robustness on naturally-trained model from 10.7% to 17.1%. Cheap but limited.

### 4.2 Which Attacks Concentrate in Which Frequency Bands

From Guo et al. (ICLR 2018) and Prakash et al. (CVPR 2018) PDF analysis:

| Attack | Primary Frequency Band | L-p Norm | Notes |
|---|---|---|---|
| FGSM | Broadband (sign of gradient) | L-inf | Sign operation spreads energy uniformly |
| I-FGSM / PGD | Primarily HFC with L-inf budget | L-inf | Iterative refinement concentrates at boundaries |
| DeepFool | Mostly LFC + mid-frequency | L-2 | Minimal norm perturbation, structured |
| C&W | LFC-structured, smooth | L-2 | Optimization finds perceptually-smooth perturbations |
| JSMA | Sparse, spatial domain | L-0 | Not frequency-band specific |

Pratice implication: L-inf attacks (FGSM, PGD) are best countered by HFC removal (low-pass filter). L-2 attacks (C&W, DeepFool) require more sophisticated approaches since they can embed perturbations in low frequencies.

From Prakash et al., Section 7: "Since adversarial attacks do not take into account the frequency content of the perturbed image, they are likely to pull the input away from the class of likely natural images in a way which can be detected and corrected using a multi-resolution analysis."

### 4.3 Frequency Band Intuition

Natural images: ~96% of spectral energy in lowest 10% of frequencies (1/f^2 power spectrum). Adversarial perturbations under L-inf constraint: energy is roughly flat across frequencies (since sign() operation in FGSM is flat-spectrum). Low-pass filtering with cutoff at 10% bandwidth removes ~90% of adversarial energy while retaining ~96% of natural image content by energy. However, the classification-relevant signal in CNNs includes significant HFC, so accuracy still degrades.

---

## 5. C++ Implementation Details

### 5.1 FFTW for 2D FFT Low-Pass Filtering

FFTW 3.3.10 is the standard high-performance FFT library for C++.

**2D real-to-complex plan (optimal for image data):**
```cpp
#include <fftw3.h>

// Allocate SIMD-aligned arrays (required for SSE/AVX)
float* in  = fftwf_alloc_real(H * W);            // input: H*W real
fftwf_complex* out = fftwf_alloc_complex(H * (W/2 + 1));  // r2c half-spectrum

// Create plan once per image size (expensive; reuse for all images of same size)
fftwf_plan plan_fwd = fftwf_plan_dft_r2c_2d(H, W, in, out, FFTW_MEASURE);
fftwf_plan plan_inv = fftwf_plan_dft_c2r_2d(H, W, out, in, FFTW_MEASURE);

// Execute (cheap, reuse plan)
fftwf_execute(plan_fwd);  // in -> out (frequency domain)

// Apply circular low-pass mask
float cx = H / 2.0f, cy = W / 2.0f;
for (int i = 0; i < H; ++i) {
    for (int j = 0; j < W/2 + 1; ++j) {
        // Account for fftshift: DC is at (0,0) in r2c output
        float fi = (i < H/2) ? i : i - H;  // centered freq index
        float fj = j;
        float d = sqrtf(fi*fi + fj*fj);
        if (d > cutoff_radius) {
            out[i*(W/2+1) + j][0] = 0.0f;  // zero real part
            out[i*(W/2+1) + j][1] = 0.0f;  // zero imag part
        }
    }
}

// Inverse transform
fftwf_execute(plan_inv);

// Normalize (FFTW does not normalize by default)
for (int k = 0; k < H * W; ++k)
    in[k] /= (H * W);
```

**Memory layout:** FFTW r2c stores the half-spectrum (H × (W/2+1) complex values) due to Hermitian symmetry of real input. Row-major C order. Input rows are H, output rows are also H but columns are W/2+1.

**SIMD alignment:**
- `fftwf_alloc_real` / `fftwf_alloc_complex` guarantee 16-byte alignment for SSE2, 32-byte for AVX
- SSE/SSE2/AVX/AVX2/AVX-512 are auto-detected at compile time if FFTW is built with `--enable-avx2 --enable-avx512`
- Use `FFTW_MEASURE` flag for plan creation: FFTW benchmarks multiple FFT algorithms and selects the fastest for the given (H, W). This is slow at plan creation (~seconds) but negligible at execute time. Use `FFTW_ESTIMATE` only for one-off transforms.

**Wisdom for batch processing (plan caching):**
```cpp
// After creating plan, export for reuse in subsequent runs
fftwf_export_wisdom_to_filename("fftw_wisdom.dat");

// On startup in subsequent runs:
fftwf_import_wisdom_from_filename("fftw_wisdom.dat");
// Then create plans with FFTW_MEASURE — uses saved wisdom
```

**Multi-threaded FFTW (for large images):**
```cpp
#include <fftw3.h>
fftwf_init_threads();           // must be called before any other FFTW call
fftwf_plan_with_nthreads(4);    // set thread count for subsequent plan creation
// Plans created after this call will use 4 threads
```
Thread parallelism is automatic within a single plan execution; no caller-side synchronization needed.

**In-place 2D FFT:**
```cpp
// In-place: pass same pointer for in and out
// For r2c, in-place requires input to have padded row width: 2*(W/2+1)
float* buf = fftwf_alloc_real(H * 2 * (W/2 + 1));  // padded
fftwf_complex* cbuf = reinterpret_cast<fftwf_complex*>(buf);
fftwf_plan p = fftwf_plan_dft_r2c_2d(H, W, buf, cbuf, FFTW_MEASURE);
```

### 5.2 Fast DCT Implementation

**libjpeg-turbo** is the standard for production JPEG compression (2–7x faster than libjpeg). It uses SIMD-accelerated 8×8 DCT with separate forward and inverse integer DCT paths.

```cpp
// Adversarial defense via libjpeg-turbo re-encode/decode
#include <turbojpeg.h>

tjhandle h_enc = tjInitCompress();
tjhandle h_dec = tjInitDecompress();

unsigned char* jpegBuf = nullptr;
unsigned long  jpegSize = 0;

// Compress at quality Q (50-75 for defense; lower = stronger filtering)
tjCompress2(h_enc, src_rgb, W, 0, H, TJPF_RGB,
            &jpegBuf, &jpegSize, TJSAMP_444, quality, TJFLAG_FASTDCT);

// Decompress back to RGB
tjDecompress2(h_dec, jpegBuf, jpegSize, dst_rgb, W, 0, H, TJPF_RGB, TJFLAG_FASTDCT);

tjFree(jpegBuf);
tjDestroy(h_enc);
tjDestroy(h_dec);
```

**Custom block DCT for selective coefficient zeroing (without full JPEG):**
```cpp
// 8x8 DCT using libjpeg internal DCT (not exposed in public API)
// Alternative: use OpenCV's dct() function on 8x8 Mat blocks
// cv::dct(block_f32, dct_block, 0);
// Zero high-frequency coefficients: set dct_block[i][j] = 0 for i+j > threshold
// cv::dct(dct_block, block_f32, cv::DCT_INVERSE);
```

OpenCV's `cv::dct()` uses the same 8×8 or arbitrary-size DCT. For 8×8 blocks, it internally uses a fast algorithm with ~5 multiplications per output (the factored Chen-Wang algorithm).

**SIMD-accelerated DCT without external libraries:**
- Intel IPP provides SIMD-optimized DCT-II/III for 8×8 blocks: `ippiDCT8x8Fwd_16s_C1I`, `ippiDCT8x8Inv_16s_C1I`
- For custom AVX2 implementation: Lee's recursive DCT factorization reduces N-point DCT to N/2-point DCTs recursively; each butterfly stage maps to 8-wide AVX2 ymm operations

### 5.3 Block-Based Processing for Large Images

For a 4K image (3840×2160), processing the full FFT is practical. Block-based DCT for custom coefficient manipulation:

```cpp
// Block-based DCT processing
constexpr int BLOCK = 8;
for (int y = 0; y < H; y += BLOCK) {
    for (int x = 0; x < W; x += BLOCK) {
        // Extract 8x8 block -> float
        // Apply forward DCT
        // Zero coefficients where i+j > threshold
        // Apply inverse DCT
        // Write back
    }
}
```

For the FFT low-pass approach: processing the full image as one FFT is correct and efficient (no block artifacts). A 4K image FFT with FFTW takes ~10–50ms on a modern CPU depending on SIMD support.

For wavelet-based processing, the DWT operates in-place with O(N) complexity using filter bank convolutions — faster than FFT for the same image.

### 5.4 Memory Efficiency

- **In-place FFT:** Requires padding to 2*(W/2+1) per row for real data. Saves one allocation but complicates row indexing.
- **Ping-pong buffers:** Better cache behavior — allocate two aligned buffers, use one for input and one for output, swap per-channel.
- **Per-channel processing:** Process Y, Cb, Cr (or R, G, B) channels independently. For 8-bit RGB images, convert to float32 (4x memory) only for the channel being processed if memory-constrained.
- **FFTW plan reuse:** Creating a plan for (H=224, W=224) once and reusing it across all images of that size is the critical optimization. Plan creation is O(seconds); execution is O(milliseconds).

---

## 6. Defense Tradeoff Summary

| Defense | Attack Type Addressed | Clean Image Impact | Adaptive Attack Resilience |
|---|---|---|---|
| JPEG Q=75 | Broadband noise | Minimal (texture loss) | Weak (differentiable approx exists) |
| JPEG Q=30 | All HFC | Visible blurring | Moderate |
| FFT low-pass (r=20 for 224px) | HFC-dominant (FGSM/PGD) | Moderate (texture degradation) | Weak (differentiable) |
| Selective DCT zeroing (i+j>8) | High-frequency AC | Moderate blurring | Weak (differentiable) |
| Wavelet soft-thresh (BayesShrink) | Multi-band | Minimal with adaptive T | Moderate |
| Pixel Deflection + DWT | All (tested FGSM/I-FGSM/DFool/C&W) | 1-2% accuracy drop | Strong (stochastic) |

The stochastic non-differentiability of Pixel Deflection is the key to its adaptive attack resilience. Pure frequency filtering (JPEG, FFT, DCT) is differentiable and can be circumvented by an attacker who knows the defense.

---

## 7. Implementation Priority for pixmask

Based on the research, the recommended implementation order is:

1. **JPEG re-encode/decode via libjpeg-turbo** (quality 50–75): Easiest, fastest to implement, good baseline. Use `TJFLAG_FASTDCT` for throughput. Expose quality as a runtime parameter.

2. **FFT circular low-pass via FFTW r2c** (cutoff radius as fraction of min(H,W)/2): Good for preprocessing before VLMs where texture statistics matter less. Use FFTW wisdom for plan caching across images of the same size.

3. **Block DCT coefficient zeroing via OpenCV cv::dct()**: Orthogonal to JPEG (different quantization pattern). Useful for targeted high-frequency removal without full JPEG pipeline.

4. **Wavelet soft-thresholding (db1/Haar, BayesShrink)**: Implement via a DWT library (libwavelet, or a custom 2-level Haar DWT in ~50 lines). Apply per-channel in YCbCr. Use `T = sigma^2 / sigma_x` per sub-band.

5. **Pixel Deflection + Wavelet**: Combine the stochastic spatial step with wavelet denoising. The spatial step has a trivial implementation (Algorithm 1 above); the wavelet denoising is step 4.

The first two are pure frequency-domain; steps 3–5 are hybrid. For pixmask's C++ pipeline, FFTW + libjpeg-turbo cover the full frequency-domain surface with production-grade SIMD performance.

---

## 8. References

- Prakash, A., Moran, N., Garber, S., DiLillo, A., Storer, J. "Deflecting Adversarial Attacks with Pixel Deflection." CVPR 2018. arXiv:1801.08926. Code: github.com/iamaaditya/pixel-deflection
- Guo, C., Rana, M., Cisse, M., van der Maaten, L. "Countering Adversarial Images using Input Transformations." ICLR 2018. arXiv:1711.00117. Code: github.com/facebookresearch/adversarial_image_defenses
- Wang, H., Wu, X., Huang, Z., Xing, E. "High-Frequency Component Helps Explain the Generalization of Convolutional Neural Networks." CVPR 2020. Code: github.com/HaohanWang/HFC
- Gu, S., Rigazio, L. "Towards Deep Neural Network Architectures Robust to Adversarial Examples." arXiv:1412.5068, 2014. (Denoising autoencoder defense baseline)
- Das, N., Shanbhogue, M., Chen, S., Hohman, F., Chen, L., Chau, D.H., Kounavis, M. "SHIELD: Fast, Practical Defense and Vaccination for Deep Learning." KDD 2018.
- Chang, S.G., Yu, B., Vetterli, M. "Adaptive Wavelet Thresholding for Image Denoising and Compression." IEEE Trans. Image Processing, 2000. (BayesShrink algorithm)
- Frigo, M., Johnson, S.G. "The Design and Implementation of FFTW3." Proc. IEEE 2005. FFTW 3.3.10 documentation: fftw.org/fftw3_doc
- libjpeg-turbo: libjpeg-turbo.org (2–7x SIMD-accelerated JPEG for JPEG defense implementation)
