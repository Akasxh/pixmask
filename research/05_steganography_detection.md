# Steganographic Attacks and Detection for Multimodal LLM Security

> Research for pixmask — C++ image sanitization layer before multimodal LLMs.
> Covers the full steganography threat model: embedding methods, VLM attack surface,
> steganalysis techniques, and minimum-cost neutralization pipelines.

---

## 1. Steganographic Embedding Methods

### 1.1 LSB (Least Significant Bit) Steganography

**Mechanism:** Each pixel byte carries payload in its least significant bit(s). For a 24-bit RGB image, replacing the lowest 2 bits per channel yields 6 payload bits per pixel with a maximum channel-value deviation of ±3 (out of 255). The change is visually imperceptible.

**Sequential vs. scattered embedding:**
- Sequential LSB replaces bits in scan order — trivially detectable.
- Random-walk LSB uses a PRNG seeded with a key to select carrier pixels, providing security through obscurity but not statistical invisibility.

**Capacity:** 1–3 bpp (bits per pixel) typical. 3 bpp hides ~375 KB per megapixel.

**Detectability:** Highly detectable by chi-square, RS, and Sample Pairs analysis (see §3). This is a 1990s-era technique.

**Payload for LLM attack:** ~375 KB/MP is more than enough to encode a full system-prompt override, jailbreak chain, or instruction set.

---

### 1.2 DCT-Domain Steganography (JPEG)

JPEG stores image data as quantized 8×8 block DCT coefficients. Steganography operates on these coefficients after quantization, before entropy coding.

#### JSteg (Derek Upham, 1997)
- Sequentially replaces LSBs of non-zero, non-one DCT coefficients.
- Creates easily detectable Pairs of Values (PoV) anomalies in the DCT histogram — chi-square attack breaks it trivially.

#### OutGuess (Niels Provos, 2001)
- Random-walk embedding into DCT coefficient LSBs, skipping 0s and 1s.
- **Key innovation:** After embedding, applies a correction pass to non-embedded coefficients to restore the global DCT histogram to match the cover image.
- Defeats first-order statistical attacks (chi-square). Vulnerable to second-order calibration attacks.

#### F5 (Andreas Westfeld, 2001)
- Embeds by *decrementing* the absolute value of non-zero DCT coefficients (never increments), preserving histogram shape directionally.
- Uses matrix embedding (Hamming codes) to minimize the number of modified coefficients for a given payload.
- Broken by the calibration attack (Fridrich et al., 2002): estimate the unmodified DCT statistics by re-compressing the decompressed image and comparing histograms.

**Common vulnerability:** All three algorithms embed in the same domain that JPEG quantization operates in. Re-quantization at a different quality factor corrupts the payload.

---

### 1.3 Spread-Spectrum Steganography

**Mechanism:** A narrow-band payload signal is modulated onto a wideband pseudorandom carrier, then added to the host image at low amplitude. The carrier is generated from a secret key; only the key-holder can correlate and recover the message.

**Domain:** Can operate in spatial domain (add low-amplitude PN sequence to pixel values) or frequency domain (modulate DCT/DFT coefficients).

**Key properties:**
- Signal energy is below the perceptual noise floor — statistically near-invisible.
- Robust: random noise addition, mild filtering, and JPEG compression at moderate quality factors may not destroy it because the receiver correlates against the full carrier.
- Low capacity: typically 0.01–0.1 bpp, used for watermarking more than high-payload steganography.

**Relevance to LLM attack:** Low capacity limits payload size. An attacker would need the image to survive transport and preprocessing intact. Robustness to mild JPEG recompression is the key concern.

---

### 1.4 Deep Learning-Based Steganography

#### HiDDeN (Zhu et al., 2018)
- End-to-end encoder-decoder architecture. The encoder network takes (cover image, message bits) and produces a stego image. The decoder recovers the message.
- A jointly trained noise layer simulates attacks (JPEG, noise, cropping) during training, making the learned encoding robust to those transforms.
- First demonstration of learned steganography with robustness constraints.

#### SteganoGAN (Zhang et al., 2019 — arXiv:1901.03892)
- GAN-based framework: encoder (generator) hides data, discriminator enforces imperceptibility, decoder recovers message.
- Capacity: **4.4 bpp** — state-of-the-art at time of publication, far exceeding classical methods.
- Adversarial training against a steganalyzer discriminator means it actively evades CNN-based detection tools.
- Vulnerable to JPEG compression without explicit robustness training.

#### Robust Neural Steganography (2023–2025)
- Networks trained with differentiable JPEG approximation layers or explicit robustness classifiers now achieve >98% bit-accuracy after JPEG recompression at QF 75+.
- DCT Residual Modulation (DRM, 2024, Signal Processing) achieves robustness by modeling which DCT coefficients survive re-quantization at the target quality factor.
- Key implication: **JPEG recompression alone is no longer a reliable neutralizer against purpose-built robust neural steganography.**

---

## 2. How Steganography Could Attack Multimodal LLMs

### 2.1 The Attack Model

A steganographic LLM attack couples two elements:
1. A visible, benign image (e.g., a photo of a cat).
2. Hidden payload: text instructions embedded in the image via steganography.

The user submits the image with a benign prompt like "Describe this image." The VLM, during visual token extraction, may recover and interpret the hidden instructions — because modern vision transformers process pixel-level information at sufficient resolution to "see" LSB patterns.

### 2.2 Demonstrated Attack Results

**Hidden Image Jailbreak (2024):**
- Uses LSB steganography to embed instructions in images.
- Achieves >90% success rate against GPT-4o and Gemini-1.5 Pro with an average of 3 queries.
- Exploits cross-modal reasoning: the model is prompted with an image description task but the hidden text overrides safety behavior.

**Invisible Injections (arXiv:2507.22304):**
- First systematic study of steganographic prompt injection across spatial, frequency, and neural steganography methods.
- Multi-domain attack success rate: 24.3% ± 3.2% (95% CI) across GPT-4V, Claude, and LLaVA.
- Neural steganography achieves up to 31.8% success rate.
- Attack imperceptibility: PSNR > 38 dB, SSIM > 0.94 — visually indistinguishable from clean images.
- Attack categories: information extraction (29.4%), behavioral modification (24.1%), content manipulation (22.8%), safety bypass (18.7%).

**FigStep / FigStep-Pro (AAAI 2025 Oral):**
- Hides harmful instructions as rendered text *in the visual content* of the image (typographic injection).
- FigStep-Pro bypasses OpenAI's OCR detector to jailbreak GPT-4V.

**Oncology VLM Attack (Nature Communications, 2025):**
- Malicious instructions embedded in medical images cause VLMs to produce harmful diagnostic outputs.
- 594 attack samples demonstrated against clinical-grade VLMs.

### 2.3 Why VLMs Are Vulnerable

Vision transformers (ViT) operate on image patches of typically 14×14 or 16×16 pixels. Each patch is projected to a token embedding. The LSBs of pixels within a patch contribute directly to the patch embedding vector. If the hidden payload is sufficiently dense, its signal leaks into the token embedding space.

Additionally, VLMs are trained to be attentive to fine-grained image details (OCR, text-in-image tasks). This same capability makes them susceptible to interpreting steganographic content as meaningful signals.

**The mechanism is not reliably reproducible** — success depends on the model's training distribution and attention patterns. But the demonstrated 90%+ success rates for LSB attacks mean this is not a theoretical concern.

### 2.4 Attack Surface Summary

| Method | Payload Capacity | VLM Exploitability | Survives Mild JPEG? |
|---|---|---|---|
| Spatial LSB | High (3+ bpp) | High (>90% in demos) | No |
| DCT LSB (JSteg/F5) | Medium (0.5–2 bpp) | Medium | Partially |
| Spread spectrum | Low (0.01–0.1 bpp) | Low (insufficient for instructions) | Yes |
| Neural (SteganoGAN) | Very high (4.4 bpp) | High | Without robust training: No |
| Robust neural (DRM 2024) | Medium (0.5–2 bpp) | Medium–High | Yes (by design) |
| EXIF/metadata injection | Unbounded | N/A (text, not pixels) | Yes |

---

## 3. Steganalysis — Detection Methods

### 3.1 Statistical Tests (Classical)

#### Chi-Square Attack (Westfeld & Pfitzmann, 2000)
- **Principle:** LSB embedding with sequential bit assignment changes the frequency of PoVs (pairs of values — pixel values that differ only in their LSB). In a natural image, PoV frequencies are unequal; after LSB embedding, they converge.
- **Statistic:** χ² test against the expected PoV distribution.
- **Effective against:** Sequential LSB, JSteg.
- **Bypassed by:** OutGuess (histogram correction), F5, random-walk LSB.

#### RS Analysis (Fridrich et al., 2001)
- **Principle:** Partition image into small groups (e.g., 2×2 blocks). Apply a discriminant function to each group. Count "regular" groups (function increases after flip) vs. "singular" groups (function decreases). Plots the R and S curves as a function of estimated embedding rate.
- **Effective against:** LSB replacement in grayscale and 24-bit color.
- **Detects:** Even random-walk LSB embedding.
- **Limitations:** Requires a significant embedded fraction; low-rate embedding is harder to detect.

#### Sample Pairs Analysis (Dumitrescu et al., 2003)
- **Principle:** Models the steganalysis problem as a finite-state machine operating on consecutive pixel pairs. Exploits higher-order sample correlations to estimate the embedding rate.
- **More sensitive than RS analysis** at low embedding rates.
- **StegExpose** combines chi-square + RS + Sample Pairs + Primary Sets in a weighted fusion for improved accuracy on unknown images.

#### Calibration Attack (Fridrich et al., 2002 — breaks F5)
- Decompresses the JPEG, re-compresses at the same quality factor, and computes the DCT histogram of the re-compressed image as an estimate of the unmodified cover statistics. The delta between original and calibrated histograms reveals embedding artifacts.

### 3.2 Deep Learning Steganalysis

#### YeNet (Ye et al., 2017)
- **Architecture:** CNN with 10 convolutional layers. First layer initializes with 30 high-pass kernels from the SRM (Spatial Rich Model) feature set — these are hand-crafted residual filters designed to suppress image content and amplify embedding artifacts.
- **Activation:** Thresholded Linear Unit (TLU) in the first layer to suppress large activations from image content; ReLU elsewhere.
- **Domain:** Spatial domain images.
- **Variant:** J-YeNet extends to JPEG domain.

#### SRNet (Boroumand et al., 2019 — Binghamton)
- **Architecture:** 12-layer deep residual network. First 7 layers form a "noise residual extraction" segment using no pooling operations — preserving the low-energy steganographic signal that pooling would destroy.
- **Key difference from YeNet:** Uses 64 learnable kernels in Layer 1 rather than pre-defined SRM filters.
- **Performance:** State-of-the-art for spatial-domain steganalysis. Detects WOW, HUGO, S-UNIWARD, and similar content-adaptive schemes.
- **Variant:** For JPEG domain, J-SRNet substitutes DCT-domain feature extraction.

#### Limitations of Deep Steganalysis
- Models trained on one steganographic algorithm often fail on unseen algorithms.
- SteganoGAN was explicitly trained to fool CNN discriminators — it partially evades YeNet/SRNet.
- Deep steganalysis requires GPU inference; integrating it inline into a preprocessing pipeline adds latency.

---

## 4. Neutralization — Destroying Steganographic Payloads

### 4.1 Bit-Depth Reduction

**Effect on LSB steganography:** Reducing channel depth from 8 bits to 5 bits (or fewer) overwrites the bottom 3 bits entirely. All LSB steganography is destroyed. No payload survives.

**Implementation:** `pixel = (pixel >> 3) << 3` per channel, or quantize to a palette of 2^k colors.

**Effect on DCT steganography:** Indirect — bit-depth reduction in the spatial domain before JPEG encoding changes the DCT coefficients computed during encoding, but does not directly address DCT-domain payloads already in a JPEG file. Re-encoding at reduced quality is more effective for JPEG stego.

**Image quality impact:** At 5 bits/channel (32 levels per channel), contouring artifacts appear in smooth gradients. At 6 bits/channel (64 levels), the impact is minimal for natural images.

### 4.2 JPEG Recompression

**Effect on spatial-domain steganography:** Converting a PNG/BMP/TIFF with LSB payload to JPEG at any lossy quality factor destroys the LSB content. DCT quantization of the 8×8 blocks overwrites fine pixel-level differences.

**Effect on DCT-domain steganography (JSteg, OutGuess, F5):** Re-compressing at a *different quality factor* re-quantizes the DCT coefficients, destroying embedded LSB bits. The payload is lost whenever the quantization step for a coefficient changes.

**Rule:** JPEG → JPEG at a different quality factor OR PNG → JPEG at any quality ≤ 95 destroys classical DCT stego. Quality factor 75 is the standard recommendation.

**Exception — robust neural steganography:** Methods like DRM (2024) explicitly model the re-quantization process during training and select embedding locations where DCT coefficients survive re-quantization. Re-compressing at the *same* quality factor these were designed for may not destroy the payload. Solution: use a random quality factor or multiple recompressions.

### 4.3 Gaussian Noise Addition

**Effect on LSB:** Adding Gaussian noise with σ ≥ 1.0 destroys LSB payload because the noise magnitude exceeds the 1-bit embedding magnitude. With σ ≥ 4.0, even 2–3 bit LSB embedding is reliably destroyed.

**Effect on spread-spectrum:** Spread-spectrum is designed to resist additive noise (the receiver correlates against the full carrier). Low-amplitude Gaussian noise may not destroy a spread-spectrum payload. High-amplitude noise (σ ≥ 8–10) degrades image quality significantly.

**Effect on neural stego:** Neural networks trained with a noise augmentation layer (HiDDeN-style) are robust to moderate Gaussian noise. Severe noise (σ > 15) degrades the payload but also visibly degrades the image.

**Conclusion:** Gaussian noise alone is insufficient as a universal sanitizer.

### 4.4 Median Filtering

**Effect:** Replaces each pixel with the median of its neighborhood. Destroys isolated outlier pixels characteristic of spatial-domain steganography. Less effective against adaptive embedding (WOW, HUGO) that embeds near texture edges where pixel values are already variable.

**Kernel size 3×3:** Sufficient to destroy most sequential LSB steganography.

**Combined with JPEG:** Median filter → JPEG recompression is more robust than either alone.

### 4.5 Re-Encoding Pipeline (Practical Universal Sanitizer)

The minimum effective pipeline that destroys all common steganographic methods:

```
Input image (any format)
     │
     ▼
[1] Decode to raw pixels
     │
     ▼
[2] Strip all metadata (EXIF, XMP, ICC, comments)         ← destroys metadata injection
     │
     ▼
[3] Bit-depth reduction: 8-bit → 6-bit per channel        ← destroys spatial LSB (1–2 bit)
    (pixel = (pixel >> 2) << 2)
     │
     ▼
[4] Gaussian blur σ=0.5 (mild, preserves sharpness)       ← softens remaining high-freq artifacts
     │
     ▼
[5] JPEG encode at random QF ∈ [70, 85]                   ← destroys DCT stego, spatial stego
     │
     ▼
[6] JPEG decode back to pixels                             ← now in spatial domain, clean
     │
     ▼
[7] Re-encode to target format (PNG or WebP lossless)
```

**What this destroys:**

| Attack Type | Destroyed? | Mechanism |
|---|---|---|
| Spatial LSB (1–3 bit) | Yes | Step 3 overwrites bottom 2 bits; step 5 overwrites all fine structure |
| JSteg / OutGuess / F5 | Yes | Step 5: re-quantization at new QF destroys DCT LSBs |
| Spread spectrum (low amplitude) | Mostly | Steps 3+5 add quantization noise exceeding signal amplitude |
| SteganoGAN (non-robust) | Yes | JPEG recompression is the standard failure mode |
| Robust neural stego (DRM) | Partially | Random QF disrupts QF-specific robustness training |
| EXIF/metadata injection | Yes | Step 2 strips all metadata |
| Typographic injection (FigStep) | No | Pixel-level information is preserved; this is visual content |

**Note on typographic injection:** FigStep-style attacks embed instructions as *rendered visible text in the image*. No sanitization of pixel values will defeat this — it requires OCR-based content filtering, not pixel-level sanitization. This is outside pixmask's scope.

### 4.6 Advanced Neutralization: Neural Purification

For adversaries using robust neural steganography (the hardest case), pixel-level preprocessing alone may be insufficient. Two complementary approaches exist:

**Deep autoencoder sanitization (Springer 2024):**
- U-Net-like encoder-decoder reconstructs the image through a semantic bottleneck. The bottleneck discards fine-grained pixel-level encoding that carries steganographic payload.
- Effective against all known steganography types but adds ~10–50ms GPU latency per image.

**SS-Net (PMC 2024 — Self-Supervised CNN Sanitizer):**
- Does not require knowledge of the steganographic scheme.
- Purification module + refinement module. Pixel-shuffle downsampling in the purification module breaks spatial correlations exploited by adaptive steganography.
- Reported BER > 0.52 on sanitized stego images (random bit error rate = payload is destroyed), PSNR > 44 dB (image quality preserved).

**For pixmask:** The C++ pipeline should implement steps 1–7 above. Neural purification can be exposed as an optional high-security mode with Python binding overhead.

---

## 5. Minimum Preprocessing to Destroy ALL Common Steganographic Methods

### 5.1 Threat-Ranked Requirements

| Threat | Minimum Countermeasure |
|---|---|
| Spatial LSB (1 bit) | Bit-depth reduction: 8→7 bits |
| Spatial LSB (2–3 bit) | Bit-depth reduction: 8→5 bits OR JPEG QF≤85 |
| JSteg | JPEG → JPEG at different QF |
| OutGuess | JPEG → JPEG at different QF (histogram correction is bypassed by re-quantization) |
| F5 | JPEG → JPEG at different QF |
| SteganoGAN (standard) | JPEG QF < 90 |
| HiDDeN (robustness-trained) | JPEG QF < 75 + light noise |
| Robust neural (DRM 2024) | Random QF ∈ [70, 85] + bit-depth reduction |
| Spread spectrum | Multiple passes: bit-depth reduction + JPEG + mild Gaussian noise |
| Metadata injection | Full metadata strip before pixel decoding |

### 5.2 Single-Pass Minimum Viable Pipeline

For a latency-constrained deployment (no GPU, C++ only):

```
strip_metadata()
  → reduce_bit_depth(bits=6)
  → jpeg_encode(quality=random(70, 85))
  → jpeg_decode()
```

This four-step pipeline, implemented in C++ using libjpeg-turbo, adds:
- ~1–3ms per image at 1080p on a modern CPU
- Destroys all classical steganography (LSB, DCT-domain)
- Destroys non-robust neural steganography (SteganoGAN without robustness training)
- Partially disrupts robust neural steganography (randomized QF attacks the QF-specific training assumption)

### 5.3 Defense-in-Depth Pipeline (Recommended for pixmask)

```
strip_metadata()
  → reduce_bit_depth(bits=6)
  → gaussian_blur(sigma=0.5)
  → jpeg_encode(quality=random(70, 85))
  → jpeg_decode()
  → random_resize(scale=random(0.9, 1.1)) + restore_to_original_size
  → png_encode()
```

The random resize step:
- Non-differentiable, frustrating gradient-based adaptive attacks.
- Destroys any spatial embedding that depends on exact pixel position (most steganography).
- Adds ~0.5ms on CPU.

This pipeline is consistent with pixmask's existing Feature Squeezing + Lossy Compression defense families (per README).

---

## 6. Integration Notes for pixmask

### Steganography vs. Adversarial Perturbations

Steganographic attacks and standard adversarial examples (FGSM, PGP, C&W) differ in their structure:

| Property | Adversarial Perturbation | Steganography |
|---|---|---|
| Objective | Fool classifier output | Inject instructions into LLM reasoning |
| Location | Distributed across gradient-sensitive features | Concentrated in LSBs / DCT LSBs / learned carrier |
| Detection | Statistical: detection requires knowing the model | Statistical: detectable without the model |
| Magnitude | ℓ∞ ≤ ε (8/255 typical) | Imperceptible (<1/255 per pixel for 1-bit LSB) |
| Destroyed by | Feature squeezing, smoothing, compression | Bit-depth reduction, compression |

The good news: **pixmask's existing pipeline already neutralizes most steganographic attacks** because bit-depth reduction (Feature Squeezing) and JPEG recompression are already present. The incremental additions needed are:

1. Explicit metadata stripping (not currently called out in the README).
2. Randomized JPEG quality factor (prevents optimization against a fixed QF).
3. Random resize before final encode (disrupts position-dependent embedding).

### Steganalysis as a Detection Layer

Rather than (or in addition to) neutralization, pixmask could expose a steganalysis detection path:

- **Cheap:** Chi-square test on LSB plane — ~0.1ms, catches sequential LSB.
- **Medium:** RS analysis — ~2ms, catches random-walk LSB.
- **Expensive:** SRNet inference — ~5–20ms GPU, catches adaptive and neural steganography.

Detection can trigger an audit log / rejection path separate from the sanitization pipeline.

---

## 7. Key References

- Westfeld & Pfitzmann (2000). *Attacks on Steganographic Systems.* (Chi-square attack)
- Fridrich et al. (2001). *Reliable Detection of LSB Steganography in Color and Grayscale Images.* (RS analysis)
- Dumitrescu et al. (2003). *Detection of LSB Steganography via Sample Pair Analysis.*
- Provos (2001). *Defending Against Statistical Steganalysis.* (OutGuess)
- Westfeld (2001). *F5—A Steganographic Algorithm.*
- Fridrich et al. (2002). *Steganalysis of JPEG Images: Breaking the F5 Algorithm.*
- Zhu et al. (2018). *HiDDeN: Hiding Data With Deep Networks.*
- Zhang et al. (2019). *SteganoGAN: High Capacity Image Steganography with GANs.* arXiv:1901.03892
- Ye et al. (2017). *Deep Learning Hierarchical Representations for Image Steganalysis.* (YeNet)
- Boroumand et al. (2019). *Deep Residual Network for Steganalysis of Digital Images.* (SRNet) — ws.binghamton.edu
- Anon (2024). *Invisible Injections: Exploiting VLMs Through Steganographic Prompt Embedding.* arXiv:2507.22304
- Clusmann et al. (2025). *Prompt Injection Attacks on VLMs in Oncology.* Nature Communications.
- Hidden Image Jailbreak. promptfoo.dev/lm-security-db/vuln/hidden-image-jailbreak-37b7539b
- Anon (2024). *Robust Image Steganography Against JPEG Compression Based on DCT Residual Modulation.* Signal Processing.
- Anon (2024). *Erasing the Shadow: Sanitization of Images with Malicious Payloads Using Deep Autoencoders.* Springer.
