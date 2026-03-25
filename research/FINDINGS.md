# pixmask Research Findings — Consolidated Synthesis

> 15 research agents, ~400KB of primary research. This document synthesizes the actionable findings.

---

## 1. Threat Model

pixmask is a **preprocessing security layer** that sits between untrusted image input and a multimodal LLM (GPT-4V/4o, Claude Vision, Gemini, LLaVA, etc.). It must neutralize attacks while preserving image utility for the VLM.

### Attack Classes (ordered by real-world danger)

| # | Attack Class | ASR (undefended) | Key Papers | Preprocessing Effective? |
|---|---|---|---|---|
| 1 | **Typographic injection** (FigStep) | 82.5% on open VLMs | Gong et al. AAAI 2025 | NO — requires OCR detection |
| 2 | **Gradient perturbations** (PGD, C&W) | >95% on GPT-4o (2025) | Qi et al. AAAI 2024, Carlini NeurIPS 2023 | YES — bit-depth + smoothing + JPEG |
| 3 | **Steganographic injection** (LSB, neural) | 24-31.8% cross-model | arXiv:2507.22304 | YES — bit-depth + JPEG destroys LSB/DCT |
| 4 | **Scaling attacks** | model-dependent | Quiring et al. USENIX 2020 | YES — INTER_AREA + jitter |
| 5 | **Malformed images** (parser exploits) | N/A (code exec) | CVE-2023-4863 (libwebp) | YES — safe parser + re-encode |
| 6 | **Composite attacks** (HADES) | 90.26% on LLaVA | Li et al. ECCV 2024 | PARTIAL — layered defense needed |

### Key Insight
**No single preprocessing defense survives fully adaptive white-box attacks** (proven by Athalye et al. ICML 2018). But the realistic production threat is **non-adaptive**: attackers craft images offline without knowledge of the specific preprocessing pipeline. For this threat model, layered preprocessing reduces ASR to 3-6% (OmniSafeBench-MM).

---

## 2. Defense Pipeline Design

### Recommended Default Pipeline

```
Input Image
  ├── Stage 0: VALIDATE (magic bytes, dimensions, file size, decompression ratio)
  ├── Stage 1: DECODE (libspng/libjpeg-turbo/libwebp — safe parsers only)
  ├── Stage 2: STRIP METADATA (EXIF, XMP, ICC profiles — zero-cost, removes hidden data)
  ├── Stage 3: SANITIZE PIXELS
  │   ├── 3a: Bit-depth reduction (8→5 bits) — collapses adversarial increments
  │   ├── 3b: Spatial smoothing (bilateral σ_s=5, σ_r=15) — edge-preserving denoise
  │   ├── 3c: Pixel deflection (K=100, r=10) — stochastic, non-differentiable
  │   └── 3d: Wavelet denoise (Haar + BayesShrink σ=0.04) — strongest standalone
  ├── Stage 4: FREQUENCY FILTER
  │   └── 4a: JPEG encode/decode (QF = random(70,85)) — destroys DCT-domain stego
  ├── Stage 5: SAFE RESIZE (INTER_AREA + random jitter ±5%)
  └── Stage 6: RE-ENCODE (PNG or JPEG, clean output)
```

### Preset Profiles

| Profile | Stages | Latency (1080p est.) | SSIM | Use Case |
|---|---|---|---|---|
| `fast` | 0→1→2→3a→4a→6 | ~3ms | ~0.92 | High-throughput API serving |
| `balanced` | 0→1→2→3a→3b→4a→5→6 | ~8ms | ~0.88 | Default for most deployments |
| `paranoid` | All stages | ~25ms | ~0.82 | High-security, adversarial-aware |

### What Cannot Be Stopped by Pixel Preprocessing
- **Typographic attacks** (FigStep): Require OCR + text safety classification (out of scope for v1, mark as future)
- **Semantic content attacks**: Image itself IS harmful content (requires content moderation, not sanitization)
- **Fully adaptive white-box**: Provably unbeatable by preprocessing alone

---

## 3. Implementation Architecture

### C++ Core Design

- **SIMD**: Google Highway for portable SSE2/AVX2/NEON dispatch
- **Median filter**: Sorting network for 3×3 (58× speedup), CTMF for ≥5×5
- **Gaussian blur**: 3-pass box blur (O(n) any σ), Van Vliet IIR for σ > 30
- **Bilateral**: Naive + range LUT for small kernels, bilateral grid for large
- **Bit-depth**: `(x >> (8-b)) << (8-b)` — trivial but needs 16-bit SIMD workaround on AVX2
- **DCT/JPEG**: AAN algorithm for 8×8 DCT, or link libjpeg-turbo
- **Wavelet**: Haar DWT (trivial: average + difference), BayesShrink thresholding
- **TV denoise**: Chambolle-Pock (no FFT dependency)
- **Memory**: Arena allocator, pre-allocated scratch buffers, zero-allocation hot path
- **Cache**: Tile to L1 (64×128 blocks), 64-byte aligned rows

### Safe Decoder Stack
- **PNG**: libspng (CERT C compliant, fuzzed, no known CVEs)
- **JPEG**: libjpeg-turbo ≥ 3.0.4
- **WebP**: libwebp ≥ 1.3.2
- **REJECT**: GIF (stb_image has open exploits), TIFF (massive attack surface), SVG (XXE/script injection)

### Python Bindings
- **nanobind** (not pybind11): 3-5× smaller binary, 2.7-4.4× faster compile, stable ABI for 3.12+
- **scikit-build-core** + CMake
- **cibuildwheel** for manylinux x86_64/aarch64, macOS arm64/x86_64, Windows x64
- Zero runtime Python deps (numpy optional peer dep)

---

## 4. Evaluation Plan

### Metrics
- **Security**: ASR before/after (PGD-40, C&W, FigStep, steganographic injection)
- **Quality**: SSIM ≥ 0.85, LPIPS ≤ 0.15, PSNR ≥ 30 dB
- **Performance**: images/sec at 224px, 512px, 1024px, 2048px; p50/p95/p99 latency
- **Adaptive**: BPDA + EOT with DiffJPEG surrogate

### Benchmarks
- Google Benchmark for C++ microbenchmarks
- JailBreakV-28K for VLM jailbreak ASR
- MM-SafetyBench for safety scenarios
- ImageNet-1K subset for clean accuracy preservation

### Baselines
- No defense
- JPEG-only (Q=75)
- Gaussian blur only (σ=1.0)
- ART FeatureSqueezing (Python)
- OpenCV equivalent pipeline

---

## 5. Competitive Position

| Competitor | Fatal Limitation | pixmask Advantage |
|---|---|---|
| ART (IBM) | Pure Python, 50-500ms/image, no VLM support | C++ core, <10ms, VLM-first |
| pycocotools | Cython build hell, mask ops only | nanobind, no Cython, zero build issues |
| Foolbox | Attack-only, no defenses | Defense-first |
| DiffPure | 30-1000× inference cost, no pip package | <10ms, pip install |
| OpenCV | 50MB, libGL Docker issues | <5MB, zero system deps |
| Nothing | No VLM image sanitization tool exists on PyPI | First mover |

---

## 6. Key References

- Xu, Evans, Qi — "Feature Squeezing", NDSS 2018
- Guo et al. — "Countering Adversarial Images via Input Transformations", ICLR 2018
- Athalye, Carlini, Wagner — "Obfuscated Gradients Give a False Sense of Security", ICML 2018
- Prakash et al. — "Deflecting Adversarial Attacks with Pixel Deflection", CVPR 2018
- Das et al. — "SHIELD", KDD 2018
- Qi et al. — "Visual Adversarial Examples Jailbreak Aligned LLMs", AAAI 2024
- Gong et al. — "FigStep", AAAI 2025
- Quiring et al. — "Image-Scaling Attacks", USENIX Security 2020
- Cohen et al. — "Certified Adversarial Robustness via Randomized Smoothing", 2019
- arXiv:2507.22304 — "Invisible Injections: Steganographic Prompt Embedding", 2025
- Wang et al. — "High-Frequency Component Helps Explain Generalization", CVPR 2020
- Zhang et al. — "Adversarial Illusions in Multi-Modal Embeddings", USENIX Security 2024

---

*Full research in individual files: `research/01_*.md` through `research/15_*.md`*
