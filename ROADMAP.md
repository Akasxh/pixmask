# pixmask Roadmap

> Living document tracking what's shipped, what's next, and what's planned.
> Last updated: 2026-03-25 after v0.1.0 PyPI release.

---

## v0.1.0 — SHIPPED (2026-03-25)

### What's in
| Component | Status | Details |
|---|---|---|
| Input validation (Stage 0) | Done | Magic bytes, dimension pre-check, decomp ratio, file size |
| Safe decode (Stage 1) | Done | stb_image, JPEG+PNG only, GIF/BMP/TGA compile-disabled |
| Metadata strip (Stage 2) | Implicit | Re-encode strips metadata by design |
| Bit-depth reduction (Stage 3) | Done | Highway SIMD (SSE2/AVX2/NEON), configurable 1-8 bits |
| Median filter 3x3 (Stage 4) | Done | 19-step Bose-Nelson sorting network, Highway SIMD |
| JPEG roundtrip (Stage 5) | Done | Random QF via OS entropy (getrandom/getentropy/BCrypt) |
| Pipeline orchestrator | Done | Arena-backed, zero-alloc hot path |
| Python API | Done | Progressive disclosure, accepts ndarray/bytes/Path/PIL |
| nanobind bindings | Done | Zero-copy output, GIL release during C++ |
| Build system | Done | scikit-build-core + cibuildwheel |
| CI | Done | 3-OS matrix, ASan/UBSan, wheel publishing |
| PyPI | Done | `pip install pixmask` works on Linux/macOS/Windows |

### Known limitations in v0.1.0
- stb_image decoder (no libspng/libjpeg-turbo — slower, fewer security guarantees)
- No WebP decode support (validate accepts it, but decode rejects it)
- Median filter RGB path is scalar (SIMD only for grayscale)
- No `paranoid` preset (warns that it's not available yet)
- No aarch64 Linux wheel (dropped QEMU, too slow — needs native ARM runner)
- Direct numpy array sanitization goes through PNG encode→C++ decode roundtrip
- No steganography detection signal (only destruction)
- No typographic/OCR attack defense

---

## v0.2.0 — Next Release

### Priority 1: Security upgrades

| Task | Why | Effort | Research ref |
|---|---|---|---|
| **Replace stb_image with libspng + libjpeg-turbo** | stb has open GIF CVEs (disabled but risky), libspng is CERT-C compliant, libjpeg-turbo has SIMD decode (5-8x faster) | Medium | `research/06_malformed_image_security.md` |
| **WebP decode via libwebp >= 1.3.2** | Currently accepted at validation but rejected at decode | Small | `research/06_malformed_image_security.md` |
| **Vendor with SHA-256 verification** | CMake FetchContent with VERIFY_HASH for all vendored libs | Small | `architecture/DEBATE.md` (Security Architect) |

### Priority 2: Defense upgrades

| Task | Why | Effort | Research ref |
|---|---|---|---|
| **Bilateral filter** | Pareto-superior to median for VLM quality (preserves edges, SSIM ~0.92 vs ~0.88) | Medium | `research/13_quality_preservation.md`, `research/14_spatial_smoothing_algorithms.md` |
| **Gaussian blur (3-pass box)** | O(n) any sigma, complements bilateral | Small | `research/14_spatial_smoothing_algorithms.md` |
| **Haar wavelet denoise** | Strongest standalone defense (98% on C&W). Inline DWT, no FFTW. BayesShrink σ=0.04 | Medium | `research/04_frequency_domain_defenses.md` |
| **Pixel deflection** | Stochastic + non-differentiable = adaptive attack resistant. K=100, r=10 | Small | `research/03_feature_squeezing_defenses.md` |
| **Safe resize (INTER_AREA + jitter)** | Defeats scaling attacks (Quiring 2020). Random ±5% jitter | Small | `research/07_scaling_attacks.md` |

### Priority 3: Detection signals

| Task | Why | Effort | Research ref |
|---|---|---|---|
| **Stego chi-square detection** | Expose `stego_suspicion_score` in SanitizeResult. ~0.1ms, zero false negatives on LSB | Small | `research/05_steganography_detection.md` |
| **Feature squeezing detection** | Compare SSIM(raw, sanitized). Large delta = likely adversarial. Zero extra cost | Small | `research/03_feature_squeezing_defenses.md` |

### Priority 4: Performance + platform

| Task | Why | Effort | Research ref |
|---|---|---|---|
| **SIMD median for RGB** | Currently scalar per-channel. Deinterleave → SIMD per channel → interleave | Medium | `research/09_cpp_simd_optimization.md` |
| **Direct pixel-buffer entry point** | Skip decode for numpy array input (avoid PNG encode roundtrip) | Small | `architecture/DECISIONS.md` |
| **aarch64 Linux wheels** | Use native ARM runner (GitHub now offers `ubuntu-24.04-arm`) instead of QEMU | Small | — |
| **Google Benchmark suite** | p50/p95/p99 at 224/512/1024/2048. Regression gate in CI | Small | `research/11_evaluation_methodology.md` |

### Priority 5: Python API enhancements

| Task | Why | Effort | Research ref |
|---|---|---|---|
| **Pipeline builder class** | `pixmask.Pipeline([BitDepthReduce(5), BilateralSmooth(5,15), ...])` | Medium | `architecture/DEBATE.md` (API/UX Designer) |
| **`paranoid` preset** | Enable once bilateral + wavelet + pixel deflection are implemented | Small | `architecture/DECISIONS.md` |
| **`SanitizeResult` with metadata** | detection scores, timing, warnings | Small | `architecture/DEBATE.md` |
| **async_sanitize** | `run_in_executor` wrapper for FastAPI/async frameworks | Small | `architecture/DEBATE.md` (API/UX Designer) |

---

## v0.3.0 — Future

### OCR / Typographic attack defense
- FigStep is the #1 real-world attack (82.5% ASR, works on closed-source VLMs)
- Pixel preprocessing cannot stop it — needs OCR extraction + text safety classification
- Options: Tesseract C++ API, PaddleOCR, or lightweight CNN text detector
- Architecturally separate: `TextDetectionCallback` hook, not inline in pipeline
- Research: `research/01_visual_prompt_injection.md`, `research/08_multimodal_jailbreaks.md`

### Total variation denoising
- Chambolle-Pock algorithm (no FFT dep, SIMD-friendly)
- ~20ms at 1080p — only for `paranoid` preset
- Research: `research/14_spatial_smoothing_algorithms.md`

### Content-aware adaptive sanitization
- Apply stronger params to "suspicious" regions (high-frequency anomaly detection)
- Apply lighter params to clean regions (preserve quality)
- Research: `research/13_quality_preservation.md`

### Image quilting
- Replace patches with clean-corpus matches — genuinely adaptive-attack-resistant
- Requires patch corpus (~50MB) — separate download or bundled
- Research: `research/03_feature_squeezing_defenses.md`

### BPDA/EOT evaluation suite
- Adaptive attack testing (differentiable approximations for each stage)
- Required before any robustness claims in papers
- Research: `research/11_evaluation_methodology.md`, `research/12_certified_defenses.md`

### Fuzz testing infrastructure
- libFuzzer targets for decode, validate, JPEG roundtrip
- OSS-Fuzz project submission
- 24-hour fuzz run gate before each release
- Research: `architecture/DEBATE.md` (Testing Architect)

---

## v0.4.0+ — Long-term vision

- **Rust port** — PyO3/maturin for even easier cross-platform builds
- **WASM target** — browser-side sanitization for web apps
- **GPU acceleration** — CUDA/Metal kernels for batch processing
- **Model-specific presets** — tuned parameters per VLM (GPT-4o, Gemini, Claude)
- **Streaming API** — sanitize chunks of video frames
- **Plugin system** — user-defined pipeline stages

---

## Research Corpus

All research is in `research/` (15 files, ~400KB). Key references per topic:

| Topic | File | Key finding |
|---|---|---|
| Prompt injection | `01_visual_prompt_injection.md` | 4 attack classes, FigStep most dangerous |
| Gradient attacks | `02_adversarial_perturbations.md` | >95% ASR on GPT-4o, randomization mandatory |
| Feature squeezing | `03_feature_squeezing_defenses.md` | 5-bit + median + JPEG = fast path |
| Frequency defenses | `04_frequency_domain_defenses.md` | Haar wavelet + BayesShrink strongest standalone |
| Steganography | `05_steganography_detection.md` | LSB >90% ASR on GPT-4o, bit-depth crush kills it |
| Malformed images | `06_malformed_image_security.md` | libspng > stb_image for security |
| Scaling attacks | `07_scaling_attacks.md` | INTER_AREA only safe interpolation |
| VLM jailbreaks | `08_multimodal_jailbreaks.md` | FigStep 82.5% ASR, needs OCR defense |
| C++ SIMD | `09_cpp_simd_optimization.md` | Highway, sorting network, cache tiling |
| Build system | `10_nanobind_build_system.md` | nanobind + scikit-build-core patterns |
| Evaluation | `11_evaluation_methodology.md` | BPDA+EOT mandatory for robustness claims |
| Certified defenses | `12_certified_defenses.md` | Randomized smoothing 150s/image (impractical) |
| Quality preservation | `13_quality_preservation.md` | SSIM ≥ 0.85, LPIPS ≤ 0.15, bilateral > gaussian |
| Spatial smoothing | `14_spatial_smoothing_algorithms.md` | CTMF, Van Vliet IIR, Chambolle-Pock |
| Competition | `15_existing_defense_libraries.md` | No compiled VLM defense lib exists |

Architecture debate and decisions: `architecture/DEBATE.md`, `architecture/DECISIONS.md`
C++ implementation references: `architecture/CPP_*.md`
