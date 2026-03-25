# pixmask v0.1.0 — Final Architecture Decisions

> Synthesized from 5-architect debate (1978 lines). Conflicts resolved below with rationale.

---

## Conflict Resolutions

### 1. Default Profile: `balanced` (3-to-1 vote)

**Decision: `balanced`** — Security Architect overruled.

Rationale: API Designer, Performance Engineer, and Minimalist all argue that `paranoid` as default causes user abandonment ("why is this so slow?"). The compromise:
- Default is `balanced` (<15ms at 1080p, SSIM ≥ 0.88)
- First import prints ONE-TIME `UserWarning` explaining `paranoid` exists
- `pixmask.sanitize(img, preset="paranoid")` is always available
- NO per-call `SecurityWarning` (trains users to suppress)

### 2. Decoders: stb_image for v0.1, libspng upgrade in v0.2

**Decision: stb_image with format restriction** — Compromise between Security and Minimalist.

Rationale: libspng + libjpeg-turbo + libwebp adds significant build complexity for v0.1. stb_image with `STBI_ONLY_JPEG` + `STBI_ONLY_PNG` (GIF/BMP/TGA/PSD disabled at compile time) + Stage 0 magic-byte gating gives acceptable security for initial release. v0.2 upgrades to libspng + libjpeg-turbo.

Risk acceptance: stb_image PNG has no known critical CVEs when GIF is disabled. The Stage 0 validation gate catches malformed inputs before they reach the parser.

### 3. v0.1.0 Pipeline (6 stages, ~2000 lines C++)

**Decision: Minimalist pipeline + Security's validation gate**

```
Stage 0: VALIDATE      — magic bytes, dimensions ≤ 8192, file size ≤ 50MB, decomp ratio ≤ 100×
Stage 1: DECODE        — stb_image (JPEG+PNG only, GIF/BMP/TGA compile-disabled)
Stage 2: STRIP META    — zero out EXIF/XMP/ICC (re-encode handles this implicitly)
Stage 3: BIT-DEPTH     — 8→5 bits per channel, SIMD (Highway)
Stage 4: MEDIAN 3×3    — sorting network (19-step Bose-Nelson), SIMD
Stage 5: JPEG RT       — encode QF=random(70,85) + decode (stb_image_write JPEG + stb decode)
Stage 6: OUTPUT        — return sanitized pixel buffer
```

**What's deferred:**
| Feature | Version | Rationale |
|---|---|---|
| Bilateral filter | v0.2 | Pareto-superior to median, but naive impl needs range LUT |
| Gaussian blur | v0.2 | 3-pass box blur, complements bilateral |
| Wavelet denoise | v0.2 | Haar DWT inline, strongest standalone defense |
| Pixel deflection | v0.2 | Stochastic, non-differentiable — adds adaptive resistance |
| Safe resize (INTER_AREA + jitter) | v0.2 | Scaling attack defense |
| libspng/libjpeg-turbo decoders | v0.2 | Security upgrade from stb |
| Stego chi-square detection | v0.2 | Detection signal, not just destruction |
| OCR/typographic detection | v0.3+ | Architecturally separate — needs text safety classifier |
| TV denoising | v0.3+ | Iterative, 50ms+, marginal over simpler transforms |
| Image quilting | v0.3+ | Requires patch corpus |
| Feature squeezing detection | v0.3+ | Needs model forward pass |

### 4. Dependencies for v0.1.0

| Dependency | Status | How |
|---|---|---|
| stb_image.h | VENDOR | Single header, compile-time format restriction |
| stb_image_write.h | VENDOR | Single header, JPEG encode only |
| Google Highway | VENDOR | FetchContent in CMake, SIMD abstraction |
| nanobind | BUILD-TIME | scikit-build-core integration |
| doctest | VENDOR | Single header, C++ unit tests |
| FFTW | **REJECTED** | GPL contamination — permanent veto |
| OpenCV | **REJECTED** | Runtime dep — 50MB, libGL issues |
| libspng | v0.2 | Security upgrade |
| libjpeg-turbo | v0.2 | Performance + security upgrade |

### 5. C++ Architecture

```
namespace pixmask {

// Core types
struct ImageView {
    uint8_t* data;
    uint32_t width;
    uint32_t height;
    uint32_t channels;  // 1, 3, or 4
    uint32_t stride;    // bytes per row (≥ width * channels)
};

struct SanitizeOptions {
    uint8_t  bit_depth       = 5;          // 1-8
    uint8_t  median_radius   = 1;          // kernel = 2r+1 (1 = 3×3)
    uint8_t  jpeg_quality_lo = 70;
    uint8_t  jpeg_quality_hi = 85;
    uint32_t max_width       = 8192;
    uint32_t max_height      = 8192;
    uint64_t max_file_bytes  = 50ULL << 20; // 50MB
    uint32_t max_decomp_ratio = 100;
};

struct SanitizeResult {
    ImageView image;           // sanitized output (owned by Arena)
    bool      success;
    uint32_t  error_code;      // 0 = ok
    const char* error_message;
};

// Main entry point
SanitizeResult sanitize(const uint8_t* input_bytes, size_t input_len,
                        const SanitizeOptions& opts = {});

// Arena-backed memory (per-pipeline, reusable)
class Arena {
    // bump-pointer allocator, 32MB default, grows if needed
};

} // namespace pixmask
```

### 6. Python API

```python
# Level 0: One-liner
import pixmask
safe = pixmask.sanitize(image)  # accepts ndarray, bytes, Path, PIL.Image

# Level 1: Presets
safe = pixmask.sanitize(image, preset="fast")      # bit-depth + JPEG only
safe = pixmask.sanitize(image, preset="balanced")   # default — all v0.1 stages
safe = pixmask.sanitize(image, preset="paranoid")   # v0.2+ (warns if not available)

# Level 2: Custom options
safe = pixmask.sanitize(image, bit_depth=4, jpeg_quality=(60, 80))

# Level 3: Bytes I/O (for API serving)
safe_bytes = pixmask.sanitize(raw_bytes, output_format="jpeg")

# Return type: numpy ndarray by default
# With return_metadata=True: SanitizeResult dataclass with .image + .warnings
```

### 7. Testing Strategy

**Framework**: doctest (single header, fastest compile)

**Unit Tests (v0.1.0 gate criteria):**
- Bit-depth: all depths 1-8, edge values {0, 127, 128, 255}
- Median 3×3: uniform, gradient, impulse noise patterns
- JPEG roundtrip: QF range, channel preservation
- Validation: oversized, zero-dim, corrupt headers, wrong magic bytes
- All preset produce valid output with SSIM ≥ threshold

**Integration Tests:**
- Clean image roundtrip: SSIM ≥ 0.88 for balanced
- Known PGD perturbation: L2 norm reduced by ≥ 50%
- Malformed input: no crash, graceful error
- Memory: ASan + UBSan on every PR

**Performance Tests:**
- Google Benchmark: resolution matrix (224, 512, 1024, 2048)
- p99 latency gate: <15ms at 512×512 for balanced
- 10% regression gate in CI

**Security Tests (v0.1.0):**
- Fuzz targets for stb_image paths (libFuzzer)
- CVE reproduction tests with ASan
- Decompression bomb rejection test

### 8. Directory Structure

```
pixmask/
├── CMakeLists.txt
├── pyproject.toml
├── LICENSE                          # MIT
├── README.md
├── src/
│   ├── cpp/
│   │   ├── include/pixmask/
│   │   │   ├── pixmask.h           # Main C API header
│   │   │   ├── arena.h             # Bump-pointer allocator
│   │   │   ├── validate.h          # Stage 0: input validation
│   │   │   ├── decode.h            # Stage 1: stb_image wrapper
│   │   │   ├── bitdepth.h          # Stage 3: bit-depth reduction (Highway)
│   │   │   ├── median.h            # Stage 4: sorting network median (Highway)
│   │   │   ├── jpeg_roundtrip.h    # Stage 5: JPEG encode+decode
│   │   │   └── pipeline.h          # Orchestrator
│   │   ├── src/
│   │   │   ├── pixmask.cpp         # Implementation
│   │   │   ├── validate.cpp
│   │   │   ├── decode.cpp
│   │   │   ├── bitdepth.cpp        # Highway SIMD dispatch
│   │   │   ├── median.cpp          # Highway SIMD dispatch
│   │   │   ├── jpeg_roundtrip.cpp
│   │   │   └── pipeline.cpp
│   │   ├── bindings/
│   │   │   └── module.cpp          # nanobind Python bindings
│   │   └── third_party/
│   │       ├── stb_image.h
│   │       ├── stb_image_write.h
│   │       └── doctest.h
│   └── tests/
│       ├── cpp/
│       │   ├── test_bitdepth.cpp
│       │   ├── test_median.cpp
│       │   ├── test_validate.cpp
│       │   ├── test_jpeg.cpp
│       │   ├── test_pipeline.cpp
│       │   └── fuzz/
│       │       ├── fuzz_decode.cpp
│       │       └── fuzz_validate.cpp
│       └── python/
│           ├── test_sanitize.py
│           ├── test_presets.py
│           ├── test_integration.py
│           └── conftest.py
├── python/
│   └── pixmask/
│       ├── __init__.py
│       ├── __init__.pyi             # Type stubs
│       └── _version.py
├── benchmarks/
│   ├── bench_pipeline.cpp
│   └── bench_individual.cpp
├── research/                        # 15 research files (read-only reference)
├── architecture/                    # Debate + decisions (read-only reference)
└── .github/
    └── workflows/
        ├── ci.yml                   # Test + lint + ASan on every PR
        └── wheels.yml               # cibuildwheel release
```

---

## Implementation Order

1. **Core infra**: `arena.h`, `pixmask.h` (types), `ImageView`
2. **Validation**: `validate.h/cpp` — magic bytes, dimensions, file size
3. **Decode**: `decode.h/cpp` — stb_image wrapper with format restriction
4. **Bit-depth**: `bitdepth.h/cpp` — Highway SIMD, all depths 1-8
5. **Median**: `median.h/cpp` — 19-step Bose-Nelson sorting network, Highway
6. **JPEG roundtrip**: `jpeg_roundtrip.h/cpp` — encode with random QF + decode
7. **Pipeline**: `pipeline.h/cpp` — orchestrate stages, arena management
8. **Python bindings**: `module.cpp` — nanobind, `sanitize()` function
9. **Tests**: Unit + integration + fuzz
10. **Build**: CMakeLists.txt, pyproject.toml, CI workflows
11. **Benchmarks**: Google Benchmark suite

---

*This document is the authoritative source for all v0.1.0 implementation decisions.*
