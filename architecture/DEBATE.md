# pixmask Architecture Debate

> 5 agents with different perspectives debate and converge on the final architecture.
> Each agent appends their arguments. Goal: converge on a final DECISIONS section.

## Context
Read `research/FINDINGS.md` for the consolidated research. This debate decides:
1. Exact feature set for v0.1.0 (MVP)
2. C++ API design (classes, namespaces, function signatures)
3. Python API design (module structure, function names, preset configs)
4. What to include vs exclude (scope boundaries)
5. Dependencies: which external libs to vendor vs link vs reject
6. Pipeline ordering and default parameters
7. Testing strategy (unit + integration + benchmark)
8. File/directory structure

---

## Security Architect

**Position**: Maximum defense coverage with safe defaults. Complexity is acceptable when it closes real attack surface. Convenience is never a reason to leave a known attack class undefended in the default path.

---

### 1. MVP Feature Set — What MUST Be in v0.1.0

**Non-negotiable from a security standpoint:**

The attack class ordering in FINDINGS.md is the priority queue. v0.1.0 must defend against classes 2–5 completely and class 6 partially. Class 1 (typographic) is architecturally different — I address it in scope boundaries below.

**Minimum Viable Defense Pipeline (every stage is mandatory, none optional at default):**

```
Stage 0: VALIDATION GATE (hard block — reject before decode)
  - Magic byte verification (not file extension — extensions are attacker-controlled)
  - Dimension limits: MAX 8192×8192, reject anything larger without explicit override
  - File size limit: MAX 50MB raw input
  - Decompression ratio limit: MAX 100× (stops zip/PNG bombs)
  - Allowlist of formats: PNG, JPEG, WebP ONLY — anything else is a hard reject

Stage 1: SAFE DECODE (parser security is the foundation)
  - libspng for PNG — non-negotiable (see dependency section)
  - libjpeg-turbo >= 3.0.4 for JPEG
  - libwebp >= 1.3.2 for WebP
  - All decoders run with explicit pixel budget enforcement

Stage 2: METADATA STRIP (zero-cost, eliminates entire data exfiltration channel)
  - Strip ALL EXIF, XMP, ICC profiles, custom chunks
  - This is not optional — metadata is a free covert channel and costs nothing to destroy

Stage 3a: BIT-DEPTH REDUCTION (8→5 bits per channel)
  - Collapses adversarial perturbations that live in low-order bits
  - Destroys LSB steganography as a side effect
  - Must be the FIRST pixel operation — order matters, doing this after smoothing loses the guarantee

Stage 3b: SPATIAL SMOOTHING — bilateral filter (σ_s=5, σ_r=15)
  - Edge-preserving: maintains semantic content while destroying high-frequency adversarial patterns
  - Must come AFTER bit-depth reduction

Stage 3c: WAVELET DENOISE — Haar DWT + BayesShrink (σ=0.04)
  - Strongest standalone defense against gradient perturbations per the research
  - Operates in frequency domain, complementary to spatial smoothing

Stage 4: JPEG ROUND-TRIP (QF randomized 70–85, randomized per image)
  - Destroys DCT-domain steganography (F5, OutGuess, model-based stego)
  - Randomized QF is critical — fixed QF is bypassable by crafting images that survive a known quantization matrix
  - The randomization must be seeded from a CSPRNG, not rand()

Stage 5: SAFE RESIZE (INTER_AREA + random jitter ±5%)
  - Destroys scaling attacks (Quiring et al.)
  - Jitter must be present even when caller does not resize — apply it at native resolution before output

Stage 6: RE-ENCODE (clean output, sanitized container)
  - Output format is always sanitizer-controlled, never passthrough of input container
```

**What this pipeline addresses:**

| Attack Class | Stage(s) That Kill It | Residual Risk |
|---|---|---|
| Gradient perturbations (PGD, C&W) | 3a + 3b + 3c + 4 | <5% ASR (non-adaptive) |
| LSB steganography | 3a (bit crush destroys it) | ~0% |
| DCT/frequency-domain stego | 4 (JPEG round-trip) | ~0% |
| Neural steganography | 3b + 3c + 4 (layered) | 5-15% residual |
| Scaling attacks | 5 | <2% |
| Parser exploits (malformed images) | 0 + 1 (allowlist + safe decoders) | CVE-dependent |
| Composite (HADES) | All stages layered | 10-20% residual |

**The residual risk on composite attacks is acceptable for v0.1.0 because**: HADES requires white-box access to the preprocessing pipeline. In non-adaptive threat model this drops to near-zero.

---

### 2. Scope Boundaries — IN vs OUT

**OCR Detection (Typographic / FigStep attacks):**

OUT of v0.1.0 — but this decision needs an explicit architectural commitment, not a vague "future work" note.

The reason is architectural, not laziness: OCR-based detection requires a text safety classifier downstream of the OCR layer. pixmask is a sanitization layer, not a content moderation layer. Mixing them creates an API contract problem — what does pixmask return when OCR detects "Ignore previous instructions"? A modified image? A rejection? An annotation? This needs a separate interface contract that should not be entangled with pixel sanitization.

**What MUST happen in v0.1.0 to address this gap:**

1. The README and API docs must explicitly state pixmask does NOT defend against typographic injection
2. The `SanitizeResult` struct must include a `warnings` field with a `TYPOGRAPHIC_ATTACK_NOT_DEFENDED` flag that fires unconditionally — making the gap visible in every integration, not just the docs
3. The API surface must reserve space for a future `TextDetectionCallback` hook so the interface doesn't need to break when this is added

**Steganography DETECTION (not just destruction):**

The current pipeline destroys steganography as a side effect. I argue we should also EXPOSE a detection signal in v0.1.0, not just destruction.

Rationale: An operator who sees repeated steganographic injection attempts on their endpoint needs to know about it. Silent sanitization without detection gives no signal for incident response. The detection mechanism is cheap — Chi-square test on LSB planes costs microseconds and is well-understood.

**What to include:**

- LSB chi-square test (runs on pre-sanitization pixel data, before Stage 3a)
- Result exposed in `SanitizeResult::stego_suspicion_score` (float 0.0–1.0)
- Threshold for "suspicious" configurable, default 0.85

This is detection signal, NOT a block gate — sanitization always runs regardless of score. The caller decides what to do with the score (log it, alert, rate-limit the source).

**Feature Squeezing Detection Mechanism:**

YES — expose the squeezed vs. unsqueezed comparison signal. This is the Xu et al. feature squeezing detector from NDSS 2018 and it costs essentially nothing since we're already computing both representations.

Mechanism: compare SSIM between raw-decoded image and post-sanitization image. Large delta (SSIM < 0.85) indicates the input was adversarially perturbed. Expose as `SanitizeResult::adversarial_perturbation_score`.

This gives operators a threat signal without adding any latency to the hot path — the comparison happens post-hoc on the already-computed outputs.

---

### 3. Safe Defaults — No Opt-In Required for Safety

**Default pipeline must be `paranoid`, not `balanced`.**

The research team's instinct to make `balanced` the default is wrong from a security perspective. Here is the argument:

- Performance cost of `paranoid` vs `balanced` is ~17ms at 1080p. This is acceptable for a security layer.
- The cost of an LLM jailbreak in production is orders of magnitude higher than 17ms of latency.
- Users who need lower latency can explicitly opt down to `balanced` or `fast` — they make an informed choice.
- Users who install pixmask and use defaults expect maximum protection. Giving them `balanced` by default is a trust violation.

If the performance team wins this argument and `balanced` becomes default, then I require that the Python API emit a `SecurityWarning` (not just a log line — an actual Python warning via the `warnings` module) when the non-paranoid profile is used without an explicit `allow_reduced_security=True` parameter. The caller must acknowledge the tradeoff.

**Unknown format handling:**

Hard reject. No exceptions in default mode. The allowlist is: PNG, JPEG, WebP. Everything else gets `PixmaskError::UnsupportedFormat`. The caller can inspect the error and decide, but pixmask will not attempt to decode an unrecognized format.

GIF: REJECT. stb_image has open exploits on GIF. Even if we used a different decoder, animated GIFs are a separate attack surface (timing side channels via frame count, palette injection). Out of scope.

TIFF: REJECT. 50+ year old format with massive parser complexity. LibTIFF CVE history is a horror show. Any legitimate TIFF input should be converted upstream before reaching pixmask.

SVG: REJECT. Not a raster format. XXE injection, script execution, external resource loading. Never.

**Memory limits (hard limits, not configurable without explicit override):**

```cpp
static constexpr size_t MAX_INPUT_BYTES      = 50 * 1024 * 1024;  // 50MB
static constexpr uint32_t MAX_DIMENSION      = 8192;               // px per side
static constexpr uint64_t MAX_PIXEL_COUNT    = 33554432ULL;        // 32MP (8192²/2 approx)
static constexpr uint32_t MAX_DECOMP_RATIO   = 100;                // vs compressed size
static constexpr size_t SCRATCH_BUFFER_LIMIT = 512 * 1024 * 1024; // 512MB peak working set
```

The decompression ratio limit is the most critical one that is often missed. A 500KB PNG file can decompress to 500MB. Without this check, Stage 0 is not actually a security gate.

**Randomized QF seed:**

The JPEG QF randomization (Stage 4) MUST use a CSPRNG. Specifically: `std::random_device` seeded `std::mt19937_64` is NOT acceptable for security use — mt19937 is not cryptographically secure. Use OS entropy directly: `getrandom()` on Linux, `BCryptGenRandom()` on Windows, `getentropy()` on macOS. The randomization prevents adaptive attacks that optimize for survival of a known QF.

---

### 4. Dependency Decisions

**libspng vs stb_image for PNG — this is not a debate:**

libspng. Unconditionally. stb_image's PNG decoder has known memory safety issues. stb_image is a single-header convenience library written for game development, not adversarial input handling. The research team already called this correctly.

Specific problem with stb_image: it does not validate chunk CRC fields, silently accepts malformed iCCP chunks (exploited in CVE-2023-4863 class attacks), and has no fuzzing corpus maintained by a security team. libspng has CERT C compliance documentation and an active security maintainer.

**Vendoring vs dynamic linking:**

Vendor all security-critical parsers. Link dynamically for nothing.

Rationale: dynamic linking means the installed system library version determines security properties. On Ubuntu 20.04 LTS, the system libwebp is 0.6.1 — which is vulnerable to CVE-2023-4863. Operators who `pip install pixmask` on LTS systems will silently get the vulnerable version.

Vendored libraries pinned to exact versions with verified checksums (SHA-256 of source tarball in CMakeLists.txt, checked at configure time). This is non-negotiable for a security library.

Build process: vendor via FetchContent with VERIFY_HASH, or git submodule at pinned commit. Not vcpkg (version resolution is non-deterministic). Not Conan (same problem).

**FFTW vs inline Haar DWT:**

Implement Haar DWT inline. No FFTW dependency.

Haar DWT does not need FFTW. FFTW is for arbitrary-length FFTs. Haar is just `(a+b)/2` and `(a-b)/2` recursively — it's 20 lines of C++. Adding FFTW as a dependency for this is:

1. A significant attack surface increase (FFTW itself has had memory safety issues)
2. A licensing problem (FFTW is GPL, which would infect the library's license unless the commercial version is purchased)
3. Completely unnecessary — the performance advantage of FFTW over inline Haar on small image tiles is zero because Haar is already O(n)

The FFTW GPL issue alone is a hard veto. pixmask needs to be usable in commercial ML pipelines.

**OpenCV dependency:**

REJECT as a runtime dependency. OpenCV is 50MB and requires system libGL. It can be used in test infrastructure (to generate adversarial examples for the test suite) but must not appear in the pixmask runtime dependency tree. Every filter we need — bilateral, resize, Gaussian — is implementable inline with SIMD. The research team already confirmed this.

---

### 5. Non-Negotiable Security Tests

The test suite must include ALL of the following before v0.1.0 ships. These are gate criteria for release, not aspirational:

**Parser Fuzzing (mandatory, not optional):**
- LibFuzzer corpus for each decoder entry point (PNG, JPEG, WebP)
- OSS-Fuzz integration from day one (submit project to OSS-Fuzz before first PyPI release)
- Minimum 24-hour fuzz run before each minor release
- Corpus includes: truncated files, malformed headers, zip-bomb inputs, files with MAX_DIMENSION+1 pixels, files claiming wrong dimensions, files with invalid color profiles

**Memory Safety:**
- ASan + UBSan CI pass on every PR (not just nightly)
- Valgrind memcheck on the full pipeline with corpus
- No use of `new`/`delete` in hot path — arena allocator must be tested for double-free and OOB under adversarial allocation patterns

**Adversarial Attack Reduction (ASR benchmarks, gate criteria):**

| Attack | Baseline ASR | Required Post-pixmask ASR | Failure = Release Block |
|---|---|---|---|
| PGD-40 (ε=8/255) | >90% | <10% | YES |
| C&W L2 | >85% | <10% | YES |
| LSB steganography (full payload) | ~100% survive | 0% survive | YES |
| DCT steganography (F5) | ~100% survive | <5% survive | YES |
| Scaling attack (Quiring 2020) | model-dependent | <5% | YES |
| Malformed PNG (zip bomb 50× ratio) | code exec | hard reject | YES |
| 8193×8192 image | OOM | hard reject | YES |

**Regression Tests (must stay green):**
- Clean image SSIM through `paranoid` pipeline: SSIM >= 0.82, PSNR >= 28 dB
- Clean image SSIM through `balanced` pipeline: SSIM >= 0.88, PSNR >= 30 dB
- Processing latency at 512×512: p99 < 15ms on reference hardware (define reference hardware in CI)
- Zero output file contains EXIF data (test with exiftool in CI)
- Zero output file contains ICC profile (strip verification)

**Threat Model Test (integration, not unit):**
- Feed a FigStep-style image through the pipeline, confirm it passes through (sanitizer does NOT claim to stop it), and confirm the `TYPOGRAPHIC_ATTACK_NOT_DEFENDED` warning flag is set in the result

**Supply Chain:**
- Verify SHA-256 of all vendored library source tarballs in CI (prevents build-time compromise)
- Reproducible builds: two independent build environments must produce bit-identical binaries for the same platform target

---

### Summary of Hard Requirements (Security Architect Position)

| Decision | Position |
|---|---|
| Default profile | `paranoid` — or emit `SecurityWarning` if not |
| Unknown formats | Hard reject, no fallback |
| GIF, TIFF, SVG | Permanently rejected, no override |
| PNG decoder | libspng only |
| Vendoring | All security-critical parsers vendored, pinned, checksummed |
| FFTW | Never — GPL contamination + unnecessary |
| Haar DWT | Inline implementation, trivial |
| Stego detection | Chi-square signal in `SanitizeResult`, v0.1.0 |
| Feature squeezing signal | SSIM delta in `SanitizeResult`, v0.1.0 |
| OCR/typographic | Out of scope, but warning flag mandatory |
| CSPRNG for QF | OS entropy (getrandom/BCryptGenRandom), not mt19937 |
| Dimension limits | 8192px, 32MP, 50MB, 100× decomp ratio — hard coded defaults |
| Fuzzing | LibFuzzer + OSS-Fuzz, gate criterion for release |
| ASan/UBSan | Every PR, not nightly |

The performance architect will argue some of this is too expensive. My counter: a sanitization library that can be bypassed is worse than no sanitization library, because it creates false confidence. The overhead of `paranoid` is 25ms. The cost of a jailbreak in a production LLM application is measured in reputational and legal liability. The tradeoff is not close.

---

## API/UX Designer

**Position: Developer Experience — pixmask must be dead-simple to use.**

The fastest path to adoption is zero-friction onboarding. If a developer cannot sanitize an image in 30 seconds from `pip install`, they will not use pixmask. Every design decision below is argued from that constraint. Security coverage and performance mean nothing if the library sits uninstalled.

---

### 1. Python API Design

#### Level 0: One-liner (the 90% case)

```python
import pixmask
safe_img = pixmask.sanitize(img)  # numpy array in, numpy array out
```

This is the entire API for most users. `img` accepts:
- `numpy.ndarray` (HWC, uint8, RGB) — the dominant format in the ecosystem
- `PIL.Image.Image` — auto-detected, zero boilerplate for PIL users
- `bytes` / `bytearray` — raw encoded image data (JPEG/PNG/WebP bytes from HTTP responses)
- `pathlib.Path` / `str` — file path, read + sanitize + return array

Return type is always `numpy.ndarray` (HWC, uint8, RGB) by default. No surprises.

Full signature:

```python
def sanitize(
    image: Union[np.ndarray, "PIL.Image.Image", bytes, bytearray, str, Path],
    *,
    preset: str = "balanced",
    output_format: str = "numpy",   # "numpy" | "pil" | "bytes"
    return_metadata: bool = False,
) -> Union[np.ndarray, "PIL.Image.Image", bytes, "SanitizeResult"]:
    ...
```

The `*` enforces keyword-only arguments after `image`. This prevents `sanitize(img, "fast")` positional misuse while keeping the one-liner clean. Note on the Security Architect's `paranoid`-as-default argument: I agree with the safety instinct, but `balanced` is the right default. A default that imposes 25ms latency without warning will cause developers to either abandon the library or immediately override the default without understanding why — both outcomes are worse than a well-documented `balanced` default that users can consciously upgrade.

#### Level 1: Presets

```python
safe_img = pixmask.sanitize(img, preset="fast")       # ~3ms — high-throughput
safe_img = pixmask.sanitize(img, preset="balanced")   # ~8ms — default
safe_img = pixmask.sanitize(img, preset="paranoid")   # ~25ms — adversarial-aware
```

Preset names are intentional. `"fast"` communicates the tradeoff immediately. `"balanced"` signals it is a reasonable default. `"paranoid"` signals maximum security and communicates the cost is worth it in high-sensitivity contexts.

Do NOT name presets `"low"` / `"medium"` / `"high"` — those are vague and require reading docs to interpret. Do NOT use numeric quality levels like JPEG — those require domain knowledge.

#### Level 2: Custom pipeline (the power-user case)

```python
pipeline = pixmask.Pipeline([
    pixmask.steps.BitDepthReduce(bits=5),
    pixmask.steps.BilateralSmooth(sigma_s=5, sigma_r=15),
    pixmask.steps.PixelDeflect(k=100, radius=10),
    pixmask.steps.JpegRoundtrip(quality_range=(70, 85)),
    pixmask.steps.SafeResize(target_size=(512, 512)),
])
safe_img = pipeline(img)
```

`Pipeline` is callable (not `.run()` or `.process()` — callable objects are the Python idiom for transforms). Each step lives in `pixmask.steps` to keep the top-level namespace clean.

Pipeline must be composable and inspectable:

```python
# Inspect what a preset does
print(pixmask.get_preset("balanced"))
# Pipeline([BitDepthReduce(bits=5), BilateralSmooth(sigma_s=5, sigma_r=15), ...])

# Extend a preset
pipeline = pixmask.get_preset("balanced") + pixmask.steps.WaveletDenoise(sigma=0.04)

# Slice for debugging
partial = pipeline[:2]
```

`Pipeline.__repr__` must be human-readable. Inspectable objects build user trust and accelerate debugging.

#### Level 3: File I/O

```python
# Returns output path
out_path = pixmask.sanitize_file("input.jpg", "output.png")

# In-place overwrite (explicit opt-in, not default)
pixmask.sanitize_file("photo.jpg", overwrite=True)

# Batch — returns list of output paths
out_paths = pixmask.sanitize_batch(
    input_dir="./uploads/",
    output_dir="./sanitized/",
    preset="balanced",
    workers=4,
    glob="*.{jpg,png,webp}",
)

# Streaming batch with progress
for result in pixmask.sanitize_batch_iter(paths, preset="balanced"):
    print(result.path, result.latency_ms, result.warnings)
```

`sanitize_batch` matters for MLOps pipelines. The GIL-free nanobind binding means real parallel throughput without multiprocessing overhead.

---

### 2. Naming Conventions

**Module top-level** (`pixmask.*`):

| Name | Type | Purpose |
|---|---|---|
| `pixmask.sanitize` | function | Primary entry point |
| `pixmask.sanitize_file` | function | File I/O variant |
| `pixmask.sanitize_batch` | function | Parallel directory processing |
| `pixmask.sanitize_batch_iter` | function | Streaming variant with progress |
| `pixmask.async_sanitize` | coroutine | Async-compatible wrapper |
| `pixmask.Pipeline` | class | Custom pipeline builder |
| `pixmask.get_preset` | function | Returns Pipeline for a named preset |
| `pixmask.list_presets` | function | Introspection |
| `pixmask.config` | object | Module-level configuration |

**Steps submodule** (`pixmask.steps.*`):

| Class | Parameters | Defense target |
|---|---|---|
| `BitDepthReduce` | `bits=5` | Gradient perturbations, LSB stego |
| `BilateralSmooth` | `sigma_s=5, sigma_r=15` | High-frequency adversarial noise |
| `GaussianSmooth` | `sigma=1.0` | Fast alternative to bilateral |
| `WaveletDenoise` | `sigma=0.04` | Strongest standalone gradient defense |
| `PixelDeflect` | `k=100, radius=10` | Stochastic non-differentiable defense |
| `JpegRoundtrip` | `quality_range=(70, 85)` | DCT-domain stego destruction |
| `MetadataStrip` | *(none)* | EXIF/XMP/ICC removal |
| `SafeResize` | `target_size, jitter=0.05` | Scaling attacks |
| `ValidateInput` | `max_size=(8192,8192)` | Parser security gate |

Naming rules:
- Classes use noun phrases, not verb phrases (`BitDepthReduce`, not `ReduceBitDepth`)
- Parameters use snake_case throughout — no abbreviations in public API
- `Jpeg` not `JPEG` in class names (PEP 8: treat acronyms as words when mid-identifier)
- No `Image` in class names — conflicts with PIL's dominant `PIL.Image` namespace
- No `Filter` suffix — implies reversibility; pixmask operations are lossy by design

---

### 3. Error Handling

Three exception types, one base:

```python
class PixmaskError(Exception):
    """Base class for all pixmask exceptions."""

class InputError(PixmaskError):
    """Invalid or unsupported input — caller's responsibility to fix."""
    # Unsupported format (GIF, TIFF, SVG), wrong dtype, wrong ndim,
    # dimensions exceeding max_size, file size over limit

class CorruptImageError(PixmaskError):
    """Image data is malformed or failed safe parsing."""
    # Truncated file, decompression bomb detected, magic byte mismatch,
    # parser error forwarded from libspng/libjpeg-turbo/libwebp

class PipelineError(PixmaskError):
    """A pipeline stage failed unexpectedly — likely a pixmask bug."""
    # Internal C++ assertion failures, OOM in arena allocator
```

**Behavior principles:**

`InputError` raises immediately before any processing (fast-fail). Error messages must name the actual problem and the remedy:
- `"GIF images are not supported (parser attack surface). Convert to PNG or JPEG first."` — not `"Unsupported format"`.
- `"Input array has dtype float32. pixmask expects uint8 in range [0, 255]."` — not `"Invalid input"`.
- `"Image dimensions 9000x6000 exceed the maximum 8192x8192. Pass max_size=(9000, 6000) to override."` — not `"Image too large"`.

For corrupt files, `CorruptImageError` forwards the underlying parser message as context. The C++ layer translates all exceptions to Python via nanobind's exception translation mechanism — no `std::terminate`, no bare `RuntimeError("C++ error")`.

**Warnings (non-fatal, use Python's `warnings` module):**

```python
import warnings

# Emitted as PixmaskWarning, not exceptions:
# - RGBA input: alpha channel stripped, processing continues
# - Grayscale input: promoted to 3-channel, processing continues
# - float32 input in [0,1]: auto-converted to uint8, processing continues
# - Image exceeds recommended size for performance
```

`stacklevel=2` is mandatory on all `warnings.warn` calls so the warning points to the caller's line, not pixmask internals. This is the single most common mistake in library warning design and it destroys debugging experience when wrong.

**Recoverable inputs — never raise, always warn and continue:**
- RGBA: strip alpha channel with `PixmaskWarning`, proceed
- Grayscale HW or HW1: promote to HWC-3 with `PixmaskWarning`, proceed
- float32 in [0, 1]: convert to uint8 with `PixmaskWarning`, proceed

The Security Architect and I agree on hard-rejecting GIF/TIFF/SVG. We agree the caller must be told why, not just `"unsupported"`. Where I differ: I oppose emitting an unconditional `TYPOGRAPHIC_ATTACK_NOT_DEFENDED` warning on every sanitize call (the Security Architect proposes this). It will be immediately suppressed with `warnings.filterwarnings("ignore")` by every production user, which is worse than not having the warning. The correct place for that information is the README and a one-time startup warning, not a per-image noise source.

---

### 4. Return Types

**Default: always `numpy.ndarray` (HWC, uint8, RGB).**

Do not invent a custom `Image` type as the default return. The ecosystem converges on numpy. OpenCV, PIL, torchvision, HuggingFace Datasets, Albumentations — they all operate on numpy arrays. A custom default return type forces users to call `.to_numpy()` and breaks every existing pipeline.

**`SanitizeResult` — opt-in via `return_metadata=True`:**

```python
result = pixmask.sanitize(img, return_metadata=True)

result.image          # np.ndarray — the sanitized image (always present)
result.ssim           # float | None — SSIM vs input (only if computed)
result.latency_ms     # float — wall time for the full pipeline
result.warnings       # list[str] — non-fatal issues encountered
result.stages_run     # list[str] — which stages executed
result.preset         # str — which preset was used
result.stego_score    # float | None — chi-square LSB suspicion score (Security Architect's request)
result.perturbation_score  # float | None — feature squeezing delta signal
```

`SanitizeResult` is a `dataclass` (not a namedtuple — default values and docstrings require it). It implements `__array__` so `np.asarray(result)` works transparently. This means existing code that expects a numpy array will not break if someone accidentally passes `return_metadata=True` deep in a call stack.

I support including `stego_score` and `perturbation_score` in `SanitizeResult` — the Security Architect's detection signals are genuinely useful for incident response. They belong in the metadata object, not as mandatory return values.

**Output format control:**

```python
# PIL output — for PIL-first workflows
safe_pil = pixmask.sanitize(img, output_format="pil")

# Encoded bytes — for HTTP serving, eliminates the re-encode step
safe_bytes = pixmask.sanitize(img, output_format="bytes")
# Default encode format is PNG (lossless, deterministic)
# Override:
safe_jpeg = pixmask.sanitize(img, output_format="bytes", encode_format="jpeg", encode_quality=85)
```

The `output_format="bytes"` shortcut eliminates the `cv2.imencode` / `PIL.Image.save(BytesIO(...))` boilerplate that every API serving integration would otherwise duplicate.

---

### 5. Integration Patterns

#### OpenAI / Anthropic / Google Vision APIs

The dominant serving pattern: load image -> encode to base64 -> POST to API.

```python
# Without pixmask (user's existing code)
with open("user_upload.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

# With pixmask — one additional line
import pixmask
safe_bytes = pixmask.sanitize("user_upload.jpg", output_format="bytes")
img_b64 = base64.b64encode(safe_bytes).decode()
```

The `output_format="bytes"` path returning PNG bytes is critical here. Without it, users must write:

```python
import io
safe_img = pixmask.sanitize(img)                # numpy
pil_img = PIL.Image.fromarray(safe_img)          # convert
buf = io.BytesIO()
pil_img.save(buf, format="PNG")                 # encode
img_b64 = base64.b64encode(buf.getvalue()).decode()
```

That four-step boilerplate after the pixmask call will cause users to skip sanitization for "quick" scripts. The shortcut matters.

For Anthropic's API specifically, accepting bytes directly:

```python
import anthropic, pixmask, base64

client = anthropic.Anthropic()
safe_bytes = pixmask.sanitize("upload.jpg", output_format="bytes")

message = client.messages.create(
    model="claude-opus-4-5",
    messages=[{
        "role": "user",
        "content": [{
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.standard_b64encode(safe_bytes).decode(),
            },
        }],
    }],
)
```

#### LangChain / LlamaIndex

These frameworks use image loaders that return bytes or arrays. pixmask integrates as a transform:

```python
from langchain_community.document_loaders import ImageLoader
from pixmask.integrations.langchain import PixmaskTransform

loader = ImageLoader("./images/") | PixmaskTransform(preset="balanced")
docs = loader.load()
```

`pixmask.integrations.langchain` ships as a thin adapter (~30 lines) in v0.1.0 if LangChain is installed. It wraps `pixmask.sanitize` as a `BaseDocumentTransformer`. Same pattern for LlamaIndex as a `BaseReader` wrapper.

These live under `pixmask.integrations.*` — opt-in, never imported by default.

#### FastAPI / Flask serving

```python
from fastapi import FastAPI, UploadFile, HTTPException
import pixmask

app = FastAPI()

@app.post("/analyze")
async def analyze_image(file: UploadFile):
    raw = await file.read()
    try:
        safe_img = pixmask.sanitize(raw, preset="balanced")
    except pixmask.CorruptImageError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    except pixmask.InputError as e:
        raise HTTPException(status_code=422, detail=str(e))
    # pass safe_img to VLM
```

The `bytes` input type on `sanitize` makes this clean — no temp files, no disk I/O, no PIL intermediary.

**Async note:** pixmask is CPU-bound. The nanobind bindings release the GIL during C++ processing, so the event loop is not blocked. Document this explicitly — it is a key selling point for async server frameworks. Provide an `async_sanitize` coroutine as a convenience:

```python
safe_img = await pixmask.async_sanitize(raw, preset="balanced")
```

This is `run_in_executor` internally, but it removes a boilerplate pattern every async user will otherwise write themselves, and it signals to async-first developers that the library is async-aware.

#### HuggingFace Datasets / Transformers

```python
from datasets import load_dataset
import pixmask, numpy as np

dataset = load_dataset("...")
sanitized = dataset.map(
    lambda ex: {"image": pixmask.sanitize(np.array(ex["image"]))},
    num_proc=4,
)
```

This works today without any adapter — because `sanitize` accepts and returns numpy arrays. The HuggingFace integration story is: there is no integration story, it just works.

---

### 6. What NOT to Expose

These C++ internals must be completely hidden from the Python API surface:

**Never expose:**
- Arena allocator details — users do not manage pixmask's memory
- Tile dimensions (64x128 L1 cache blocks) — internal optimization, not a tuning knob
- SIMD dispatch decisions (SSE2 vs AVX2 vs NEON) — auto-detected at runtime
- Scratch buffer sizes — implementation detail
- Individual DCT or wavelet coefficients — pixmask is not a frequency analysis library
- Parser routing logic (which decoder handles which format) — implementation detail
- Raw pixel buffer pointers — memory safety boundary, must not cross into Python

**Never expose as configuration:**
- Tile size, arena size, SIMD width — users cannot meaningfully tune these
- Internal quality metrics used during processing (intermediate SSIM per stage)

**Rationale:** Every parameter you expose is a parameter the user can set incorrectly. The C++ layer is already optimized for the use case — trust the defaults, hide the machinery. The right abstraction level for the Python API is: which defense operations run, in what order, with what semantic parameters. Not: how does the bilateral filter tile its working memory.

**What IS appropriate to expose at Level 2:**
- Per-step semantic parameters (`bits`, `sigma_s`, `sigma_r`, `quality_range`) — clear domain meaning
- Stage composition — via `Pipeline` constructor
- Module-level safety limits (`max_image_size`, `max_file_size_mb`) — operators need to tune these for their infra
- Random seed for reproducibility — `pixmask.set_seed(42)` matters for testing

---

### 7. Module-Level Configuration

```python
# Set once at application startup — not per-call
pixmask.config.max_image_size = (4096, 4096)    # reject oversized inputs
pixmask.config.max_file_size_mb = 50             # reject oversized files
pixmask.config.default_preset = "balanced"       # override default preset
pixmask.config.raise_on_alpha = False            # strip alpha silently vs warn
```

This is preferable to environment variable hacks and preferable to threading config through every function call. Document `config` writes as not thread-safe — only read during processing, only written at startup.

---

### 8. The `__all__` Contract

Public API surface for v0.1.0 — nothing more, nothing less:

```python
__all__ = [
    # Primary functions
    "sanitize",
    "sanitize_file",
    "sanitize_batch",
    "sanitize_batch_iter",
    "async_sanitize",
    # Pipeline
    "Pipeline",
    "get_preset",
    "list_presets",
    # Return type
    "SanitizeResult",
    # Exceptions
    "PixmaskError",
    "InputError",
    "CorruptImageError",
    "PipelineError",
    "PixmaskWarning",
    # Config
    "config",
    "set_seed",
    # Submodules
    "steps",
    "integrations",
]
```

Everything else is `_` prefixed and considered private. Version explicitly: `pixmask.__version__` (semver string) and `pixmask.__api_version__ = 1` (integer, incremented on breaking changes).

---

### 9. Anti-Patterns to Actively Prevent

1. **Return `None` on error** — always raise. Silent failures in a security library are catastrophic.
2. **Mutate the input array** — always return a new array. Users will pass the same array to multiple calls.
3. **Write temp files to disk during processing** — all operations in memory. Disk I/O only when the user explicitly calls `sanitize_file`.
4. **Require format specification** — auto-detect from magic bytes, not file extension (extensions are attacker-controlled and lie).
5. **Add a `verbose` parameter** — use `logging.getLogger("pixmask")` instead. Users control verbosity through their existing logging configuration.
6. **Use `print` anywhere in the library** — ever, under any circumstances.
7. **Import heavy optional dependencies at module load time** — PIL, torch, langchain are optional. Import inside functions that use them, guarded by `try/except ImportError` with a clear actionable error: `"PIL is required for PIL output: pip install Pillow"`.
8. **Per-image unconditional security warnings** — the Security Architect proposes a `TYPOGRAPHIC_ATTACK_NOT_DEFENDED` warning on every `sanitize` call. This will be immediately suppressed by all production users, which is worse than silence. The correct place is README, docs, and a one-time `warnings.warn` on first import — not per-call noise.
9. **Ship a CLI as part of v0.1.0** — focus the Python API first. A CLI is a thin wrapper that can come in v0.2.0 once the API is stable.

---

### 10. Disagreements with the Security Architect

I agree on: hard-rejecting GIF/TIFF/SVG, vendoring parsers, never trusting file extensions, all hard limits on dimensions and file size, CSPRNG for QF randomization, exposing stego and perturbation scores in `SanitizeResult`.

I disagree on two points with concrete reasons:

**Default preset should be `balanced`, not `paranoid`:**

The Security Architect's argument is that 17ms extra latency is cheap compared to the cost of a jailbreak. But the real failure mode is not "user chose `balanced` and got jailbroken" — it is "user saw 25ms latency in benchmarks, disabled pixmask entirely, and got jailbroken." The library only provides security value if it gets used. A default that surprises users with unexplained latency causes abandonment. `balanced` is the right default; `paranoid` must be opt-in with clear documentation of what it adds.

**Per-call unconditional `SecurityWarning` for typographic attacks:**

A warning that fires on every single sanitize call will be suppressed in every production deployment within the first week. Suppressed warnings provide no information. A one-time warning on import + prominent documentation is more effective than a per-call warning that trains users to silence all pixmask warnings.

---

### Summary Position

The API I am proposing has one rule: users should never need to understand the internals to use the tool correctly. The preset system covers 95% of use cases. The `Pipeline` class covers the remaining 5%. SIMD dispatch, memory layout, parser selection — these are implementation details the library owns, not the user.

The biggest adoption risk is not security coverage or performance. It is that someone does `pip install pixmask`, hits a confusing API or a cryptic error message on their first image, and abandons the library. The type-flexible input, the sensible defaults, the `output_format` shortcut, the actionable error messages — these are not polish. They are the product.



---

## Minimalist / Scope Guard

**Position: SHIP FAST, SCOPE SMALL. v0.1.0 is not the final product — it is the proof of concept that earns the right to build v0.2.**

---

### 1. Ruthless MVP Scoping

The core value proposition of pixmask is exactly one sentence: **strip adversarial perturbations from images before they reach a VLM, with zero external system dependencies and <10ms latency.** Every feature decision must be evaluated against that sentence. If it doesn't directly serve it, it doesn't ship in v0.1.

The research correctly identifies six attack classes. Reframe them by implementation cost:

| Attack Class | Defense Complexity | v0.1 verdict |
|---|---|---|
| Gradient perturbations (PGD/C&W) | Low — bit-depth + median + JPEG | IN |
| Steganographic injection (LSB) | Low — bit-depth + JPEG destroys it | IN |
| Malformed images (parser exploits) | Low — validate + reject bad formats | IN |
| Scaling attacks | Low — resize with INTER_AREA | IN |
| Composite attacks (HADES) | Partially covered by above | PARTIAL, acceptable |
| Typographic injection (FigStep) | High — needs OCR + classifier | OUT until v0.3 |

The first four attack classes are covered by a pipeline that fits in under 2000 lines of C++. Ship that. The fifth is partially covered for free. The sixth requires a fundamentally different subsystem. Do not let FigStep drive v0.1 scope.

#### Phased Roadmap

**v0.1.0 — "It works, it's fast, it's safe to install"**
- 6-stage pipeline: Validate → Decode → Strip metadata → Bit-depth reduce → Median 3x3 → JPEG Q=random(70,85) → Output
- Supported formats: JPEG and PNG only
- Python bindings via nanobind
- `pip install pixmask` works on Linux x86_64, macOS arm64
- README with threat model, benchmarks, and honest limitations
- Zero external C dependencies beyond two vendored single-header files

**v0.2.0 — "Harder to break"**
- Pixel deflection (stochastic, non-differentiable)
- WebP format support (libwebp vendored via FetchContent)
- Swap stb_image for libspng + libjpeg-turbo (vendored, pinned, checksummed)
- INTER_AREA resize with jitter (±5%)
- Naive bilateral filter (small kernel only, documented performance caveat)
- Preset profiles: `fast`, `balanced`, `paranoid`
- LSB chi-square stego detection signal in result struct
- Windows wheels via cibuildwheel

**v0.3.0 — "Handles the hard stuff"**
- Wavelet denoising (Haar DWT + BayesShrink, inline ~100 lines)
- TV denoising (Chambolle-Pock)
- SIMD dispatch via Google Highway
- Optional OCR-based typographic attack detection (heavy optional dep, disabled by default)
- Adaptive attack evaluation harness (BPDA + EOT)

**v0.4.0+ — "Enterprise and research features"**
- Image quilting (needs patch corpus — out of scope before this)
- Feature squeezing detection (requires model forward pass — a different product)
- Neural steganography detection
- Streaming / partial-image processing

---

### 2. Dependency Minimization

The competitive advantage over OpenCV is **zero system deps, small binary, pip install just works**. Every external dependency is a tax on that promise.

#### Can we ship v0.1 with zero external C deps?

Close, but not quite — and that's fine. The goal is zero *system* deps, not zero *vendored* deps.

- **stb_image**: Single-header, no build system, trivially vendored. The known CVEs are for GIF and obscure formats. Restrict to JPEG and PNG only at compile time: `STBI_NO_GIF`, `STBI_NO_BMP`, `STBI_NO_TGA`, `STBI_NO_PSD`, `STBI_NO_HDR`, `STBI_NO_PIC`, `STBI_NO_PNM`. The JPEG and PNG paths have no open CVEs as of 2025.
- **stb_image_write**: Single-header, handles JPEG and PNG output. Vendored trivially.

**v0.1 has two vendored headers and zero link-time external dependencies.** Total vendor footprint: ~7000 lines of C, upstream-fuzzed.

#### Why not libspng + libjpeg-turbo in v0.1?

They are the right answer for v0.2. For v0.1:

- libspng requires linking — a CMake `find_package` or git submodule. Either breaks `pip install` on systems without dev headers.
- libjpeg-turbo's SIMD path requires nasm. More moving parts, more CI matrix entries.
- stb_image on JPEG/PNG-only (all other formats disabled at compile time) has an acceptable, auditable, documented security surface.

Upgrade path: v0.2 swaps stb_image for libspng + libjpeg-turbo via CMake FetchContent with pinned SHA-256 checksums. The C++ and Python APIs do not change.

#### Do we need FFTW? No. Never.

Haar DWT is 10 lines:

```cpp
for (int i = 0; i < n; i += 2) {
    float a = buf[i], b = buf[i+1];
    buf[i]   = (a + b) * 0.5f;
    buf[i+1] = (a - b) * 0.5f;
}
```

Apply row-then-column, threshold detail coefficients, inverse. That is wavelet denoising. No FFTW dependency, no GPL contamination (the Security Architect correctly flagged this), no nasm requirement, ~100 lines total when we get to v0.3.

---

### 3. What to CUT from v0.1

#### OCR detection — CUT, target v0.3+

FigStep has 82.5% ASR on open VLMs. Real threat. But OCR detection requires Tesseract (100MB+ system dep, GPL) or a neural model (weights + inference runtime). Neither belongs in a C++ preprocessing layer. Document this as an explicit non-goal in SECURITY.md. Users who need typographic attack defense compose pixmask with a separate OCR safety classifier.

The Security Architect's `TYPOGRAPHIC_ATTACK_NOT_DEFENDED` flag in the result struct sounds useful but fires unconditionally on every call — because we never defend against typographic attacks in v0.1. A warning that fires 100% of the time is not a warning; it is a constant. It belongs in SECURITY.md, not in the hot path result struct.

#### TV denoising (Chambolle-Pock) — CUT, target v0.3

Iterative algorithm (30-100 iterations), careful parameter tuning required, slow naive implementation. Marginal benefit over median + JPEG for the v0.1 threat model does not justify the implementation cost.

#### Bilateral filter — CUT from v0.1

Naive bilateral at σ_s=5 on a 1080p image misses the <10ms target. Fast bilateral grid is genuinely complex (3D histogram, trilinear interpolation). Median 3x3 is the v0.1 spatial defense — fast, cache-friendly, effective. Bilateral goes into v0.2 with small-kernel-only restriction.

#### Image quilting — CUT indefinitely

Requires a patch corpus, patch matching algorithm, and seam-finding. Texture synthesis, not sanitization. Not relevant to the threat model at any version boundary we can see.

#### Wavelet denoising — CUT from v0.1, target v0.3

JPEG Q=random(70,85) covers ~80% of wavelet denoising's benefit at 0% of its implementation cost. In v0.1, JPEG is the frequency-domain defense. Wavelet is an additional layer for the `paranoid` preset in v0.3, justified by real usage data.

#### Feature squeezing detection — CUT, permanently out of scope for the core library

Feature squeezing (Xu et al. NDSS 2018) uses model prediction disagreement between original and squeezed representations. It requires a model forward pass. pixmask is not a model. The Security Architect's SSIM-delta framing is not a substitute: JPEG compression at Q=75 on a clean natural image routinely produces SSIM 0.82-0.88. Our own sanitization pipeline would flag clean images as adversarially perturbed. False positive rate makes this useless.

#### Pixel deflection — CUT from v0.1, target v0.2

Good defense, ~30 lines. Cut because stochastic output makes test assertions statistical. `assert output == expected` stops working. The v0.1 test suite needs deterministic I/O. Ship deterministic v0.1, add stochastic operations in v0.2 when the test infrastructure can handle them.

#### Stego chi-square detection — CUT from v0.1, target v0.2

Good idea for v0.2. In v0.1 it adds API surface we must maintain and semantics we must define precisely (what threshold? what false positive rate on compressed images?). The pipeline destroys stego regardless of whether we detect it. Save this for v0.2 when we have a test corpus to tune against.

---

### 4. The Simple v0.1 Pipeline

```
Input bytes
  └── Stage 0: VALIDATE
        - Magic bytes: JPEG (FF D8 FF) or PNG (89 50 4E 47 0D 0A 1A 0A) only
        - File size: reject > 20MB
        - Decoded dimensions: reject > 8192x8192
        - Decompression ratio: reject > 50x (zip bomb guard)
        - On failure: return error, zero allocation, no decode attempted

  └── Stage 1: DECODE
        - stb_image with STBI_NO_GIF + STBI_NO_BMP + all non-JPEG/PNG formats disabled
        - Decode to uint8 RGB interleaved
        - Single contiguous allocation: width * height * 3 bytes

  └── Stage 2: STRIP METADATA
        - stb_image discards metadata on decode — no-op in v0.1
        - Named stage for API completeness and v0.2 upgrade path

  └── Stage 3: BIT-DEPTH REDUCTION
        - 8-bit channels to 5 effective bits: pixel = (pixel >> 3) << 3
        - In-place, single pass, O(n)
        - Collapses adversarial perturbations relying on low-bit precision
        - Destroys LSB steganography as a free side effect

  └── Stage 4: MEDIAN FILTER 3x3
        - Sorting-network-based 3x3 median (9 elements, 19 compare-swaps)
        - Per-channel, single row-buffer ping-pong
        - Smooths adversarial pixel patterns, preserves edges better than Gaussian

  └── Stage 5: JPEG RE-ENCODE at Q=random(70,85)
        - Quality factor from uniform random [70, 85] per call
        - Seed from OS entropy: getrandom() / getentropy() / BCryptGenRandom() — not rand()
        - Destroys DCT-domain steganography and high-frequency adversarial noise
        - stb_image_write JPEG output

  └── Stage 6: OUTPUT
        - Return sanitized bytes as JPEG (default) or PNG
        - Python: bytes object or numpy array (uint8, HWC)
```

**What this defeats:**

| Attack Class | Stages | Expected Post-pixmask ASR |
|---|---|---|
| PGD/C&W gradient perturbations | 3 + 4 + 5 | 5-15% (non-adaptive) |
| LSB steganography | 3 + 5 | ~0% |
| DCT-domain steganography | 5 | ~0% |
| Parser exploits / malformed images | 0 + 1 | Hard reject |
| Zip bombs / decompression bombs | 0 | Hard reject |

**What this honestly does NOT defeat (documented in SECURITY.md):**
- Typographic injection (FigStep): requires OCR
- Fully adaptive white-box attacks: no preprocessing defends against these
- Semantic harmful content: requires content moderation, not sanitization

---

### 5. Directory Structure

```
pixmask/
├── CMakeLists.txt              # single top-level build, no nested CMake
├── pyproject.toml              # scikit-build-core + nanobind + cibuildwheel config
├── README.md
├── SECURITY.md                 # threat model, explicit non-goals, CVE process
├── LICENSE
│
├── include/
│   └── pixmask/
│       └── pixmask.h           # single public header, entire public API surface
│
├── src/
│   ├── pixmask.cpp             # pipeline orchestration (~200 lines)
│   ├── validate.cpp            # stage 0 (~100 lines)
│   ├── decode.cpp              # stage 1: stb_image wrapper (~80 lines)
│   ├── bitdepth.cpp            # stage 3 (~50 lines)
│   ├── median.cpp              # stage 4: sorting-network 3x3 (~150 lines)
│   ├── jpeg_round.cpp          # stage 5: stb JPEG encode + decode (~100 lines)
│   └── bindings.cpp            # nanobind Python bindings (~150 lines)
│
├── vendor/
│   ├── stb_image.h             # pinned commit SHA documented in CMakeLists.txt
│   └── stb_image_write.h       # pinned commit SHA documented in CMakeLists.txt
│
└── tests/
    ├── test_validate.cpp        # doctest (single-header, vendored)
    ├── test_bitdepth.cpp
    ├── test_median.cpp
    ├── test_pipeline.cpp        # end-to-end with known test images
    └── python/
        └── test_bindings.py    # pytest
```

**What this structure deliberately excludes:**
- No `bench/` in v0.1. Benchmarks come after correctness.
- No `examples/` directory. The README has a 5-line usage example. That is enough.
- No `cmake/` subdirectory with Find modules. We vendor everything; there is nothing to find.
- No `scripts/` directory. CI lives in `.github/workflows/`.
- No `include/pixmask/detail/` for internal headers. Internal headers live in `src/`. Public API lives in `include/`.

**The entire public C++ API surface for v0.1:**

```cpp
namespace pixmask {

struct Options {
    int jpeg_quality_min = 70;
    int jpeg_quality_max = 85;
    int bit_depth = 5;           // effective bits per channel, 1-8
};

enum class Format { JPEG, PNG };

struct Result {
    std::vector<uint8_t> data;
    Format format;
    int width;
    int height;
};

// Throws PixmaskError on validation failure or decode error.
Result sanitize(const uint8_t* input, size_t len,
                Format output_format = Format::JPEG,
                const Options& opts = {});

} // namespace pixmask
```

One function. One options struct. One result struct. Python binding: `pixmask.sanitize(data: bytes, **kwargs) -> bytes`. Done.

---

### 6. Direct Rebuttals

**To the Security Architect — "Default profile must be `paranoid`"**

The `paranoid` profile as described requires bilateral filtering, wavelet denoising, and TV denoising — none of which exist in v0.1. There is no `paranoid` profile to default to. In v0.1 there is one pipeline. The profile debate is v0.2 scope entirely.

**To the Security Architect — "stb_image is unacceptable, use libspng"**

The concern is valid for the general stb_image case. It is overstated for the compile-time-restricted case. With every format except JPEG and PNG disabled and our own Stage 0 magic-byte gate running before stb_image is ever called, the remaining attack surface is narrow and auditable. We ship with stb, document the restriction and the v0.2 upgrade path in SECURITY.md, and swap to libspng on a known timeline. "libspng or nothing" blocks shipping. That is the wrong tradeoff in week one.

**To the Security Architect — "`TYPOGRAPHIC_ATTACK_NOT_DEFENDED` flag in every result"**

A flag that fires unconditionally on every call is not a signal — it is a constant. Callers will filter it out in the first week. It adds noise, forces every caller to handle it, and documents a non-goal in the worst possible location. SECURITY.md is the right place for this.

**To the Security Architect — "SSIM delta as adversarial perturbation score"**

JPEG compression at Q=75 on a clean natural image routinely yields SSIM 0.82-0.88. Our pipeline intentionally applies JPEG re-encoding. We would flag clean images as adversarially perturbed. This has a false positive rate that makes the signal useless in practice. Feature squeezing uses model prediction disagreement, not pixel SSIM, for exactly this reason.

**To the API/UX Designer — "Pipeline builder class in v0.1"**

The `Pipeline` builder class is a good v0.2 API. In v0.1 it is premature abstraction. There is one pipeline. A builder for one pipeline is ceremony. Ship the `sanitize()` function. The builder follows naturally in v0.2 when we have preset profiles to compose.

**To the API/UX Designer — "Numpy array and PIL Image as input types"**

The Python binding should accept `bytes` in v0.1. Full stop. Numpy and PIL are convenient peer dependencies; they are not required for the use case. Adding type dispatch in v0.1 means we own the behavior of `sanitize(np.array(...))` forever. Let users do `pixmask.sanitize(img.tobytes())` in v0.1. Add the convenience overloads in v0.2 after we have seen how people actually use the library.

---

### Summary

v0.1.0 is: validate, decode, bit-depth, median, JPEG re-encode, output. Two vendored headers. One public function. One options struct. One test suite. CI targeting Linux x86_64 and macOS arm64.

Every other feature discussed in this debate is correct work for some future version. None of it justifies blocking v0.1.

The first-mover gap is real (FINDINGS.md: "No VLM image sanitization tool exists on PyPI"). That gap closes the moment someone else ships a working library. Shipping the theoretically optimal library in twelve months is less valuable than shipping a solid, documented, correctly-scoped library in three weeks.

The minimal pipeline above reduces non-adaptive adversarial ASR from >90% to 5-15%. That is the core claim. Ship it. Iterate from a position of having users.

---

## Performance Engineer

> Thesis: pixmask's only defensible competitive moat is speed. IBM ART takes 50-500ms. DiffPure takes seconds. We win by being 10-100x faster for equivalent security coverage. Every architectural decision gets evaluated through one lens: **what is the security-per-microsecond ratio?**

---

### 1. MVP Feature Set: Security-Per-Microsecond Analysis

The research establishes six attack classes. Only four are stoppable by pixel preprocessing. Of those, two dominate the real threat model: gradient perturbations (PGD/C&W, >95% ASR undefended) and steganographic injection (24-31% cross-model ASR). This narrows the v0.1.0 mandate.

**Latency budget (hard constraints):**

| Profile | Budget | Rationale |
|---|---|---|
| `fast` | **<5ms** for 1080p (2MP) | LLM inference is 200-2000ms; preprocessing must be imperceptible overhead |
| `balanced` | **<15ms** for 1080p | Batch serving at 60+ img/s; this is the default for 90% of deployments |
| `paranoid` | <50ms | High-security only; acceptable for async pipelines |

These are not aspirational. At AVX2 throughput, bit-depth reduction on 2MP takes ~0.4ms (based on the benchmarked 0.42ms for 1920x1080 sorting network). The budget is achievable with correct implementation. Exceed it and we lose the competitive argument entirely.

**Security-per-microsecond ranking of transforms:**

```
Transform          | Est. latency (1080p) | Security coverage                           | Ratio
-------------------|----------------------|---------------------------------------------|-------
Bit-depth (5-bit)  | ~0.4ms               | Destroys gradient perturbations, LSB stego  | BEST
3x3 median (SIMD)  | ~0.4ms               | L0 attacks (CW0, JSMA), pixel deflection    | EXCELLENT
3-pass box blur    | ~1-2ms               | Kills L2/C&W, smooths HF perturbations      | EXCELLENT
JPEG encode/decode | ~5-10ms              | DCT-domain stego, frequency attacks         | GOOD
INTER_AREA resize  | ~1ms                 | Scaling attacks, incidental JPEG artifact   | GOOD
Bilateral filter   | ~8-20ms (naive)      | Edge-preserving, good for VLM quality       | MARGINAL
Pixel deflection   | ~3-5ms               | Stochastic, non-differentiable              | MARGINAL
Wavelet denoise    | ~15-40ms             | Strongest standalone, but complex           | POOR for v0.1
NLM                | >200ms               | Best quality, unacceptable cost             | REJECT
TV minimization    | >50ms (Chambolle)    | Good, but iterative                         | REJECT
```

**v0.1.0 `fast` profile (target: <5ms total for 1080p):**
- Stage 0: Validate (magic bytes + dim bounds) - <0.1ms, near-free
- Stage 1: Bit-depth reduction to 5 bits - ~0.4ms, AVX2 AND-mask, single pass
- Stage 2: 3x3 median filter - ~0.4ms, SIMD sorting network, 32px/cycle
- Stage 3: 3-pass box blur sigma=1.0 - ~1.5ms, O(1)/pixel sliding accumulator
- Stage 4: JPEG transcode QF=random(70,85) - ~2-3ms (libjpeg-turbo hardware path)
- **Total: ~4-5ms. Achievable.**

**v0.1.0 `balanced` profile (target: <15ms total for 1080p):**
- All `fast` stages plus:
- Stage 5: INTER_AREA resize with +-5% jitter - ~1ms
- Stage 6: Bilateral filter (naive + range-LUT, sigma_s=3, sigma_r=25) - ~5-8ms at r=9
- **Total: ~10-14ms. Achievable.**

**`paranoid` deferred to v0.2.** Wavelet denoise, pixel deflection, and TV minimization each need >15ms individually. Ship them later with proper tuning, not as an afterthought that blows the latency budget.

#### Direct rebuttal to Security Architect on default profile

The Security Architect argues the default must be `paranoid`. This is wrong for a reason beyond latency: **the library that never gets deployed provides zero security**. If pixmask's default profile causes production deployments to fail on p99 latency SLAs, operators skip it entirely. We end up with worse security in the ecosystem than if we had shipped `balanced` as default with clear documentation.

The Security Architect's acknowledgment mechanism has merit. The concrete position: `balanced` as default. If the caller explicitly sets `profile="fast"`, emit `warnings.warn` with a SecurityWarning. If the caller sets `profile="paranoid"`, no warning is needed. The Security Architect gets the acknowledgment mechanism; the default is deployable.

The Security Architect also proposed a per-call unconditional `SecurityWarning` for typographic attacks. Per-call warnings that fire on every invocation are suppressed in production within the first week. Suppressed warnings are noise. A one-time import warning plus documentation is strictly more effective.

---

### 2. C++ API Design

#### 2.1 `ImageView` — The Foundational Type

No ownership, no allocation, just a descriptor:

```cpp
namespace pixmask {

struct ImageView {
    uint8_t*  data;      // pointer to first pixel, never null
    int32_t   width;     // pixels
    int32_t   height;    // pixels
    int32_t   channels;  // 1=gray, 3=RGB, 4=RGBA
    int32_t   stride;    // bytes per row (>= width * channels, enforced 64-byte aligned)

    size_t   byte_size()  const noexcept { return stride * height; }
    uint8_t* row(int y)   const noexcept { return data + y * stride; }
};

} // namespace pixmask
```

Critical properties:
- `stride` is mandatory, not derived from `width * channels`. Callers pass numpy arrays with arbitrary strides; deriving stride would require a copy on every call from a non-contiguous array.
- No virtual functions, no inheritance, trivially copyable. On x86-64 this struct fits in 4 registers and passes entirely in-register.
- Const-qualifying the pointer in the call site expresses read-only intent without a separate `ConstImageView` type.

#### 2.2 Pipeline: Runtime `std::variant` Vector, NOT Compile-Time Templates

The compile-time template chain (expression templates, pipeline-as-type) argument has one fatal problem: the preset profiles must be runtime-selectable from Python. A template chain selected by a Python string at import time requires either: (a) an explicit dispatch on every supported combination at the Python layer, or (b) type erasure — which is exactly what `std::variant` gives us, but with more complexity.

Additional arguments against compile-time chains:
- Monomorphization bloat: every unique stage sequence generates a new instantiation. With 7 potential stages, you risk 2^7 combinations in the binary.
- Any template chain change that adds a stage changes the pipeline type, breaking stored instances in Python.
- Template metaprogramming errors in user-facing library code are unjustifiable maintenance overhead.

`std::visit` on a `std::variant` with 6 types compiles to a 6-entry jump table — effectively zero overhead versus a virtual call, with no heap allocation, no type erasure overhead beyond the jump, and guaranteed inlining of small stage dispatchers by any modern compiler.

```cpp
namespace pixmask {

struct BitDepthParams  { int bits = 5; };
struct MedianParams    { int radius = 1; };
struct BoxBlurParams   { float sigma = 1.0f; };
struct JpegParams      { int qf_lo = 70; int qf_hi = 85; };
struct BilateralParams { float sigma_s = 3.0f; float sigma_r = 25.0f; };
struct ResizeParams    { int target_w = -1; int target_h = -1; float jitter = 0.05f; };

using StageVariant = std::variant<
    BitDepthParams, MedianParams, BoxBlurParams,
    JpegParams, BilateralParams, ResizeParams
>;

class Pipeline {
public:
    explicit Pipeline(size_t arena_bytes = 32 * 1024 * 1024);
    Pipeline(const Pipeline&) = delete;
    Pipeline(Pipeline&&)      = default;

    Pipeline& add(StageVariant stage);
    void run(const ImageView& src, ImageView& dst) const;

    static Pipeline fast();
    static Pipeline balanced();
    static Pipeline paranoid();  // v0.2; stub that throws NotImplemented in v0.1

private:
    std::vector<StageVariant> stages_;
    Arena scratch_;
};

} // namespace pixmask
```

The `stages_` vector is built once at construction. `run()` iterates it with zero allocation. `scratch_` is the pre-allocated arena (see section 3).

#### 2.3 SIMD Dispatch: Highway `HWY_EXPORT` Pattern

No alternatives considered. Highway is the only option that:
1. Delivers zero-cost dispatch after the first call — CPU detection amortized to one-time cost, subsequent calls are direct jumps
2. Handles SSE2 through AVX-512BW, NEON, SVE, RVV in one codebase without ifdefs
3. Is production-proven in libjxl, AV1 (libaom), Chromium, and libvips

The dispatch pattern, one `.h`/`.cc` pair per transform:

```
src/cpp/ops/
    bit_depth.h       # HWY_BEFORE_NAMESPACE() + template implementation
    bit_depth.cc      # HWY_TARGET_INCLUDE "bit_depth.h" + HWY_EXPORT
    median.h
    median.cc
    box_blur.h
    box_blur.cc
    bilateral.h
    bilateral.cc
    jpeg_transcode.h
    jpeg_transcode.cc
```

`foreach_target.h` recompiles the `.h` body for every enabled ISA target in one `.cc`. The linker retains all variants. `HWY_DYNAMIC_DISPATCH` calls the best at runtime.

Hard constraint from Highway's design: **each `.cc` must include exactly one `.h` via `HWY_TARGET_INCLUDE`**. This is an ODR requirement. Violate it and you get silent miscompilation. This constraint is the forcing function for the file layout above.

---

### 3. Memory Architecture

#### 3.1 Arena Allocator: Bump Pointer, No Free Path

The entire `run()` hot path is zero-allocation. Every transform that needs scratch space gets memory from the arena:

```cpp
class Arena {
public:
    explicit Arena(size_t capacity);
    void*  alloc(size_t n, size_t align = 64) noexcept;  // bump pointer, O(1)
    void   reset() noexcept;                              // pos_ = 0, no free

private:
    std::unique_ptr<uint8_t[], AlignedDeleter> buf_;
    size_t cap_, pos_;
};
```

The bump pointer is the only correct design. No free-list, no slab, no pool. The transforms run sequentially. Scratch buffers from stage N are dead when stage N+1 starts. `reset()` between images reuses the entire arena. Fragmentation is structurally impossible.

**Arena sizing:** Worst-case scratch is bilateral filter with float working buffer: `width * height * channels * sizeof(float)`. For 2MP RGB: 24MB. Pre-allocate 32MB per `Pipeline` instance. One `malloc` at construction, zero during `run()`.

#### 3.2 Thread Safety Model

`Pipeline` is non-copyable, move-only. Each thread owns its own `Pipeline` instance. The threading contract is enforced by the type, not by documentation.

This is the right tradeoff. A shared `Pipeline` with a mutex would serialize all requests through the mutex and destroy throughput. Per-thread `Pipeline` instances with 32MB arenas use 32MB * thread_count of memory — for a 16-thread server, that is 512MB. This is acceptable for a library that would otherwise require locking at every image boundary.

Python binding: the `Pipeline` Python wrapper must not be shared across threads. Document this. Enforce it in debug builds via a thread-id assertion in `run()`.

#### 3.3 Zero-Copy NumPy Interop via nanobind

The Python binding entry point avoids all copies on input:

```cpp
#include <nanobind/ndarray.h>
namespace nb = nanobind;

using NpArray = nb::ndarray<nb::numpy, uint8_t, nb::shape<-1,-1,-1>>;

NpArray sanitize(Pipeline& pipeline, NpArray image) {
    nb::gil_scoped_release release;  // GIL released for entire C++ execution

    ImageView src {
        .data     = image.data(),
        .width    = static_cast<int32_t>(image.shape(1)),
        .height   = static_cast<int32_t>(image.shape(0)),
        .channels = static_cast<int32_t>(image.shape(2)),
        .stride   = static_cast<int32_t>(image.stride(0)),  // bytes per row
    };
    // ... allocate dst from arena, run pipeline, return nb::ndarray view of dst
}
```

For a writable numpy input, modify in-place and return the same array — eliminates one full 2MP copy for `fast` mode. For read-only input, allocate output from the arena and return a numpy array that borrows from it. Caller must not call `run()` again before consuming the output (document this; the arena reset invalidates prior outputs).

The `nb::gil_scoped_release` means Python threads are unblocked during the entire sanitize call. This is mandatory for throughput: a GIL-held sanitize call would serialize all Python threads regardless of how many CPU cores are available.

---

### 4. Algorithm Choices

#### 4.1 Median Filter: Sorting Network (3x3) vs CTMF

The research benchmark data is decisive:
- **3x3 SIMD sorting network on AVX2**: 0.42ms for 1920x1080 uint8. 58x over scalar. Processes 32 pixels per `_mm256_min_epu8`/`_mm256_max_epu8` cycle. 19 CAS steps (Bose-Nelson network for 9-element median extraction, zero branches).
- **CTMF at radius=1**: histogram setup and two-tier coarse/fine scan adds constant overhead. Estimated ~2ms for 2MP. 5x slower than the sorting network at the one radius that actually matters for defense.
- **CTMF crossover**: wins at radius>=5 because its O(1)/pixel constant (32-entry coarse/fine scan vs O(k^2) network growth) becomes favorable.

**v0.1.0 ships the SIMD sorting network only, for radius=1.** The research from Feature Squeezing (Xu et al., NDSS 2018) is explicit: 3x3 is the sweet spot for L0 attacks. Larger kernels (5x5+) crush fine details that VLMs depend on for comprehension. CTMF ships in v0.2 for paranoid mode if a use case for radius>=5 emerges.

**Implementation detail:** Use the 19-step Bose-Nelson network for median extraction, not 25 steps. The 25-step count in the research document describes a full sort of 9 elements; median extraction only needs to guarantee element [4] is the median, which takes 19 CAS. OpenCV's `median_blur.simd.hpp` confirms 19 steps. The AVX2 pattern processes 32 pixels simultaneously via the `SORT2` macro operating on 256-bit registers.

#### 4.2 Gaussian Blur: 3-Pass Box Blur as Default

**3-pass box blur (Kutskir / bfraboni)** is the only defensible choice for v0.1.0:
- O(1) per pixel via sliding accumulator, O(n) total regardless of sigma
- sigma=0.5 and sigma=20 take identical wall-clock time — critical for deterministic latency
- 0.04% average pixel error vs true Gaussian — negligible for adversarial defense use
- Zero external dependency, ~80 lines of C++ including the cache-optimal tiled transpose

**Van Vliet IIR Gaussian** is the right choice for sigma>6, but that is a v0.2 concern. The recommended sigma=1.0 for `fast` and sigma=1.5 for `balanced` are well within box blur's valid range. The Triggs-Sdika boundary correction for Van Vliet adds ~100 lines of non-trivial numerical code and a 3x3 matrix inversion at startup. Not worth carrying in v0.1.

**Reject separable FIR convolution.** For sigma=1.0 the kernel is 5 elements: O(5*W*H) multiply-adds vs O(W*H) for box blur. The only advantage of FIR is exact Gaussian coefficients, which matter only if the downstream security analysis requires exact frequency domain behavior. It does not. Box blur is faster, simpler, and correct enough.

#### 4.3 Wavelet Denoise: Inline Haar, No FFTW, Deferred to v0.2

The Security Architect already issued the GPL veto on FFTW. The performance argument reinforces it: FFTW is designed for arbitrary-length FFTs with a runtime planning phase that introduces unpredictable latency spikes on first call. Haar wavelet decomposition is two multiply-adds per pixel per level — it does not need FFTW.

But the entire wavelet denoise stage is deferred to v0.2 for latency reasons. Even a simple 3-level Haar + BayesShrink on 2MP floating-point requires:
- 3-level decomposition: 3 passes over the image, shrinking by 2x each time
- Per-coefficient threshold estimation from sub-band noise variance
- 3-level reconstruction
- In-place transform to avoid double-buffering the full image

Estimated latency: 15-25ms for 2MP. This alone exceeds the `balanced` budget of 15ms. Wavelet denoise belongs in `paranoid` mode, which is a v0.2 feature.

#### 4.4 Bilateral Filter: Naive + Range LUT in v0.1.0

**For `balanced` profile**, the naive bilateral with two precomputed LUTs is the correct v0.1.0 choice:
- Spatial weights: k*k float LUT (k=19 for r=9, sigma_s=3) — computed once at pipeline construction
- Range weights: 256-entry float LUT indexed by `|I_p - I_q|` — computed once at pipeline construction
- No `expf` in the inner loop
- Estimated latency: 6-8ms for 2MP RGB at r=9

The bilateral grid (Paris-Durand) would be faster for large sigma values but requires a 3D grid allocation, trilinear interpolation, and a 3D separable Gaussian blur. That is 2+ weeks of correct implementation. Ship it in v0.2.

The Chaudhury O(1) trigonometric bilateral is theoretically elegant (decompose range kernel into K Gaussian convolutions) but requires floating-point throughout and K=4-5 Gaussian blur passes. At sigma_s=3 and sigma_r=25, K is not small enough to beat the precomputed-LUT naive approach for this kernel size. Reject for v0.1.

---

### 5. What to CUT from v0.1.0

**Cut absolutely — latency or complexity exceeds v0.1 constraints:**

| Feature | Reason |
|---|---|
| Non-Local Means | O(21609 * W * H). >200ms. Structurally incompatible with any latency budget. |
| TV minimization (Chambolle-Pock) | Iterative convergence loop, 50ms+. Add in v0.2 with early stopping and max-iter limit. |
| Wavelet denoise (Haar + BayesShrink) | 15-25ms for 2MP, exceeds `balanced` budget alone. v0.2 / `paranoid` only. |
| Bilateral grid (Paris-Durand) | 3D grid + trilinear interp, 2+ weeks implementation. v0.2. |
| Pixel deflection | Stochastic random-pixel lookup adds ~3-5ms; security gain is additive to bit-depth reduction which covers most of the same L_inf threat. Defer. |
| CTMF large-radius median | Only needed for radius>=5. v0.1 ships radius=1 only. |
| Van Vliet IIR Gaussian | Box blur covers sigma<6. All v0.1 profiles use sigma<=2. |
| Adaptive INTER_AREA resize | Scaling attacks matter, but this adds complexity to the libjpeg-turbo integration. Implement in v0.2 with proper sub-pixel jitter. |
| GIF/TIFF/SVG parsers | Attack surface. Never ship. Permanent reject. |
| OCR typographic defense | Wrong library boundary. Requires a downstream text classifier. Out of scope for this library permanently. |

**Cut from API surface (keep internal only):**

| Feature | Reason |
|---|---|
| Per-channel independent processing | Adversarial perturbations are spatially correlated across channels. Channel-independent median is slower with no additional security benefit. |
| Float32 pixel I/O | Bilateral uses float internally; that is an implementation detail. Input and output are always uint8. No float I/O API in v0.1. |
| General-radius SIMD sorting network | The 3x3 network is hand-tuned and verified. A general-radius SIMD sorting network for arbitrary k is a separate implementation project, not an MVP feature. |

**Reject as runtime dependencies (permanent):**

| Dependency | Reason |
|---|---|
| FFTW | GPL contamination. Unnecessary for Haar. Permanent veto. |
| OpenCV | 50MB binary, requires libGL. We are building the alternative. Test infrastructure only. |
| Eigen | No matrix operations needed for v0.1 transforms. |
| Boost | Never. |

---

### 6. Cache-Optimal Memory Access: Non-Negotiable Constraints

All algorithms must be validated against cache behavior before the PR merges:

**Tiling:** Process images in 64x128-pixel tiles (8KB for uint8 RGB, fits in 16KB L1 on all modern CPUs). The box blur transpose in particular must use block-wise tiled transpose. Naive row-swap transpose on 2MP incurs ~40% L2 miss rate on the column pass — measured empirically. A 32x32 tiled AVX2 transpose using `_mm256_unpacklo/hi_epi8` is 3x faster on 2MP.

**Row alignment:** All rows are 64-byte aligned (one cache line). The arena enforces this — `stride` is always rounded up to the next multiple of 64. Waste: at most 63 bytes per row. For 1080p: ~68KB waste against a 6MB image. Accepted.

**Load alignment in Highway:** Use `LoadU` (unaligned load) in all Highway ops. Aligned `Load` requires the pointer to be aligned at every call site, including mid-row positions — alignment faults in scalar tail handling are a correctness risk that outweighs the 10% load throughput gain. When profiling shows load bandwidth is the bottleneck (not compute), revisit aligned loads for row-start positions only.

**The transpose buffer:** Box blur requires a transposed intermediate. This is one arena allocation, reused across all three blur passes. The arena `reset()` between images returns this memory to the pool — no fragmentation, no re-allocation.

---

### 7. Benchmark SLA: Hard Release Gate

Published with every release. Measured on reproducible hardware: AWS c6i.xlarge (Intel Ice Lake, AVX-512) for x86-64, AWS m7g.xlarge (ARM Graviton3, NEON) for ARM.

```
pixmask bench --profile fast     --size 224  --iterations 10000
pixmask bench --profile fast     --size 1080 --iterations 1000
pixmask bench --profile balanced --size 1080 --iterations 1000
```

Reported metrics: p50, p95, p99 latency in microseconds; images/second.

**Hard SLAs:**
- `fast` at 1080p: p99 < 5ms
- `balanced` at 1080p: p99 < 15ms
- `fast` at 224px: p99 < 1ms

If any SLA is missed, the release does not ship until it is fixed. No exceptions. Any PR that regresses p99 by >10% requires a written performance analysis before merge.

---

### Summary Position

v0.1.0 ships **five pixel transforms** — bit-depth reduction, 3x3 SIMD sorting network median, 3-pass box blur, JPEG transcode, and naive bilateral with LUT — plus metadata strip and validation, delivering two profiles: `fast` (<5ms, 1080p) and `balanced` (<15ms, 1080p).

The implementation uses Highway for SIMD dispatch (zero-cost runtime CPU detection), an arena allocator (zero hot-path allocation), nanobind zero-copy numpy interop (GIL released for the full C++ call), and a runtime `std::variant` pipeline (no virtual dispatch, no heap per stage).

Everything else — wavelet denoise, bilateral grid, NLM, TV minimization, pixel deflection, CTMF, Van Vliet IIR — is explicitly deferred to v0.2 with a latency justification required at that time.

Agreement with Security Architect: libspng for PNG (non-negotiable), vendored dependencies with SHA-256 verification, CSPRNG for QF randomization (OS entropy, not mt19937), hard dimension/decomp limits, no GIF/TIFF/SVG ever.

Disagreement with Security Architect: `balanced` as the default profile, not `paranoid`. A sanitizer that gets deployed at `balanced` is infinitely more valuable than a sanitizer that gets disabled because `paranoid` blew the production latency budget.

Speed is what gets this library into production pipelines. Security coverage is what justifies keeping it there.

---

## Testing & Quality Architect

**Perspective: Correctness and Verification — nothing ships without tests.**

If it is not tested, it is broken. Every claim in FINDINGS.md — SSIM floors, latency targets, attack reduction rates — is a contract with users. The test suite is the mechanism that makes those contracts enforceable. An untested security library is not a library; it is a liability.

---

### 1. C++ Testing Framework: doctest

**Decision: doctest. Not Catch2. Not Google Test.**

**Against Google Test:**
GTest's registration machinery and `gtest-main` linkage add 30-60 seconds to incremental C++ builds on a test suite with 100+ cases. For a library where developers will run tests constantly during filter implementation, this matters. More fundamentally, GTest's value-adds — mocking framework, death tests, parameterized test classes with typed fixtures — are liabilities here, not assets. pixmask has no virtual dispatch hot paths to mock. We test pure functions on pixel buffers. GTest's mock framework is irrelevant and its additional macros create cognitive overhead.

**Against Catch2 v3:**
Catch2 v3 is a genuinely good framework, but it has moved to a multi-header structure requiring CMake integration via submodule or FetchContent. Its Matchers DSL and Generators add compile time and complexity we will not use. For a codebase where the test author iterates fast on image transform implementations, the extra compile overhead is a tax with no return.

**For doctest:**
- Single header: `tests/doctest.h`. Zero CMake wrangling. `#include "doctest.h"` and go.
- Compile time: doctest's registration is compile-time constexpr, adding near-zero overhead.
- Feature parity for our needs: REQUIRE/CHECK assertions, subcases (equivalent to Catch2 SECTION), test filtering by tag and name, minimal output format.
- Subcases are specifically important for image filter tests: run the same assertion body over a matrix of bit depths or kernel sizes without creating 30 separate TEST_CASE functions.

The only legitimate argument for GTest is IDE integration (CLion native support). That is not an architectural decision. doctest test output is parseable by all major CI systems.

---

### 2. Unit Test Coverage for v0.1

#### 2a. Bit-Depth Reduction

The operation is `pixel = (pixel >> (8-b)) << (8-b)`. This is two lines but has six distinct failure modes in SIMD implementations: off-by-one on shift count, sign extension when promoting 8-bit to 16-bit for AVX2, wrap-around at 255, wrong lane width in the vector shuffle, stride bug for non-contiguous rows, and incorrect handling of `bit_depth=8` (which must be identity, no shift).

Required test cases per bit depth (1 through 8):

```
bit_depth=1: {0→0, 127→0, 128→128, 255→128}
bit_depth=2: {0→0, 63→0, 64→64, 127→64, 128→128, 191→128, 255→192}
bit_depth=5: {0→0, 7→0, 8→8, 127→120, 128→128, 248→248, 255→248}
bit_depth=8: all 256 values must map to themselves (identity assertion)
```

Full edge-value matrix: for each bit depth b in [1,8], test pixel values `{0, step/2 - 1, step/2, step - 1, 127, 128, 254, 255}` where `step = 1 << (8-b)`. This is 64 assertions covering every quantization boundary.

SIMD correctness gate: the same inputs must produce byte-for-byte identical output from the scalar reference implementation and the SIMD-dispatched implementation. Byte-level identity is the contract. Any divergence is a bug. See Section 6c for how to force specific SIMD dispatch paths.

Memory layout tests: run with stride != width (padded rows), in-place mutation (src == dst pointer), and non-contiguous images. Stride bugs will not appear on contiguous square images.

#### 2b. Median Filter

The median filter has more implementation-specific failure modes than any other stage. The 3×3 sorting network (25 compare-and-swap operations) is brittle — a single wrong comparison index produces a wrong median. CTMF for 5×5 has a histogram management bug class where pixel contributions are double-counted at tile boundaries.

Required test cases:

- **Known-output patterns:**
  - Uniform image (all pixels identical value v): output must be v everywhere, including border pixels.
  - Impulse image (single hot pixel at center of zero-filled image): with 3×3 kernel, center pixel's output is 0 (median of 8 zeros and 1 nonzero = 0 for a center pixel's neighborhood where it is the sole nonzero).
  - Checkerboard 8×8 (alternating 0 and 255): analytically the median of any 3×3 neighborhood is either 0 or 255 depending on the center pixel's color. Verify this analytically-derived result.

- **Kernel size boundary:** the 3→5 switch is an implementation-specific threshold that must be tested explicitly at kernel_size=3 and kernel_size=5. Do not assume both branches are exercised by resolution tests alone.

- **Border handling modes:** test reflect, replicate, and zero-pad separately. For a 1×1 image with 3×3 kernel under replicate padding: the 3×3 neighborhood is 9 copies of the single pixel, so output must equal that pixel exactly. Assert this.

- **Sorting network exhaustion (3×3):** there are 3^9 = 19683 distinct combinations of {0, 127, 255}^9. Run all of them in a subcase loop. Compute reference median via `std::sort`. Assert the sorting network produces the same value as the reference. This runs in under 100ms and proves the sorting network is correct over its full input space.

- **Non-square images:** 1×N (single row), N×1 (single column), 7×3. These force short-row fallback scalar loops that differ from the SIMD main path.

#### 2c. JPEG Encode/Decode Roundtrip

JPEG tests assert properties, not exact values.

Required test cases:

- **Quality factor matrix:** q in {1, 10, 25, 50, 70, 75, 85, 95, 100}. For each: encode, decode, compute PSNR. Assert PSNR >= expected floor for that quality. The expected floor table is hardcoded in the test file with a comment citing the libjpeg-turbo source. This makes the expectation explicit and reviewable.

- **Random quality range validation:** the pipeline uses `random(70, 85)`. Run 1000 encode-decode cycles. Assert: all 1000 produce a valid decodable JPEG buffer; the quantization table embedded in the JFIF header encodes a quality factor within [70, 85]; no two consecutive QFs are identical (catches a broken RNG seeding producing constant output).

- **Color channel integrity:** encode a synthetic RGB image where R=255,G=0,B=0 in all pixels. Decode. Assert mean(R) > 200, mean(G) < 30, mean(B) < 30. This catches the JCS_RGB vs JCS_EXT_RGB channel swap bug, which is a documented libjpeg-turbo integration footgun.

- **Dimension preservation:** after encode/decode at all quality factors, width and height must be unchanged. Test explicitly with odd dimensions (e.g., 7×11). JPEG chroma subsampling rounds dimensions to multiples of 16 internally; the output must still match input dimensions.

- **Grayscale path:** test 1-channel (Y-only) input separately. libjpeg-turbo has a distinct code path for grayscale that RGB tests do not exercise.

- **Corrupt input rejection:** feed a 1024-byte buffer with a valid SOI marker (0xFF 0xD8) prefix followed by random bytes. Assert: decode returns error code, the output image struct is in a well-defined null/empty state, no heap corruption under ASan.

#### 2d. Input Validation

This is a security boundary. Tests must be exhaustive. Every invalid input must produce a clean error, never UB, never a crash.

Required test cases:

- **Dimension limits:** width=8193 (one over limit), height=1 → PIXMASK_ERR_DIMENSION. width=65536, height=65536 → same. Test both axes independently.
- **Zero dimensions:** width=0 (height=valid), width=valid (height=0), width=0 height=0. All three → error.
- **Negative dimensions (signed int API):** -1, INT_MIN → error before any decode attempt.
- **Null pointer:** null `data` with valid `size` → error. Non-null `data` with `size=0` → error.
- **Corrupt PNG header:** valid PNG magic (89 50 4E 47) with garbage remainder → decode error.
- **Corrupt JPEG header:** SOI marker (FF D8) with garbage remainder → decode error.
- **Polyglot files:** a buffer that passes PNG magic validation but is simultaneously a valid ZIP file. Assert the file is processed through the PNG decoder only and produces either a valid PNG output or a clean decode error — never a ZIP-extraction code path. This is the attack class noted explicitly in FINDINGS.md.
- **Decompression ratio bomb:** a valid PNG with 4×4 pixel canvas but an iCCP or zTXt chunk containing a 100MB zlib stream. The Stage 0 validation gate must reject this before any zlib decompression occurs. Assert: function returns within 5ms (fast rejection at header scan time); returns PIXMASK_ERR_DECOMP_RATIO.
- **File size limit:** buffer > 50MB → PIXMASK_ERR_TOO_LARGE before any parsing.

For each test: verify the specific error code is the documented one (not PIXMASK_ERR_UNKNOWN), verify no heap allocation was made after rejection (instrument with a custom allocator in test mode), verify no crash under ASan.

#### 2e. Metadata Stripping

Metadata stripping must be verifiable without depending on third-party metadata libraries in test code. Write a minimal IFD walker in `tests/helpers/exif_walker.hpp` (~40 lines) that scans JPEG APP1 Exif segments for the GPS IFD tag (0x8825). This is the only external verification needed for EXIF GPS removal.

Required test cases:

- **GPS EXIF removal:** load `exif_gps.jpg` fixture. Run strip. Parse output JPEG's APP1 data with the inline IFD walker. Assert: no GPS IFD (tag 0x8825 in IFD0) is present in the output.
- **Full APP1 removal:** if the strip policy is "remove all Exif", assert no APP1 segment marker (0xFF 0xE1) exists in the output JPEG buffer. This is a byte-search, not a semantic parse.
- **XMP removal from PNG:** load `xmp_metadata.png`. Run strip. Scan the output PNG for iTXt/tEXt chunks. Assert none contain the string `xmlns:xmp`.
- **Large ICC profile removal:** load `icc_large.png` (~400KB ICC). Run strip. Assert: output file size is at least 350KB smaller than input. Assert the output is a valid decodable PNG.
- **Embedded thumbnail removal:** the SubIFD at tag 0x0201 (JPEGInterchangeFormat) must be absent or zero-length in the output. Use the inline IFD walker.
- **No-metadata identity:** load `clean_224.png` (no EXIF, no XMP). Run strip. Assert output is byte-identical to input (for PNG with no metadata, the strip is a no-op).

---

### 3. Integration Tests

#### 3a. Full Pipeline Roundtrip with SSIM Gate

For each preset (`fast`, `balanced`, `paranoid`), run the full pipeline on all 8 natural-image fixtures and assert:

```
fast:      SSIM >= 0.90, PSNR >= 28 dB
balanced:  SSIM >= 0.85, PSNR >= 26 dB
paranoid:  SSIM >= 0.80, PSNR >= 24 dB
```

These values come directly from the FINDINGS.md preset table and are the public contract. If the implementation cannot meet them, the preset configuration is wrong, not the test.

SSIM implementation: do not use `cv::quality::QualitySSIM`. Write a standalone 11×11 window SSIM implementation in `tests/helpers/ssim.hpp` using the formula from `research/11_evaluation_methodology.md`. This is approximately 60 lines of C++ and eliminates `opencv_contrib` as a test dependency. The OpenCV Quality module requires `opencv_contrib`, which is frequently unavailable in CI base images and introduces a large optional dependency just for a test helper.

The 8 natural-image fixtures cover cases where SSIM fails non-obviously: near-black images (SSIM denominator approaches zero), near-white images, high-frequency patterns, and uniform images. Any implementation that only tests on "natural photos" will miss these corner cases.

PSNR is trivial inline: `10 * log10(65025.0 / mse)`. No library needed.

#### 3b. Known Adversarial Images — Perturbation Reduction Gate

Test that the sanitizer measurably reduces adversarial perturbation energy at the pixel level. This does not require running a VLM.

Assertion:

```
||sanitized(adv) - clean||_2  <=  reduction_factor * ||adv - clean||_2
```

Where `reduction_factor = 0.5` for `balanced` preset and `0.4` for `paranoid` preset.

Fixtures: 5 precomputed PGD-8/255 adversarial images and their paired clean originals, stored in `tests/fixtures/adversarial/`. Generated once offline using torchattacks against a ResNet-50 checkpoint; committed as binary PNG blobs. The generation script is in `scripts/gen_adversarial_fixtures.py` and documents the exact parameters (model, epsilon, steps, step size, target class) so the procedure is reproducible.

Also test with 5 precomputed C&W L2 adversarial images to cover the L2-norm attack class separately from the L-inf class.

This tests necessary conditions, not sufficient conditions. Full VLM-in-the-loop ASR testing belongs in the evaluation suite, not in CI.

#### 3c. Malformed Input: No Crash, Clean Error

Run the full pipeline on 8 malformed inputs under AddressSanitizer. Assert: no crash (SIGSEGV, SIGABRT, ASan finding), return code != PIXMASK_OK:

1. Truncated PNG: valid header and IHDR, data stream cut at byte 500.
2. Truncated JPEG: valid SOI+APP0+SOF0, cut before EOI.
3. Valid PNG structure with corrupted IDAT zlib data (single bit flip in compressed stream).
4. Zero-byte buffer.
5. 1-byte buffer containing 0xFF.
6. 1KB of random bytes from fixed seed (reproducible).
7. Valid PNG where IHDR claims 65535×65535 pixels but data stream is 1KB.
8. WebP file with RIFF chunk size set to 0xFFFFFFFF (the CVE-2023-4863 class of chunk size overflow).

All stored as binary fixtures in `tests/fixtures/malformed/`. Do not generate at runtime — deterministic fixtures make CI failures reproducible.

#### 3d. All Presets Produce Valid Output

Smoke test: for each preset and each supported format (PNG, JPEG, WebP), with a 224×224 clean RGB input:

1. Function returns PIXMASK_OK.
2. Output buffer decodes successfully (libspng/libjpeg-turbo returns success).
3. Output dimensions match configured output resolution.
4. All output pixel values are in [0, 255] (no overflow from intermediate float arithmetic).
5. Output buffer is non-null and size > 0.

This catches: preset misconfiguration, wrong stage ordering, buffer aliasing bugs where output overlaps input, and encoder bugs producing syntactically invalid PNG/JPEG.

---

### 4. Performance Tests

#### 4a. Google Benchmark Microbenchmarks (C++)

One `BM_` function per pipeline stage, parameterized over VLM-relevant resolutions:

```
224×224   (ImageNet/ViT-B)
336×336   (LLaVA-1.5)
448×448   (InternVL2 tile)
512×512
1024×1024
2048×2048  (document understanding)
```

These resolutions come from `research/11_evaluation_methodology.md` Section 7. They are not arbitrary — they match the actual VLM input sizes pixmask users will encounter.

Each benchmark must use `benchmark::DoNotOptimize` on the output buffer and `benchmark::ClobberMemory()`. Without these, the compiler elides the filter body under `-O2`. Reports `img/s` via `Counter::kIsRate` and bytes/s via `SetBytesProcessed`.

Stages to benchmark individually:

```
BM_BitDepthReduce_5bit
BM_MedianFilter_3x3
BM_MedianFilter_5x5
BM_BilateralFilter_s5_r15
BM_WaveletDenoise_Haar_BayesShrink
BM_JPEGEncodeDecode_q75
BM_JPEGEncodeDecode_random
BM_MetadataStrip
BM_FullPipeline_fast
BM_FullPipeline_balanced
BM_FullPipeline_paranoid
```

Run with `--benchmark_repetitions=100`. Do not report mean — it is skewed by cache miss outliers and scheduler preemption. Use `scripts/percentiles.py` (already prototyped in `research/11_evaluation_methodology.md` Section 7) to compute p50/p95/p99. This script must be committed in `scripts/` and invoked in CI after the benchmark binary runs.

Thread scaling: `->Threads(1)->Threads(2)->Threads(4)->Threads(8)` on `BM_FullPipeline_*` only. Individual stage benchmarks stay single-threaded.

Latency targets for CI gate (p99, single-threaded, 512×512 on `ubuntu-22.04` reference runner):

```
fast preset:      p99 < 8ms
balanced preset:  p99 < 15ms
paranoid preset:  p99 < 35ms
```

These are looser than the research estimates (~3ms, ~8ms, ~25ms) to accommodate CI runner variability (shared VMs have ±15-20% timing noise).

#### 4b. Python API Benchmarks (pytest-benchmark)

```python
# tests/bench/test_python_api.py
import pytest
import numpy as np
import pixmask

@pytest.fixture
def sample_rgb(request):
    res = request.param
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (res, res, 3), dtype=np.uint8)

@pytest.mark.parametrize("sample_rgb", [224, 512, 1024], indirect=True)
@pytest.mark.parametrize("preset", ["fast", "balanced", "paranoid"])
def test_sanitize_throughput(benchmark, sample_rgb, preset):
    result = benchmark(pixmask.sanitize, sample_rgb, preset=preset)
    assert result is not None
```

pytest-benchmark handles warmup, iteration count, and JSON output. The Python API should add less than 0.5ms overhead versus the equivalent C++ benchmark. If it adds more, the nanobind buffer protocol is copying the numpy array rather than zero-copying — this is a correctness issue with the binding, not just a performance issue.

Add a dedicated zero-copy test: verify the input numpy array's data pointer matches the `ImageView.data` pointer seen by the C++ layer. If they differ, a copy occurred. This test fails fast on binding regressions.

#### 4c. CI Regression Gate

Store a benchmark baseline in `tests/bench/baseline.json`. After each benchmark run in CI:

```bash
python3 scripts/check_regression.py \
    --baseline tests/bench/baseline.json \
    --current bench_output.json \
    --threshold 0.10
```

Script compares `real_time` per benchmark name and exits nonzero if any exceeds baseline by more than 10%. Threshold is deliberately loose — CI runners have ±5-8% timing variance from CPU frequency scaling. Tight thresholds produce false positives that train developers to ignore CI failures.

The baseline is updated **manually** only, via a commit with message `perf: update benchmark baseline`. Never updated automatically. Automatic updates create a ratchet where regressions are silently absorbed.

The regression gate runs only on `ubuntu-22.04` with `sudo cpupower frequency-set -g performance` (added as a CI step before benchmark execution). Running it on macOS introduces cross-platform timing variables that make the gate meaningless.

---

### 5. Security Tests

#### 5a. Fuzz Testing with libFuzzer

libFuzzer over AFL++ for this project:
1. No separate process. Fuzz targets link directly into a binary. ASan and UBSan run inline.
2. CMake integration: `target_link_libraries(fuzz_target -fsanitize=fuzzer)` — one line.
3. libFuzzer's in-process mutation engine is faster than AFL's fork-based model for the small input sizes typical here (valid image headers are 100-500 bytes before the fuzz target rejects them).

Required fuzz targets in `tests/fuzz/`:

```cpp
// fuzz_decode_png.cpp
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    pixmask::DecodeResult r = pixmask::decode_png(data, size);
    (void)r;
    return 0;
}
```

Same pattern for `fuzz_decode_jpeg.cpp`, `fuzz_decode_webp.cpp`, and `fuzz_sanitize.cpp` (full pipeline). The full-pipeline fuzzer is the most valuable: it exercises composition of all stages, not just individual decoders.

Execution schedule:
1. **Weekly scheduled job**: 30 minutes per target on a Linux runner.
2. **Pre-release gate**: 24 hours per target before any PyPI release (`make fuzz-release`).
3. **PR diff trigger**: any PR modifying files under `src/decode/` or `src/pipeline/` triggers a 5-minute fuzz run on affected targets (via path filter in CI YAML).

Fuzz targets must be in the repository from day one. Writing fuzz targets retroactively after a bug is found is too late — they must be written while the author understands the code path.

Seed corpus in `tests/fuzz/corpus/`: minimum 20 seed files per format (small valid images, intentionally truncated images, images with unusual-but-legal chunks, known-malformed inputs). The corpus grows as libFuzzer discovers new coverage edges; new seeds are committed periodically.

**OSS-Fuzz submission**: agree with the Security Architect. Submit to OSS-Fuzz before the first PyPI release. The submission requires having libFuzzer targets already written — this plan provides them. OSS-Fuzz provides 24/7 fuzzing at zero cost after submission, strictly better than our weekly job. Add OSS-Fuzz submission as a release gate criterion, not an aspirational item.

#### 5b. Memory Safety: ASan + MSan + UBSan

**ASan+UBSan on every test build, not a separate job:**

```cmake
if(PIXMASK_SANITIZE)
    target_compile_options(pixmask_tests PRIVATE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
        -fno-sanitize-recover=undefined
    )
    target_link_options(pixmask_tests PRIVATE -fsanitize=address,undefined)
endif()
```

`-fno-sanitize-recover=undefined` is critical. Without it, UBSan reports and continues execution, hiding multiple bugs per run. Hard failure on first UB hit.

`PIXMASK_SANITIZE=ON` must be the default when `BUILD_TESTS=ON`. This is not opt-in.

**MSan (separate CI job, clang only, deferred to v0.2 for full coverage):**

MSan requires all dependencies to be MSan-instrumented or it generates false positives from reads of uninstrumented library memory. For v0.1, run MSan only on pure-C++ modules with no third-party deps: bit-depth reduction, median filter, bilateral filter, wavelet. These modules have no external library calls and can be MSan-tested correctly. Document this scope limitation in `CI.md`.

Full MSan with instrumented third-party deps (requires rebuilding libspng, libjpeg-turbo, libwebp against a MSan-instrumented sysroot) is a v0.2 task.

**CI structure:**

```yaml
jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-22.04, ubuntu-24.04, macos-14]
        python: ["3.10", "3.11", "3.12", "3.13"]
        build_type: [Debug, Release]
    steps:
      - run: cmake -DPIXMASK_SANITIZE=ON -DBUILD_TESTS=ON ...

  sanitizer_msan:
    runs-on: ubuntu-22.04
    steps:
      - run: cmake -DCMAKE_CXX_COMPILER=clang++ -DPIXMASK_MSAN=ON ...
```

#### 5c. Known CVE Reproduction Tests

Maintain fixtures for documented CVEs in image parsing libraries. Test contract: pixmask either rejects the file with a clean error before reaching vulnerable code, or is compiled against a non-vulnerable library version.

| Fixture | CVE | Vulnerable Library | Expected |
|---|---|---|---|
| `cve_2023_4863_trigger.webp` | CVE-2023-4863 | libwebp < 1.3.2 | error OR success without crash |
| `cve_2022_1622_trigger.png` | CVE-2022-1622 | libpng < 1.6.37 | error OR success without crash |
| `cve_2020_35538_trigger.jpg` | CVE-2020-35538 | libjpeg-turbo < 2.1.3 | error OR success without crash |

The assertion is: `(result.error_code != PIXMASK_OK) OR (no ASan finding)`. Under ASan, a heap-buffer-overflow that would be a CVE is immediately detectable as a test failure. The test passing under ASan means either the CVE is fixed in the vendored version or Stage 0 validation rejected the file before the vulnerable parse path.

Also add a CMake configure-time check: if vendored libwebp < 1.3.2, emit `cmake_fatal_error`. The version enforcement at configure time is itself a test — it prevents building a vulnerable binary.

---

### 6. Test Infrastructure

#### 6a. Test Image Fixtures

**Policy**: static fixtures < 50KB each are committed to `tests/fixtures/`. Large images (>= 512×512) are generated at test runtime from a fixed seed — not stored in git.

**Committed fixtures** (~2MB total):

```
tests/fixtures/
  clean_224.png          — 224×224 natural photo (CC0 license, source in SOURCES.md)
  clean_512.png          — 512×512 same source
  gradient_224.png       — synthetic horizontal gradient, analytically known values
  uniform_gray_224.png   — all pixels = 128
  checkerboard_224.png   — 8×8 black/white tile pattern
  text_in_image_224.png  — synthetic black text on white background
  near_black_224.png     — all pixels in [0, 5]
  near_white_224.png     — all pixels in [250, 255]
  exif_gps.jpg           — synthetic GPS EXIF (fake coordinates)
  xmp_metadata.png       — PNG with synthetic XMP iTXt chunk
  icc_large.png          — PNG with ~400KB ICC profile
  adversarial/
    pgd8_clean_{1..5}.png
    pgd8_adv_{1..5}.png
    cw_l2_clean_{1..5}.png
    cw_l2_adv_{1..5}.png
  malformed/
    truncated.png
    truncated.jpg
    corrupt_idat.png
    zero_bytes.bin
    one_byte_ff.bin
    random_kb.bin
    dimension_lie.png
    riff_overflow.webp
  cve/
    cve_2023_4863_trigger.webp
    cve_2022_1622_trigger.png
    cve_2020_35538_trigger.jpg
  SOURCES.md
```

`scripts/gen_fixtures.py` creates all programmatically-derivable fixtures (gradient, checkerboard, uniform fields, synthetic EXIF/XMP) from code. Running it reproduces the exact committed binaries. `scripts/gen_adversarial_fixtures.py` generates the adversarial pairs and documents the model checkpoint, epsilon, and steps.

**Runtime-generated images** (not stored): for large-resolution unit tests and benchmark inputs, use `np.random.default_rng(42).integers(0, 255, (H, W, 3), dtype=np.uint8)`. Fixed seed makes failures reproducible.

#### 6b. CI Matrix

```yaml
strategy:
  matrix:
    os: [ubuntu-22.04, ubuntu-24.04, macos-14]
    python: ["3.10", "3.11", "3.12", "3.13"]
    build_type: [Debug, Release]
```

24 combinations. All run on every PR. The benchmark regression gate runs only on `ubuntu-22.04 / python 3.12 / Release`.

**Deferred to v0.2**: Windows. MSVC's ASan integration is not equivalent to clang/gcc. It requires a separate CI story and has known limitations with the sanitizer API that would produce false positives on our memory patterns.

**Apple Silicon (macos-14, arm64)**: Highway dispatches to NEON. The full unit test suite must pass on macos-14. A test failure on macos-14 that passes on x86_64 is a Highway dispatch configuration bug and blocks the PR.

**Scheduled jobs** (not per-PR):
- Weekly fuzz: 30 minutes per target, Linux only.
- Weekly MSan full run on pure-C++ modules.
- Monthly: manually update `baseline.json` if benchmarks improved.

#### 6c. Testing SIMD Paths

Highway's runtime dispatch selects the best available SIMD target. Testing SIMD variants requires explicitly forcing each dispatch level.

**Method 1: Highway env vars at runtime:**

```bash
# Scalar-only path
HWY_DISABLE_AVX2=1 HWY_DISABLE_SSE4=1 HWY_DISABLE_SSSE3=1 HWY_DISABLE_SSE3=1 \
    ctest -R test_bit_depth

# SSE4.2-only path
HWY_DISABLE_AVX2=1 ctest -R test_bit_depth

# AVX2 (default on modern x86_64)
ctest -R test_bit_depth
```

These work when the binary is compiled with `HWY_COMPILE_ALL_ATTAINABLE`. Add a test that logs `hwy::TargetName(GetActiveTarget())` to verify the env var actually forced the expected target — not just that the test ran.

**Method 2: Compile-time target pinning (for CI matrix):**

```cmake
foreach(simd_target scalar sse2 avx2)
    string(TOUPPER ${simd_target} simd_upper)
    add_executable(test_bit_depth_${simd_target}
        tests/unit/test_bit_depth.cpp)
    target_compile_definitions(test_bit_depth_${simd_target}
        PRIVATE HWY_COMPILE_ONLY_${simd_upper})
    add_test(NAME bit_depth_${simd_target}
        COMMAND test_bit_depth_${simd_target})
endforeach()
```

Three separate binaries. The `sse2` binary runs on any x86_64 hardware, exercising SSE2 even on AVX2-capable CI runners. On Apple Silicon, only `scalar` and `neon` binaries are built.

**The SIMD correctness invariant**: for each of {bit-depth reduction, 3×3 median filter, bilateral filter, wavelet}, run identical inputs through all available SIMD targets and assert byte-for-byte identical output. This is the only test class that catches lane-boundary rounding errors, saturation arithmetic differences between SSE and AVX2, and Haar averaging precision mismatches. A dedicated `TEST_CASE("simd_cross_path_identity")` runs all paths against the scalar reference on the same input data.

---

### 7. Test Directory Structure

```
tests/
├── unit/
│   ├── test_bit_depth.cpp
│   ├── test_median_filter.cpp
│   ├── test_jpeg_roundtrip.cpp
│   ├── test_input_validation.cpp
│   └── test_metadata_strip.cpp
├── integration/
│   ├── test_pipeline_roundtrip.cpp
│   ├── test_adversarial_reduction.cpp
│   ├── test_malformed_inputs.cpp
│   └── test_preset_output_validity.cpp
├── bench/
│   ├── bench_pipeline.cpp
│   ├── baseline.json
│   └── test_python_api.py
├── fuzz/
│   ├── fuzz_decode_png.cpp
│   ├── fuzz_decode_jpeg.cpp
│   ├── fuzz_decode_webp.cpp
│   ├── fuzz_sanitize.cpp
│   └── corpus/
├── fixtures/
│   ├── (see Section 6a for complete list)
│   └── SOURCES.md
└── helpers/
    ├── ssim.hpp          (~60 lines, standalone SSIM — no opencv_contrib)
    ├── image_gen.hpp     (programmatic image generation with fixed-seed RNG)
    └── exif_walker.hpp   (~40 lines, minimal IFD walker for EXIF verification)
```

`helpers/` contains only code used by multiple test files. Code used by a single test file stays colocated with that file.

---

### 8. Non-Negotiable Positions

**I will block release without these:**

1. **ASan on default test build**. `PIXMASK_SANITIZE=ON` is the default when `BUILD_TESTS=ON`. Any PR that disables ASan to make a test pass is hiding a bug, not fixing it.

2. **Fuzz targets in the repository from day one**. Every parser entry point gets a fuzz target when it is written, not retroactively after a bug is found. The two tasks are coupled — the fuzz target is part of implementing the parser.

3. **Standalone SSIM in `helpers/ssim.hpp`**. Not opencv_contrib. The SSIM gate for integration tests must not fail to build on CI. If SSIM computation depends on `opencv_contrib`, the gate will fail intermittently and developers will learn to ignore it.

4. **SIMD cross-path identity tests** for all transforms with SIMD paths. This class of test catches the bugs that actually appear in practice on SIMD code: lane boundary errors, saturation arithmetic differences, alignment assumptions. None of these bugs are visible without explicitly comparing scalar and SIMD output.

5. **Adversarial fixture pairs committed as binary blobs**. They must be reproducible without GPU or PyTorch in the CI environment. A test that requires running PGD in CI is a test that breaks on torchattacks version conflicts.

**Willing to defer to v0.2:**

- Full MSan with instrumented third-party deps.
- Windows CI.
- Full VLM-in-the-loop ASR testing.

**On the default profile debate (paranoid vs balanced):**

From a testing perspective: whichever profile is the default, the SSIM and latency thresholds in integration tests must be co-located with the default profile configuration in a shared constants header. The test thresholds and the preset configuration must be a single source of truth. If the preset parameters change, the test thresholds change with them — automatically, not by remembering to update two files.

**On the Minimalist Architect's position (ship fast):**

Agree on scope. Disagree on test coverage. The minimalist pipeline — validate, decode, bit-depth, median, JPEG re-encode, output — absolutely can and should ship in v0.1. But that pipeline needs the full test suite described here. The test suite is not a feature that competes with pipeline features. It is the mechanism by which we know the pipeline actually works. A five-stage pipeline with no tests ships broken. The same pipeline with tests ships correctly or fails visibly.

The adversarial fixtures, SIMD identity tests, and fuzz targets are not gold-plating. They test the exact properties that make pixmask useful: that it actually reduces adversarial perturbations, that the SIMD optimizations are correct, and that the parsers do not crash on malformed input. Without those tests, we cannot make the claims in the README.

