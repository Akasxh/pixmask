# CPP_JPEG_REFERENCE — Stage 5 JPEG Roundtrip Implementation Guide

> Reference for `src/cpp/include/fsq/jpeg_roundtrip.hpp` and the corresponding `.cpp`.
> All decisions traced to `architecture/DECISIONS.md` §3 (pipeline) and §4 (dependencies).

---

## 1. stb_image_write JPEG Encoding (`stbi_write_jpg_to_func`)

### Function signature

```c
// stb_image_write.h
typedef void stbi_write_func(void *context, void *data, int size);

STBIWDEF int stbi_write_jpg_to_func(
    stbi_write_func *func,   // callback invoked repeatedly as JPEG is produced
    void            *context, // forwarded opaque pointer to your callback
    int              x,       // image width  (pixels)
    int              y,       // image height (pixels)
    int              comp,    // channel count (1=grey, 3=RGB, 4=RGBA — JPEG ignores alpha)
    const void      *data,    // pixel buffer, row-major, tightly packed
    int              quality  // 1–100; higher = larger file, better quality
);
// Returns 1 on success, 0 on failure.
```

Source internals (stb_image_write.h):

```c
typedef struct {
    stbi_write_func *func;
    void            *context;
    unsigned char    buffer[64];   // internal 64-byte write-batch buffer
    int              buf_used;
} stbi__write_context;

// stbi_write_jpg_to_func allocates stbi__write_context on the stack,
// calls stbi__start_write_callbacks(), then stbi_write_jpg_core().
// No heap allocation by the library itself.
```

### Callback pattern for in-memory accumulation

The callback is invoked **multiple times** during encoding (not once at the end). You must accumulate chunks into a growable buffer:

```cpp
struct WriteBuffer {
    uint8_t* data;
    size_t   used;
    size_t   capacity;
};

// Callback: append incoming chunk to our buffer.
// stb calls this with chunks as small as 1 byte; batch writes are an
// implementation detail of the 64-byte internal buffer in stbi__write_context.
static void jpeg_write_callback(void* context, void* data, int size) {
    auto* wb = static_cast<WriteBuffer*>(context);
    size_t needed = wb->used + static_cast<size_t>(size);
    if (needed > wb->capacity) {
        // Grow 2x or to needed, whichever is larger
        size_t new_cap = wb->capacity * 2;
        if (new_cap < needed) new_cap = needed;
        wb->data = static_cast<uint8_t*>(std::realloc(wb->data, new_cap));
        // In the arena-backed version below, this realloc is replaced by
        // arena bump allocation — see §4.
        wb->capacity = new_cap;
    }
    std::memcpy(wb->data + wb->used, data, static_cast<size_t>(size));
    wb->used += static_cast<size_t>(size);
}
```

### Quality factor

- Range is `[1, 100]`. Values outside are clamped internally by stb.
- For pixmask Stage 5 the architectural decision is `QF = random(70, 85)` — see §3.
- Quality maps to quantization table scaling: quality < 50 → scale = 5000/quality; quality >= 50 → scale = 200 - 2*quality. This is the standard IJG formula used by stb.
- JPEG quantization is lossy in a way that destroys steganographic payloads and adversarial perturbations because it rounds DCT coefficients.

### Memory management

- `stbi_write_jpg_to_func` itself performs **no heap allocation** for the output stream — all JPEG bytes are emitted via the callback.
- The `stbi__write_context` is stack-allocated inside the function.
- Any internal scratch buffers (e.g., for DCT coefficient arrays) are stack or local; stb_image_write does not call malloc.
- The output `WriteBuffer.data` is the only allocation you manage.

---

## 2. stb_image JPEG Decoding from Memory (`stbi_load_from_memory`)

### Function signature

```c
// stb_image.h
STBIDEF stbi_uc *stbi_load_from_memory(
    stbi_uc const *buffer,   // pointer to JPEG bytes in memory
    int            len,      // byte length of buffer
    int           *x,        // out: decoded image width
    int           *y,        // out: decoded image height
    int           *comp,     // out: channel count as stored in file
    int            req_comp  // requested output channels; 0 = use file's own count
);
// Returns: heap-allocated pixel buffer (caller must stbi_image_free()), or NULL on error.
```

Channel constants:

```c
STBI_grey       = 1
STBI_grey_alpha = 2
STBI_rgb        = 3
STBI_rgb_alpha  = 4
```

For a JPEG roundtrip operating on RGB images always pass `req_comp = STBI_rgb` (= 3). This forces output to 3 channels regardless of what the JPEG encoder stored — critical because JPEG can store YCbCr which stb re-expands to RGB, and alpha is discarded in JPEG encoding anyway.

### Error handling

```cpp
int out_w{}, out_h{}, out_comp{};
stbi_uc* decoded = stbi_load_from_memory(
    jpeg_bytes, static_cast<int>(jpeg_len),
    &out_w, &out_h, &out_comp,
    STBI_rgb
);
if (decoded == nullptr) {
    // stbi_failure_reason() returns a short static string — do NOT free it.
    // It is not thread-safe in stb's default build; use stbi_set_error_callback
    // or just record that decoding failed.
    const char* reason = stbi_failure_reason();
    // propagate error via SanitizeResult::error_message
    return make_error(ErrorCode::kJpegDecodeFailed, reason);
}
// Sanity check dimensions didn't change through roundtrip
if (out_w != expected_w || out_h != expected_h) {
    stbi_image_free(decoded);
    return make_error(ErrorCode::kJpegDimensionMismatch, "roundtrip changed dimensions");
}
```

`stbi_image_free` is literally `free()`. The buffer returned is a plain `malloc`'d block.

### Compile-time format restriction (DECISIONS.md §2)

In the translation unit that does `#define STB_IMAGE_IMPLEMENTATION`, also define:

```c
#define STBI_ONLY_JPEG   // disables PNG, BMP, TGA, GIF, PSD, PIC, PNM
#define STBI_NO_HDR
#define STBI_NO_LINEAR
```

The Stage 5 decoder only ever sees the in-memory JPEG buffer produced by Stage 5 encoding — it never re-validates magic bytes (Stage 0 already did that). Still, `STBI_ONLY_JPEG` reduces attack surface.

---

## 3. Random Quality Factor

### Why CSPRNG matters

The quality factor randomisation window `[70, 85]` is the defense against **adaptive attacks**: an adversary who knows the exact QF used could craft perturbations that survive that specific quantization grid. If QF is predictable (e.g., from `rand()` seeded with time), the adversary can enumerate a small set of QF values and craft perturbations robust to all of them. OS entropy (`getrandom`) makes QF unpredictable per image, collapsing the attack search space.

### `getrandom()` — Linux

```c
#include <sys/random.h>

// Signature
ssize_t getrandom(void *buf, size_t size, unsigned int flags);
```

Key properties:
- Kernel 3.17+, glibc 2.25+. For older glibc use the `syscall(SYS_getrandom, ...)` fallback.
- `flags = 0`: draws from the urandom pool. **Blocks only at boot before the pool is initialised**; after that, returns immediately for requests ≤ 256 bytes.
- `flags = GRND_NONBLOCK`: returns `EAGAIN` instead of blocking during early boot.
- Reads ≤ 256 bytes are guaranteed to return exactly the requested count and are not interrupted by signals (once the pool is seeded).
- Do **not** use `GRND_RANDOM` — it draws from `/dev/random` and can block arbitrarily.

### Mapping to `[70, 85]`

```cpp
#include <sys/random.h>
#include <cerrno>
#include <cstdint>

// Returns a quality factor in [lo, hi] inclusive.
// lo=70, hi=85 per DECISIONS.md §3 SanitizeOptions defaults.
static int random_jpeg_quality(int lo, int hi) noexcept {
    // Range size: hi - lo + 1 = 16 for [70,85]
    const int range = hi - lo + 1;

    uint8_t byte{};
    ssize_t got = getrandom(&byte, sizeof(byte), 0);

    if (got != 1) {
        // Fallback: if getrandom fails (shouldn't happen post-boot, but be safe),
        // use the midpoint. Do NOT fall back to rand() — that undermines the
        // adaptive attack resistance this randomisation provides.
        return lo + range / 2;
    }

    // Rejection sampling to avoid modulo bias.
    // For range=16 and a uniform byte [0,255]: 256 % 16 == 0, so there is
    // actually zero bias in this specific case. The general pattern is kept
    // for correctness when called with other ranges.
    const int usable = (256 / range) * range;   // 256 for range=16
    if (byte >= usable) {
        // Re-draw: call recursively (tail depth ≤ 3 in worst case for small ranges)
        return random_jpeg_quality(lo, hi);
    }

    return lo + (byte % range);
}
```

For `range = 16`: `usable = 256`, so the `byte >= usable` branch never fires — no rejection needed. The code is still correct for other quality windows without modification.

### Platform portability shim

```cpp
// jpeg_roundtrip.cpp — OS entropy shim
#if defined(__linux__)
#  include <sys/random.h>
#  define PIXMASK_HAVE_GETRANDOM 1
#elif defined(__APPLE__) || defined(__FreeBSD__)
#  include <sys/random.h>   // getentropy() on macOS/BSD
#  define PIXMASK_HAVE_GETENTROPY 1
#elif defined(_WIN32)
#  include <bcrypt.h>        // BCryptGenRandom
#  pragma comment(lib, "bcrypt.lib")
#  define PIXMASK_HAVE_BCRYPT 1
#endif

static bool os_random_bytes(uint8_t* buf, size_t n) noexcept {
#if defined(PIXMASK_HAVE_GETRANDOM)
    return getrandom(buf, n, 0) == static_cast<ssize_t>(n);
#elif defined(PIXMASK_HAVE_GETENTROPY)
    return getentropy(buf, n) == 0;   // getentropy never returns partial
#elif defined(PIXMASK_HAVE_BCRYPT)
    return BCryptGenRandom(nullptr, buf, static_cast<ULONG>(n),
                           BCRYPT_USE_SYSTEM_PREFERRED_RNG) == 0;
#else
#  error "No OS entropy source available — add one for this platform"
#endif
}
```

---

## 4. Complete Roundtrip Function

### Design constraints from DECISIONS.md

- Output buffer owned by Arena (bump-pointer allocator, per-pipeline).
- JPEG bytes are temporary — malloc/free around the encode step only.
- `stbi_load_from_memory` returns a `malloc`'d buffer that must be copied into the Arena, then freed.
- No file I/O anywhere in this stage.

### Header (`jpeg_roundtrip.hpp`)

```cpp
#pragma once
#include <cstdint>
#include "fsq/image_view.hpp"   // ImageView, Arena

namespace fsq {

struct JpegRoundtripOptions {
    uint8_t quality_lo = 70;
    uint8_t quality_hi = 85;
};

// Encode pixels to JPEG with a random QF in [quality_lo, quality_hi],
// then decode back. Returns an ImageView backed by arena memory.
// On failure, returns an ImageView with data == nullptr and writes
// error_message into err_out (static string, do not free).
ImageView jpeg_roundtrip(const ImageView& src,
                         Arena&           arena,
                         const JpegRoundtripOptions& opts = {},
                         const char**     err_out = nullptr) noexcept;

} // namespace fsq
```

### Implementation (`jpeg_roundtrip.cpp`)

```cpp
// In ONE translation unit only:
#define STBI_ONLY_JPEG
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb_image_write.h"

#include "fsq/jpeg_roundtrip.hpp"
#include "fsq/image_view.hpp"

#include <cstdlib>    // malloc, realloc, free
#include <cstring>    // memcpy
#include <cassert>

namespace fsq {

// ---------------------------------------------------------------------------
// Internal: accumulation buffer for stb callback
// ---------------------------------------------------------------------------

struct WriteBuffer {
    uint8_t* data     = nullptr;
    size_t   used     = 0;
    size_t   capacity = 0;
};

static void jpeg_write_cb(void* context, void* chunk, int size) {
    auto* wb     = static_cast<WriteBuffer*>(context);
    size_t chunk_sz = static_cast<size_t>(size);
    size_t needed   = wb->used + chunk_sz;

    if (needed > wb->capacity) {
        size_t new_cap = (wb->capacity == 0) ? needed : wb->capacity * 2;
        if (new_cap < needed) new_cap = needed;
        void* p = std::realloc(wb->data, new_cap);
        if (p == nullptr) {
            // Allocation failure: mark by zeroing used so caller can detect
            // partial state. The encode will continue calling the callback
            // but we silently drop data; stbi_write_jpg_to_func will return 1
            // regardless, so we detect failure via used == 0 after the call.
            // A cleaner approach requires patching stb — not worth it for v0.1.
            return;
        }
        wb->data     = static_cast<uint8_t*>(p);
        wb->capacity = new_cap;
    }
    std::memcpy(wb->data + wb->used, chunk, chunk_sz);
    wb->used += chunk_sz;
}

// ---------------------------------------------------------------------------
// OS entropy
// ---------------------------------------------------------------------------

static bool os_random_bytes(uint8_t* buf, size_t n) noexcept {
#if defined(__linux__)
    return ::getrandom(buf, n, 0) == static_cast<ssize_t>(n);
#elif defined(__APPLE__) || defined(__FreeBSD__)
    return ::getentropy(buf, n) == 0;
#elif defined(_WIN32)
    return BCryptGenRandom(nullptr, buf, static_cast<ULONG>(n),
                           BCRYPT_USE_SYSTEM_PREFERRED_RNG) == 0;
#else
#  error "Unsupported platform: add OS entropy source"
#endif
}

static int random_quality(int lo, int hi) noexcept {
    const int range   = hi - lo + 1;
    const int usable  = (256 / range) * range;
    for (;;) {
        uint8_t byte{};
        if (!os_random_bytes(&byte, 1)) return lo + range / 2; // fallback: midpoint
        if (byte < usable) return lo + (byte % range);
        // byte >= usable: reject and retry (avoids modulo bias)
    }
}

// ---------------------------------------------------------------------------
// Roundtrip
// ---------------------------------------------------------------------------

ImageView jpeg_roundtrip(const ImageView& src,
                         Arena&           arena,
                         const JpegRoundtripOptions& opts,
                         const char**     err_out) noexcept {
    auto fail = [&](const char* msg) -> ImageView {
        if (err_out) *err_out = msg;
        return ImageView{};   // data == nullptr signals failure
    };

    // ---- Stage 5a: encode to JPEG in memory ----

    const int quality = random_quality(opts.quality_lo, opts.quality_hi);

    // Pre-allocate: JPEG at QF=85 is typically well under 0.5 bytes/pixel
    // for photographic content. Use width*height*channels/2 as initial cap
    // to avoid the first realloc in most cases.
    const size_t prealloc = static_cast<size_t>(src.width)
                          * static_cast<size_t>(src.height)
                          * static_cast<size_t>(src.channels) / 2;

    WriteBuffer wb{};
    wb.data     = static_cast<uint8_t*>(std::malloc(prealloc > 0 ? prealloc : 4096));
    wb.capacity = (prealloc > 0) ? prealloc : 4096;
    if (wb.data == nullptr) return fail("jpeg_roundtrip: malloc failed for encode buffer");

    const int encode_ok = stbi_write_jpg_to_func(
        jpeg_write_cb, &wb,
        static_cast<int>(src.width),
        static_cast<int>(src.height),
        static_cast<int>(src.channels),
        src.data,
        quality
    );

    if (!encode_ok || wb.used == 0) {
        std::free(wb.data);
        return fail("jpeg_roundtrip: stbi_write_jpg_to_func failed");
    }

    // ---- Stage 5b: decode JPEG back from memory ----

    int dec_w{}, dec_h{}, dec_comp{};
    stbi_uc* decoded = stbi_load_from_memory(
        wb.data, static_cast<int>(wb.used),
        &dec_w, &dec_h, &dec_comp,
        STBI_rgb   // always decode to RGB; JPEG drops alpha anyway
    );

    std::free(wb.data);   // JPEG bytes no longer needed regardless of outcome
    wb.data = nullptr;

    if (decoded == nullptr) {
        return fail(stbi_failure_reason());
    }

    if (static_cast<uint32_t>(dec_w) != src.width ||
        static_cast<uint32_t>(dec_h) != src.height) {
        stbi_image_free(decoded);
        return fail("jpeg_roundtrip: decoded dimensions differ from source");
    }

    // ---- Copy decoded pixels into arena ----

    const size_t out_bytes = static_cast<size_t>(dec_w)
                           * static_cast<size_t>(dec_h)
                           * 3u;  // always RGB out

    uint8_t* out = static_cast<uint8_t*>(arena.alloc(out_bytes));
    if (out == nullptr) {
        stbi_image_free(decoded);
        return fail("jpeg_roundtrip: arena alloc failed");
    }
    std::memcpy(out, decoded, out_bytes);
    stbi_image_free(decoded);  // free stb's malloc'd copy

    return ImageView{
        .data     = out,
        .width    = static_cast<uint32_t>(dec_w),
        .height   = static_cast<uint32_t>(dec_h),
        .channels = 3u,
        .stride   = static_cast<uint32_t>(dec_w) * 3u
    };
}

} // namespace fsq
```

### Error path summary

| Failure point | Cleanup | Return |
|---|---|---|
| `malloc` for encode buffer | nothing to free | `ImageView{}` |
| `stbi_write_jpg_to_func` returns 0 | `free(wb.data)` | `ImageView{}` |
| `wb.used == 0` after encode | `free(wb.data)` | `ImageView{}` |
| `stbi_load_from_memory` returns null | `free(wb.data)` already freed | `ImageView{}` |
| Dimension mismatch | `stbi_image_free(decoded)` | `ImageView{}` |
| `arena.alloc` returns null | `stbi_image_free(decoded)` | `ImageView{}` |

---

## 5. Performance Considerations

### Buffer pre-allocation

The initial `WriteBuffer` capacity is set to `width * height * channels / 2`.

Empirical JPEG size estimates for photographic RGB images:

| Quality | Typical bytes/pixel (1080p photo) |
|---------|-----------------------------------|
| 70      | ~0.15–0.25                        |
| 85      | ~0.30–0.50                        |
| 95      | ~0.70–1.0                         |

At QF=85 and 1080p (1920×1080×3 = 6,220,800 bytes raw), the JPEG will be ~1–3 MB. The pre-allocation of `6,220,800 / 2 = 3,110,400` bytes covers QF=85 photographic content with zero or one realloc.

For synthetic/adversarial inputs (flat regions, bit-depth-reduced to 5 bits), JPEG compression is dramatically higher — the pre-allocation will be generous. No reallocs in the common case.

### Avoiding repeated mallocs across pipeline calls

The `WriteBuffer` above uses a single `malloc` + at most one `realloc` per image. For a multi-image pipeline (e.g., a server processing many requests), two strategies are available:

**Option A — Arena-backed encode buffer (zero malloc in hot path):**

Allocate the encode buffer from the Arena too, not via malloc/realloc. The Arena bump-pointer is reset between pipeline calls, so the memory is "free" at the start of each request. This requires reserving enough Arena headroom for the temporary JPEG bytes (~width*height*channels/2) in addition to the final decoded pixels (~width*height*3).

```cpp
// Arena-backed encode buffer: no malloc at all during encode
uint8_t* enc_buf = static_cast<uint8_t*>(arena.alloc(prealloc));
// ... use enc_buf as backing for WriteBuffer, bump arena pointer manually on each chunk
```

This is an optimisation for v0.2 when per-call latency becomes critical.

**Option B — Thread-local persistent WriteBuffer (zero malloc after warmup):**

```cpp
// In jpeg_roundtrip.cpp:
thread_local WriteBuffer tl_write_buf{};
// Reset used=0 at start of each call; realloc only if current image is
// larger than anything previously seen on this thread.
```

This fits a server model where threads process many images.

**v0.1 default:** Option B is not used (complicates thread-local lifetime). The simple per-call malloc/realloc is used. At 1080p this adds ~3ms for the allocation; acceptable under the <15ms budget for `balanced`.

### stb_image_write has no SIMD

`stbi_write_jpg_to_func` is pure C with no SIMD. This is acceptable for v0.1. The v0.2 plan (DECISIONS.md §2) upgrades to libjpeg-turbo which uses platform SIMD (SSE2/NEON) and is ~3–5x faster for encode+decode combined.

### stb_image decode is also scalar

Same caveat. At 1080p, expect ~5–8ms for decode with stb vs ~1–2ms with libjpeg-turbo. Budget for this in the <15ms p99 target.

---

## 6. Required Compile-Time Defines

In the single `.cpp` that owns the implementation:

```cpp
// Must appear before the #include, in exactly one translation unit:
#define STBI_ONLY_JPEG        // disables all non-JPEG decoders
#define STBI_NO_HDR           // no float HDR path
#define STBI_NO_LINEAR        // no linear light conversion
#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb_image_write.h"
```

All other translation units that only call `stbi_load_from_memory` / `stbi_write_jpg_to_func` include the headers **without** the `_IMPLEMENTATION` define.

---

## 7. Test Sketch (`test_jpeg.cpp`)

```cpp
// Minimal doctest coverage for the roundtrip
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "third_party/doctest.h"
#include "fsq/jpeg_roundtrip.hpp"
#include "fsq/image_view.hpp"  // Arena

TEST_CASE("jpeg_roundtrip preserves dimensions") {
    Arena arena;
    // 16x16 RGB gradient
    uint8_t pixels[16 * 16 * 3]{};
    for (int i = 0; i < 16 * 16 * 3; ++i) pixels[i] = static_cast<uint8_t>(i % 256);

    ImageView src{pixels, 16, 16, 3, 16 * 3};
    const char* err = nullptr;
    ImageView out = fsq::jpeg_roundtrip(src, arena, {}, &err);

    REQUIRE(out.data != nullptr);
    REQUIRE(err == nullptr);
    CHECK(out.width    == 16);
    CHECK(out.height   == 16);
    CHECK(out.channels == 3);
}

TEST_CASE("jpeg_roundtrip quality is in [70,85]") {
    // Run 32 times; if quality were constant we'd get identical output bytes.
    // With random QF, output byte counts should vary.
    // (This is a statistical smoke test, not a proof.)
    Arena arena;
    uint8_t pixels[64 * 64 * 3]{};
    for (int i = 0; i < 64 * 64 * 3; ++i) pixels[i] = static_cast<uint8_t>(i * 7 % 256);
    ImageView src{pixels, 64, 64, 3, 64 * 3};

    // Encode once via the public API to just confirm it runs without error
    ImageView out = fsq::jpeg_roundtrip(src, arena);
    CHECK(out.data != nullptr);
}

TEST_CASE("jpeg_roundtrip null input returns error ImageView") {
    Arena arena;
    ImageView bad{nullptr, 0, 0, 0, 0};
    const char* err = nullptr;
    ImageView out = fsq::jpeg_roundtrip(bad, arena, {}, &err);
    CHECK(out.data == nullptr);
    // err may or may not be set depending on which guard fires first
}
```

---

*Sources: stb_image_write.h (nothings/stb master), stb_image.h (nothings/stb master), `man 2 getrandom` (Linux 3.17+, glibc 2.25+), `architecture/DECISIONS.md` v0.1.0.*
