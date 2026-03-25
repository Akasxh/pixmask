# pixmask v0.1.0 — C++ Arena Allocator & Pipeline Reference

> Authoritative implementation reference for `arena.h`, `image_view.h`, and `pipeline.h/cpp`.
> Derived from `DECISIONS.md`. All decisions in DECISIONS.md take precedence over this file.

---

## 1. Arena Allocator (`include/pixmask/arena.h`)

### Design constraints (from DECISIONS.md §5)

- Bump-pointer allocator, 32 MB default first block
- Grows via linked list of additional blocks (no realloc, pointers stay valid)
- `reset()` reuses the same memory — does NOT free blocks, just resets offset to 0
- Non-copyable, move-only
- Thread safety: NONE intentional. One `Arena` per `Pipeline` instance; `Pipeline` is not shared across threads. Adding atomics would be pure overhead.

### Header

```cpp
// include/pixmask/arena.h
#pragma once
#include <cstddef>
#include <cstdint>
#include <cassert>
#include <new>          // std::bad_alloc
#include <utility>      // std::exchange

namespace pixmask {

class Arena {
public:
    static constexpr size_t kDefaultBlockSize = 32ULL << 20; // 32 MB

    explicit Arena(size_t block_size = kDefaultBlockSize)
        : block_size_(block_size)
    {
        head_ = alloc_block(block_size_);
        current_ = head_;
    }

    ~Arena() noexcept {
        Block* b = head_;
        while (b) {
            Block* next = b->next;
            ::operator delete(b->storage, std::align_val_t{64});
            delete b;
            b = next;
        }
    }

    // Non-copyable
    Arena(const Arena&)            = delete;
    Arena& operator=(const Arena&) = delete;

    // Move-only
    Arena(Arena&& other) noexcept
        : head_(std::exchange(other.head_, nullptr))
        , current_(std::exchange(other.current_, nullptr))
        , block_size_(other.block_size_)
    {}

    Arena& operator=(Arena&& other) noexcept {
        if (this != &other) {
            this->~Arena();
            head_       = std::exchange(other.head_, nullptr);
            current_    = std::exchange(other.current_, nullptr);
            block_size_ = other.block_size_;
        }
        return *this;
    }

    // Core allocation. Returns aligned pointer. Never returns null — throws std::bad_alloc on OOM.
    // alignment must be a power of 2. Default 64 matches typical cache line + SIMD requirements.
    [[nodiscard]] void* allocate(size_t bytes, size_t alignment = 64) {
        assert((alignment & (alignment - 1)) == 0 && "alignment must be power of 2");

        // Try to satisfy from current block
        void* ptr = bump(current_, bytes, alignment);
        if (ptr) return ptr;

        // Current block exhausted. Walk existing overflow blocks first.
        // (After reset(), we reuse them in order, so this path is rare in steady-state.)
        for (Block* b = current_->next; b != nullptr; b = b->next) {
            ptr = bump(b, bytes, alignment);
            if (ptr) {
                current_ = b;
                return ptr;
            }
        }

        // Allocate a new block. Size = max(block_size_, bytes + alignment).
        size_t new_size = block_size_;
        if (bytes + alignment > new_size) new_size = bytes + alignment;
        Block* nb = alloc_block(new_size);
        current_->next = nb;
        current_ = nb;
        ptr = bump(current_, bytes, alignment);
        assert(ptr && "freshly allocated block must satisfy request");
        return ptr;
    }

    // Typed helper — allocates and default-constructs trivially constructible types.
    template <typename T>
    [[nodiscard]] T* allocate_array(size_t count) {
        static_assert(__is_trivially_constructible(T),
                      "Arena only supports trivially constructible types");
        return static_cast<T*>(allocate(sizeof(T) * count, alignof(T)));
    }

    // Reset: rewinds all block offsets to zero. Does NOT free memory.
    // All previously returned pointers are invalidated after reset().
    void reset() noexcept {
        for (Block* b = head_; b != nullptr; b = b->next) {
            b->offset = 0;
        }
        current_ = head_;
    }

    // Diagnostic: total bytes allocated across all blocks (capacity, not used bytes).
    size_t capacity_bytes() const noexcept {
        size_t total = 0;
        for (const Block* b = head_; b != nullptr; b = b->next) {
            total += b->size;
        }
        return total;
    }

private:
    struct Block {
        uint8_t* storage = nullptr;
        size_t   size    = 0;
        size_t   offset  = 0;
        Block*   next    = nullptr;
    };

    // Try to bump-allocate `bytes` with `alignment` from `b`.
    // Returns pointer on success, nullptr if insufficient space.
    static void* bump(Block* b, size_t bytes, size_t alignment) noexcept {
        uintptr_t base    = reinterpret_cast<uintptr_t>(b->storage) + b->offset;
        uintptr_t aligned = (base + (alignment - 1)) & ~(alignment - 1);
        size_t    waste   = aligned - base;
        size_t    needed  = waste + bytes;
        if (b->offset + needed > b->size) return nullptr;
        b->offset += needed;
        return reinterpret_cast<void*>(aligned);
    }

    static Block* alloc_block(size_t size) {
        Block* b = new Block{};
        b->storage = static_cast<uint8_t*>(
            ::operator new(size, std::align_val_t{64})
        );
        b->size = size;
        return b;
    }

    Block* head_       = nullptr;
    Block* current_    = nullptr;
    size_t block_size_ = kDefaultBlockSize;
};

} // namespace pixmask
```

### Key invariants

| Property | Value |
|---|---|
| Default first block | 32 MB |
| Alignment default | 64 bytes (cache line + AVX-512 width) |
| Overflow behavior | Allocates new block, appends to linked list |
| `reset()` effect | Rewinds offsets; blocks stay allocated |
| Thread safety | None. One arena per Pipeline, Pipelines are thread-local. |
| Pointer stability | Pointers from before `reset()` are invalidated after `reset()`. Pointers within one epoch are stable (no realloc). |

---

## 2. ImageView (`include/pixmask/image_view.h`)

### Design constraints

- Trivially copyable — cheap to pass by value through stage functions
- Does NOT own memory — `Arena` owns the underlying buffer
- `stride` can be larger than `width * channels` (aligned rows)
- Helpers must be `constexpr`/`noexcept` — called in hot loops

```cpp
// include/pixmask/image_view.h
#pragma once
#include <cstdint>
#include <cstddef>

namespace pixmask {

struct ImageView {
    uint8_t* data     = nullptr;
    uint32_t width    = 0;
    uint32_t height   = 0;
    uint32_t channels = 0;   // 1 (gray), 3 (RGB), 4 (RGBA)
    uint32_t stride   = 0;   // bytes per row; stride >= width * channels

    // Bytes in one row as stored (may include padding).
    [[nodiscard]] constexpr size_t row_bytes() const noexcept {
        return stride;
    }

    // Total bytes of the pixel buffer (height * stride).
    [[nodiscard]] constexpr size_t total_bytes() const noexcept {
        return static_cast<size_t>(height) * stride;
    }

    // Pointer to the start of row y. No bounds check — callers own that.
    [[nodiscard]] uint8_t* row(uint32_t y) noexcept {
        return data + static_cast<size_t>(y) * stride;
    }
    [[nodiscard]] const uint8_t* row(uint32_t y) const noexcept {
        return data + static_cast<size_t>(y) * stride;
    }

    // Pixel pointer (x, y). No bounds check.
    [[nodiscard]] uint8_t* pixel(uint32_t x, uint32_t y) noexcept {
        return row(y) + static_cast<size_t>(x) * channels;
    }
    [[nodiscard]] const uint8_t* pixel(uint32_t x, uint32_t y) const noexcept {
        return row(y) + static_cast<size_t>(x) * channels;
    }

    [[nodiscard]] bool valid() const noexcept {
        return data != nullptr && width > 0 && height > 0
               && (channels == 1 || channels == 3 || channels == 4)
               && stride >= width * channels;
    }
};

} // namespace pixmask
```

### Why stride is separate from `width * channels`

stb_image can return packed rows (`stride == width * channels`). When we allocate arena buffers for filter output, we align stride to 64 bytes so that each row starts on a cache line. This enables Highway SIMD to process whole rows without masking at the start.

Stride calculation for arena-allocated images:

```cpp
// Align stride to 64 bytes for SIMD-friendly row starts.
static inline uint32_t aligned_stride(uint32_t width, uint32_t channels,
                                      uint32_t alignment = 64) {
    uint32_t raw = width * channels;
    return (raw + alignment - 1) & ~(alignment - 1);
}
```

---

## 3. Supporting Types (`include/pixmask/pixmask.h`)

Already specified in DECISIONS.md §5. Reproduced here for completeness:

```cpp
// include/pixmask/pixmask.h
#pragma once
#include <cstdint>
#include <cstddef>
#include "pixmask/image_view.h"

namespace pixmask {

struct SanitizeOptions {
    uint8_t  bit_depth        = 5;
    uint8_t  median_radius    = 1;          // kernel = 2r+1 → 1 = 3×3
    uint8_t  jpeg_quality_lo  = 70;
    uint8_t  jpeg_quality_hi  = 85;
    uint32_t max_width        = 8192;
    uint32_t max_height       = 8192;
    uint64_t max_file_bytes   = 50ULL << 20;
    uint32_t max_decomp_ratio = 100;
};

// Error codes (error_code field in SanitizeResult)
enum class SanitizeError : uint32_t {
    Ok                = 0,
    BadMagicBytes     = 1,
    DimensionsTooLarge = 2,
    FileTooLarge      = 3,
    DecompRatioBreach = 4,
    UnsupportedFormat = 5,
    DecodeFailed      = 6,
    EncodeFailed      = 7,
    OomFailed         = 8,
};

struct SanitizeResult {
    ImageView   image;          // Valid only when success == true. Owned by Pipeline's Arena.
    bool        success  = false;
    SanitizeError error_code = SanitizeError::Ok;
    const char* error_message = nullptr;  // Points to string literal, not arena. Always valid.
};

} // namespace pixmask
```

---

## 4. In-place vs Copy Semantics

| Stage | Operation | In-place? | Reason |
|---|---|---|---|
| Stage 3: Bit-depth | `pixel[i] = (pixel[i] >> (8 - bits)) << (8 - bits)` | YES | Each pixel is independent; no neighbor reads |
| Stage 4: Median 3×3 | Sorting network over 9-pixel neighborhood | NO | Reading neighbors while writing output would corrupt reads; must use separate output buffer |
| Stage 5: JPEG RT | Encode + decode cycle | NO (inherently) | encode→compressed bytes→decode→new pixel buffer; three distinct buffers exist simultaneously |

For bit-depth reduction:
- Input `ImageView` modified directly
- No new arena allocation needed
- Return same `ImageView` from stage function (for chaining clarity)

For median filter:
- Input `ImageView` is read-only during the filter pass
- Output buffer allocated from arena: `arena.allocate_array<uint8_t>(out_stride * height)`
- Output `ImageView` constructed with new pointer, same dimensions
- Input buffer is now dead — arena will reclaim on next `reset()`

For JPEG roundtrip:
- Stage allocates a compressed JPEG byte buffer from arena (encode destination)
- stb_image_write callback writes into this buffer
- stb_image decode reads from it, writes into a *second* arena buffer
- The compressed buffer is dead after decode completes

---

## 5. Pipeline Orchestrator

### Header (`include/pixmask/pipeline.h`)

```cpp
// include/pixmask/pipeline.h
#pragma once
#include "pixmask/pixmask.h"
#include "pixmask/arena.h"
#include "pixmask/image_view.h"

namespace pixmask {

class Pipeline {
public:
    explicit Pipeline(SanitizeOptions opts = {});

    // Non-copyable (Arena is non-copyable)
    Pipeline(const Pipeline&)            = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    // Movable
    Pipeline(Pipeline&&)            = default;
    Pipeline& operator=(Pipeline&&) = default;

    // Primary entry point. Thread safety: NOT thread-safe.
    // Each call resets the arena. Do not share a Pipeline across threads.
    SanitizeResult run(const uint8_t* input, size_t len);

    // Expose arena for diagnostics/testing only.
    const Arena& arena() const noexcept { return arena_; }

private:
    SanitizeOptions opts_;
    Arena           arena_;

    // Per-stage helpers. All take/return ImageView by value (cheap — 24 bytes).
    // They may allocate from arena_ but never call arena_.reset().
    bool      stage0_validate(const uint8_t* input, size_t len);
    ImageView stage1_decode(const uint8_t* input, size_t len);
    void      stage3_bitdepth(ImageView& img);              // in-place
    ImageView stage4_median(const ImageView& src);          // allocates new buffer
    ImageView stage5_jpeg_roundtrip(const ImageView& src);  // allocates new buffer
};

} // namespace pixmask
```

### Implementation (`src/pipeline.cpp`)

```cpp
// src/pipeline.cpp
#include "pixmask/pipeline.h"
#include "pixmask/validate.h"        // stage0_validate_impl
#include "pixmask/decode.h"          // stb_image wrapper
#include "pixmask/bitdepth.h"        // bit_depth_reduce
#include "pixmask/median.h"          // median_3x3
#include "pixmask/jpeg_roundtrip.h"  // jpeg_roundtrip

#include <cstring>
#include <cstdlib>  // arc4random_uniform / rand fallback

// ---- stb_image_write arena-backed write callback --------------------------------

namespace pixmask::detail {

struct ArenaWriteCtx {
    Arena*   arena;
    uint8_t* buf;
    size_t   capacity;
    size_t   written;
};

// Called by stb_image_write during JPEG encode.
// Grows the arena-backed buffer if needed (by allocating a larger slab and memcpy-ing).
// NOTE: stb_image_write calls this multiple times in sequence; we accumulate.
static void arena_write_func(void* ctx_ptr, void* data, int size) {
    auto* ctx = static_cast<ArenaWriteCtx*>(ctx_ptr);
    if (size <= 0) return;

    size_t needed = ctx->written + static_cast<size_t>(size);
    if (needed > ctx->capacity) {
        // Double-or-fit growth.
        size_t new_cap = ctx->capacity * 2;
        if (new_cap < needed) new_cap = needed;
        uint8_t* new_buf = ctx->arena->allocate_array<uint8_t>(new_cap);
        std::memcpy(new_buf, ctx->buf, ctx->written);
        ctx->buf      = new_buf;
        ctx->capacity = new_cap;
        // Old buffer is dead but arena owns it — will be reclaimed on reset().
    }
    std::memcpy(ctx->buf + ctx->written, data, static_cast<size_t>(size));
    ctx->written += static_cast<size_t>(size);
}

} // namespace pixmask::detail

// ---- Pipeline implementation ---------------------------------------------------

namespace pixmask {

Pipeline::Pipeline(SanitizeOptions opts)
    : opts_(opts)
    , arena_(Arena::kDefaultBlockSize)
{}

SanitizeResult Pipeline::run(const uint8_t* input, size_t len) {
    // Step 1: Reset arena. All previous pointers from this Pipeline are invalidated.
    arena_.reset();

    // Step 2: Validate — returns early on any violation.
    if (!stage0_validate(input, len)) {
        // Error code/message set inside stage0_validate.
        // Return a sentinel result.
        SanitizeResult r{};
        r.success       = false;
        r.error_code    = last_error_;
        r.error_message = last_error_message_;
        return r;
    }

    // Step 3: Decode into arena-allocated buffer.
    ImageView img = stage1_decode(input, len);
    if (!img.valid()) {
        SanitizeResult r{};
        r.success       = false;
        r.error_code    = SanitizeError::DecodeFailed;
        r.error_message = "stb_image decode failed";
        return r;
    }

    // Step 4: Bit-depth reduce — in-place.
    stage3_bitdepth(img);

    // Step 5: Median 3×3 — allocates new buffer from arena.
    ImageView filtered = stage4_median(img);
    if (!filtered.valid()) {
        SanitizeResult r{};
        r.success       = false;
        r.error_code    = SanitizeError::OomFailed;
        r.error_message = "median filter allocation failed";
        return r;
    }

    // Step 6: JPEG roundtrip — allocates JPEG compressed buffer + decode output buffer.
    ImageView final_img = stage5_jpeg_roundtrip(filtered);
    if (!final_img.valid()) {
        SanitizeResult r{};
        r.success       = false;
        r.error_code    = SanitizeError::EncodeFailed;
        r.error_message = "JPEG roundtrip failed";
        return r;
    }

    // Step 7: Return final ImageView (still owned by arena_).
    SanitizeResult r{};
    r.image         = final_img;
    r.success       = true;
    r.error_code    = SanitizeError::Ok;
    r.error_message = nullptr;
    return r;
}

// ---- Stage implementations -----------------------------------------------------

bool Pipeline::stage0_validate(const uint8_t* input, size_t len) {
    return validate(input, len, opts_, &last_error_, &last_error_message_);
}

ImageView Pipeline::stage1_decode(const uint8_t* input, size_t len) {
    int w = 0, h = 0, ch = 0;

    // stb_image decode: allocates via stbi malloc. We immediately copy into arena.
    // Keeping the stbi buffer alive only for the memcpy then freeing.
    uint8_t* stbi_buf = stbi_load_from_memory(
        input, static_cast<int>(len), &w, &h, &ch, 0
    );
    if (!stbi_buf) return {};  // invalid ImageView (data == nullptr)

    // Enforce dimension limits (belt-and-suspenders after stage0).
    if (static_cast<uint32_t>(w) > opts_.max_width ||
        static_cast<uint32_t>(h) > opts_.max_height) {
        stbi_image_free(stbi_buf);
        return {};
    }

    // Allocate arena buffer with aligned stride.
    uint32_t stride = aligned_stride(
        static_cast<uint32_t>(w),
        static_cast<uint32_t>(ch)
    );
    uint8_t* arena_buf = arena_.allocate_array<uint8_t>(
        static_cast<size_t>(h) * stride
    );

    // Copy row-by-row (packed → strided).
    uint32_t packed_stride = static_cast<uint32_t>(w * ch);
    for (int y = 0; y < h; ++y) {
        std::memcpy(
            arena_buf + static_cast<size_t>(y) * stride,
            stbi_buf  + static_cast<size_t>(y) * packed_stride,
            packed_stride
        );
    }
    stbi_image_free(stbi_buf);

    ImageView img;
    img.data     = arena_buf;
    img.width    = static_cast<uint32_t>(w);
    img.height   = static_cast<uint32_t>(h);
    img.channels = static_cast<uint32_t>(ch);
    img.stride   = stride;
    return img;
}

void Pipeline::stage3_bitdepth(ImageView& img) {
    // Delegates to bitdepth.h (Highway SIMD dispatch).
    // Modifies img.data in-place. img fields (width/height/stride) unchanged.
    bit_depth_reduce(img, opts_.bit_depth);
}

ImageView Pipeline::stage4_median(const ImageView& src) {
    // Allocate output buffer — separate from src.data.
    uint8_t* out_buf = arena_.allocate_array<uint8_t>(src.total_bytes());
    if (!out_buf) return {};

    ImageView dst = src;       // copy metadata
    dst.data      = out_buf;   // point to new buffer

    // Delegates to median.h (Bose-Nelson sorting network, Highway SIMD).
    median_3x3(src, dst, opts_.median_radius);
    return dst;
}

ImageView Pipeline::stage5_jpeg_roundtrip(const ImageView& src) {
    // Pick a random JPEG quality in [jpeg_quality_lo, jpeg_quality_hi].
    // arc4random_uniform if available (OpenBSD/macOS/Linux with libbsd);
    // fall back to a seeded LCG for portability.
    int quality_range = static_cast<int>(opts_.jpeg_quality_hi)
                      - static_cast<int>(opts_.jpeg_quality_lo)
                      + 1;
#if defined(__linux__) || defined(__APPLE__)
    int quality = static_cast<int>(opts_.jpeg_quality_lo)
                + static_cast<int>(arc4random_uniform(
                      static_cast<uint32_t>(quality_range)));
#else
    // Deterministic fallback — acceptable; JPEG QF randomization is defense-in-depth.
    int quality = static_cast<int>(opts_.jpeg_quality_lo)
                + (rand() % quality_range);
#endif

    // --- Encode phase ---
    // Pre-allocate a JPEG compressed buffer from arena.
    // Upper bound: width * height * channels * 2 is extremely conservative for JPEG.
    size_t initial_cap = static_cast<size_t>(src.width) * src.height * src.channels;
    detail::ArenaWriteCtx write_ctx{};
    write_ctx.arena    = &arena_;
    write_ctx.buf      = arena_.allocate_array<uint8_t>(initial_cap);
    write_ctx.capacity = initial_cap;
    write_ctx.written  = 0;

    // stb_image_write expects packed (non-strided) input.
    // If src is strided, we need to pack it first.
    const uint8_t* encode_src = src.data;
    uint32_t packed_stride    = src.width * src.channels;
    std::unique_ptr<uint8_t[]> pack_buf;  // only allocated if needed

    if (src.stride != packed_stride) {
        // Pack rows into a contiguous buffer (stack for small images, heap otherwise).
        // Arena allocation is fine here — this buffer is short-lived.
        uint8_t* packed = arena_.allocate_array<uint8_t>(
            static_cast<size_t>(src.height) * packed_stride
        );
        for (uint32_t y = 0; y < src.height; ++y) {
            std::memcpy(
                packed + static_cast<size_t>(y) * packed_stride,
                src.row(y),
                packed_stride
            );
        }
        encode_src = packed;
    }

    int encode_ok = stbi_write_jpg_to_func(
        detail::arena_write_func,
        &write_ctx,
        static_cast<int>(src.width),
        static_cast<int>(src.height),
        static_cast<int>(src.channels),
        encode_src,
        quality
    );
    if (!encode_ok || write_ctx.written == 0) return {};

    // --- Decode phase ---
    int w2 = 0, h2 = 0, ch2 = 0;
    uint8_t* stbi_out = stbi_load_from_memory(
        write_ctx.buf,
        static_cast<int>(write_ctx.written),
        &w2, &h2, &ch2, 0
    );
    if (!stbi_out) return {};

    // Copy decode output into arena (same pattern as stage1_decode).
    uint32_t out_stride = aligned_stride(
        static_cast<uint32_t>(w2),
        static_cast<uint32_t>(ch2)
    );
    uint8_t* arena_out = arena_.allocate_array<uint8_t>(
        static_cast<size_t>(h2) * out_stride
    );
    uint32_t packed_out = static_cast<uint32_t>(w2 * ch2);
    for (int y = 0; y < h2; ++y) {
        std::memcpy(
            arena_out + static_cast<size_t>(y) * out_stride,
            stbi_out  + static_cast<size_t>(y) * packed_out,
            packed_out
        );
    }
    stbi_image_free(stbi_out);

    ImageView out;
    out.data     = arena_out;
    out.width    = static_cast<uint32_t>(w2);
    out.height   = static_cast<uint32_t>(h2);
    out.channels = static_cast<uint32_t>(ch2);
    out.stride   = out_stride;
    return out;
}

} // namespace pixmask
```

---

## 6. Arena Memory Lifecycle Within One `run()` Call

```
arena_.reset()
    │
    ▼
stage1_decode: stbi_buf (heap, stbi-owned)
               ──memcpy──► arena slab A  [decode output, strided]
               stbi_image_free(stbi_buf)
    │
    ▼
stage3_bitdepth: modifies slab A in-place
    │
    ▼
stage4_median:  reads slab A (const)
               ──median_3x3──► arena slab B  [filter output]
               slab A is now dead (arena owns it, reclaimed on next reset())
    │
    ▼
stage5_jpeg_roundtrip:
    ├── (if strided) pack A→ arena slab C  [packed input for stbi_write]
    ├── encode slab B ──stbi_write_jpg_to_func──► arena slab D  [JPEG bytes]
    ├── stbi_load_from_memory(slab D) → stbi heap buf
    └── ──memcpy──► arena slab E  [final output, strided]
         stbi_image_free(stbi heap buf)
    │
    ▼
SanitizeResult.image.data = slab E  ← valid until next arena_.reset()
```

Slabs B, C, D are dead after `run()` returns but remain in the arena until the next `reset()`. This is correct: arena is a monotonic bump allocator and the caller consumes `SanitizeResult.image` before calling `run()` again.

---

## 7. Thread Safety Model

From DECISIONS.md §5 and the arena design:

- `Pipeline` owns one `Arena`. `Arena` has no synchronization.
- `Pipeline::run()` is NOT thread-safe.
- Correct usage: one `Pipeline` per thread, or one `Pipeline` with external mutual exclusion.
- For a web server with N worker threads: allocate N `Pipeline` instances (e.g., in a `thread_local` or a fixed-size pool).

```cpp
// Correct multi-threaded usage pattern:
thread_local Pipeline tl_pipeline{};  // one per OS thread

SanitizeResult safe_sanitize(const uint8_t* data, size_t len) {
    return tl_pipeline.run(data, len);
}
```

---

## 8. Error Handling Contract

| Condition | Behavior |
|---|---|
| `input == nullptr` | `stage0_validate` returns false, `SanitizeError::BadMagicBytes` |
| `len == 0` | Same as above |
| magic bytes not JPEG/PNG | `SanitizeError::BadMagicBytes` |
| `width/height > max_width/height` | `SanitizeError::DimensionsTooLarge` |
| `len > max_file_bytes` | `SanitizeError::FileTooLarge` |
| stb_image decode failure | `SanitizeError::DecodeFailed` |
| arena OOM (system OOM) | `Arena::allocate` throws `std::bad_alloc`; pipeline does NOT catch it — propagates to caller |
| stbi_write_jpg failure | `SanitizeError::EncodeFailed` |

The decision to let `std::bad_alloc` propagate (rather than catching and returning `SanitizeError::OomFailed`) is intentional: OOM in a 32 MB arena means the system is critically short on memory, and silently returning an error is worse than a crash that surfaces the root cause.

---

## 9. `aligned_stride` Utility (shared across headers)

```cpp
// include/pixmask/image_view.h  (add to namespace pixmask)
static inline uint32_t aligned_stride(uint32_t width, uint32_t channels,
                                      uint32_t alignment = 64) noexcept {
    uint32_t raw = width * channels;
    return (raw + alignment - 1) & ~(alignment - 1);
}
```

---

## 10. Checklist Before Implementing

- [ ] `arena.h` — implement and unit-test `allocate`, `reset`, multi-block growth
- [ ] `image_view.h` — trivially copyable (static_assert), all helpers noexcept
- [ ] `validate.h/cpp` — magic byte table (JPEG: `FF D8 FF`, PNG: `89 50 4E 47 0D 0A 1A 0A`)
- [ ] `decode.h/cpp` — `#define STBI_ONLY_JPEG` + `#define STBI_ONLY_PNG` before including stb_image.h
- [ ] `bitdepth.h/cpp` — Highway SIMD dispatch, in-place, all depths 1-8
- [ ] `median.h/cpp` — Bose-Nelson 19-comparator network, separate src/dst buffers
- [ ] `jpeg_roundtrip.h/cpp` — `arc4random_uniform` for QF selection, write callback
- [ ] `pipeline.h/cpp` — assembles the above, owns Arena, non-copyable
- [ ] ASan + UBSan on all unit tests (check arena alignment, stride arithmetic)

---

*This file is implementation reference only. DECISIONS.md remains the authoritative spec.*
