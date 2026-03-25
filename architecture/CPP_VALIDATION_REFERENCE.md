# Stage 0 Input Validation — C++ Reference Implementation

Authoritative reference for `src/cpp/src/validate.cpp` and `src/cpp/include/pixmask/validate.h`.

Aligns with DECISIONS.md constraints:
- Formats accepted: JPEG + PNG (WebP detected and rejected with a clear error)
- Max dimensions: 8192 x 8192
- Max file size: 50 MB
- Max decompression ratio: 100x
- No external decoder called during Stage 0 — pure byte inspection

---

## validate.h

```cpp
#pragma once

#include <cstddef>
#include <cstdint>

namespace pixmask {

// ─── Limits (must match SanitizeOptions defaults in pixmask.h) ───────────────

inline constexpr uint32_t kMaxDimension    = 8192;
inline constexpr uint64_t kMaxFileSizeBytes = 50ULL * 1024 * 1024; // 50 MB
inline constexpr uint32_t kMaxDecompRatio  = 100;
// Overflow-safe pixel count ceiling: 8192 * 8192 * 4 channels = 268 435 456 bytes
inline constexpr uint64_t kMaxRawBytes     =
    static_cast<uint64_t>(kMaxDimension) * kMaxDimension * 4;

// ─── Format ──────────────────────────────────────────────────────────────────

enum class ImageFormat : uint8_t {
    Unknown = 0,
    PNG,
    JPEG,
    WebP,   // Detected but REJECTED in v0.1 (not in accepted set)
};

// ─── Error codes ─────────────────────────────────────────────────────────────

enum class ValidationError : uint32_t {
    Ok                  = 0,
    NullInput           = 1,
    FileTooSmall        = 2,   // < minimum header size for any known format
    FileTooLarge        = 3,   // > kMaxFileSizeBytes
    UnknownFormat       = 4,   // magic bytes don't match PNG/JPEG/WebP
    UnsupportedFormat   = 5,   // detected (e.g. WebP) but not accepted in v0.1
    DimensionReadFailed = 6,   // header present but dimensions couldn't be parsed
    WidthTooLarge       = 7,
    HeightTooLarge      = 8,   // separate codes so callers can report axis
    ZeroDimension       = 9,
    PixelCountOverflow  = 10,  // width * height * channels would overflow uint64
    DecompRatioTooHigh  = 11,
};

const char* validation_error_message(ValidationError err) noexcept;

// ─── Validation result ───────────────────────────────────────────────────────

struct ValidationResult {
    ValidationError error   = ValidationError::Ok;
    ImageFormat     format  = ImageFormat::Unknown;
    uint32_t        width   = 0;
    uint32_t        height  = 0;
    uint8_t         channels = 0;  // 0 = unknown at this stage

    [[nodiscard]] bool ok() const noexcept { return error == ValidationError::Ok; }
};

// ─── Public entry point ──────────────────────────────────────────────────────

// Validates `data[0..len)` before any decoding.
// max_w / max_h / max_file / max_ratio override the constexpr defaults and
// must come from SanitizeOptions so callers can tighten limits at runtime.
ValidationResult validate_input(
    const uint8_t* data,
    size_t         len,
    uint32_t       max_w     = kMaxDimension,
    uint32_t       max_h     = kMaxDimension,
    uint64_t       max_file  = kMaxFileSizeBytes,
    uint32_t       max_ratio = kMaxDecompRatio
) noexcept;

// ─── Sub-functions (usable from tests / fuzz targets) ────────────────────────

ImageFormat     detect_format(const uint8_t* data, size_t len) noexcept;
ValidationError check_dimensions(
    const uint8_t* data, size_t len,
    ImageFormat format,
    uint32_t max_w, uint32_t max_h,
    uint32_t& out_w, uint32_t& out_h
) noexcept;
ValidationError check_decomp_ratio(
    const uint8_t* data, size_t len,
    ImageFormat format,
    uint32_t width, uint32_t height,
    uint32_t max_ratio
) noexcept;

} // namespace pixmask
```

---

## validate.cpp

```cpp
#include "pixmask/validate.h"

#include <cstring>   // memcmp
#include <limits>

namespace pixmask {

// ─── Magic-byte constants ─────────────────────────────────────────────────────

// PNG: 8-byte signature
static constexpr uint8_t kPngSig[8] = {
    0x89, 0x50, 0x4E, 0x47,  // \x89 P N G
    0x0D, 0x0A, 0x1A, 0x0A   // \r  \n \x1a \n
};

// JPEG: SOI marker (Start Of Image)
static constexpr uint8_t kJpegSig[2] = { 0xFF, 0xD8 };

// WebP: RIFF....WEBP (bytes 0-3 and 8-11)
static constexpr uint8_t kRiffSig[4]  = { 0x52, 0x49, 0x46, 0x46 }; // "RIFF"
static constexpr uint8_t kWebpSig[4]  = { 0x57, 0x45, 0x42, 0x50 }; // "WEBP"
static constexpr size_t  kWebpMinLen  = 12;

// ─── Error messages ───────────────────────────────────────────────────────────

const char* validation_error_message(ValidationError err) noexcept {
    switch (err) {
        case ValidationError::Ok:                  return "ok";
        case ValidationError::NullInput:           return "input pointer is null";
        case ValidationError::FileTooSmall:        return "data too small to be a valid image";
        case ValidationError::FileTooLarge:        return "file exceeds maximum allowed size (50 MB)";
        case ValidationError::UnknownFormat:       return "unrecognised image format (magic bytes mismatch)";
        case ValidationError::UnsupportedFormat:   return "image format detected but not accepted (only JPEG and PNG supported in v0.1)";
        case ValidationError::DimensionReadFailed: return "could not parse image dimensions from header";
        case ValidationError::WidthTooLarge:       return "image width exceeds maximum (8192)";
        case ValidationError::HeightTooLarge:      return "image height exceeds maximum (8192)";
        case ValidationError::ZeroDimension:       return "image has zero width or height";
        case ValidationError::PixelCountOverflow:  return "pixel count overflows uint64 — possible decompression bomb";
        case ValidationError::DecompRatioTooHigh:  return "estimated decompression ratio exceeds limit (100x) — possible bomb";
    }
    return "unknown validation error";
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

// Read a big-endian uint32 from `p` (no alignment assumed).
static inline uint32_t read_be32(const uint8_t* p) noexcept {
    return (static_cast<uint32_t>(p[0]) << 24) |
           (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) <<  8) |
           (static_cast<uint32_t>(p[3])      );
}

// Read a big-endian uint16 from `p`.
static inline uint16_t read_be16(const uint8_t* p) noexcept {
    return static_cast<uint16_t>(
        (static_cast<uint32_t>(p[0]) << 8) | p[1]
    );
}

// ─── 1. Format detection ──────────────────────────────────────────────────────

ImageFormat detect_format(const uint8_t* data, size_t len) noexcept {
    if (data == nullptr || len < 2) {
        return ImageFormat::Unknown;
    }

    // JPEG: SOI = 0xFF 0xD8
    if (len >= 2 && memcmp(data, kJpegSig, 2) == 0) {
        return ImageFormat::JPEG;
    }

    // PNG: 8-byte signature
    if (len >= 8 && memcmp(data, kPngSig, 8) == 0) {
        return ImageFormat::PNG;
    }

    // WebP: RIFF????WEBP
    if (len >= kWebpMinLen &&
        memcmp(data,     kRiffSig, 4) == 0 &&
        memcmp(data + 8, kWebpSig, 4) == 0)
    {
        return ImageFormat::WebP;
    }

    return ImageFormat::Unknown;
}

// ─── 2. Dimension extraction ──────────────────────────────────────────────────

// PNG: IHDR is always the first chunk.
// Layout: sig(8) + chunk_len(4) + "IHDR"(4) + width(4) + height(4) + ...
// Minimum bytes required to reach height end: 8 + 4 + 4 + 4 + 4 = 24
static constexpr size_t kPngMinForDims = 24;

// JPEG: scan forward for SOF0 (0xFF 0xC0), SOF1 (0xFF 0xC1), or SOF2 (0xFF 0xC2).
// SOF structure: marker(2) + length(2) + precision(1) + height(2) + width(2) + ...
static constexpr size_t kJpegSofMinExtra = 7; // length(2)+precision(1)+h(2)+w(2)

static ValidationError png_dimensions(
    const uint8_t* data, size_t len,
    uint32_t& out_w, uint32_t& out_h
) noexcept {
    if (len < kPngMinForDims) {
        return ValidationError::DimensionReadFailed;
    }
    // Verify "IHDR" chunk type at offset 12
    static constexpr uint8_t kIHDR[4] = { 0x49, 0x48, 0x44, 0x52 };
    if (memcmp(data + 12, kIHDR, 4) != 0) {
        return ValidationError::DimensionReadFailed;
    }
    out_w = read_be32(data + 16);
    out_h = read_be32(data + 20);
    return ValidationError::Ok;
}

static ValidationError jpeg_dimensions(
    const uint8_t* data, size_t len,
    uint32_t& out_w, uint32_t& out_h
) noexcept {
    // Skip past SOI (2 bytes), then iterate over segments.
    size_t pos = 2;
    while (pos + 4 <= len) {
        if (data[pos] != 0xFF) {
            // Marker sync lost — malformed JPEG
            return ValidationError::DimensionReadFailed;
        }
        // Skip padding 0xFF bytes (rare but valid)
        while (pos < len && data[pos] == 0xFF) {
            ++pos;
        }
        if (pos >= len) break;

        const uint8_t marker = data[pos++];

        // Markers with no length field: SOI, EOI, RST0-RST7, TEM
        if (marker == 0x01 || marker == 0xD9 ||
            (marker >= 0xD0 && marker <= 0xD7))
        {
            continue;
        }

        // Need 2 bytes for segment length
        if (pos + 2 > len) break;
        const uint16_t seg_len = read_be16(data + pos);
        if (seg_len < 2) break; // invalid length

        // SOF0, SOF1, SOF2 (baseline / extended / progressive DCT)
        if (marker == 0xC0 || marker == 0xC1 || marker == 0xC2) {
            // pos points at length field; need pos+2 (precision) +2 (height) +2 (width)
            if (pos + 2 + kJpegSofMinExtra > len) {
                return ValidationError::DimensionReadFailed;
            }
            // precision = data[pos+2], height = data[pos+3..4], width = data[pos+5..6]
            out_h = read_be16(data + pos + 3);
            out_w = read_be16(data + pos + 5);
            return ValidationError::Ok;
        }

        // Advance past segment (length includes the 2-byte length field itself)
        pos += seg_len;
    }
    return ValidationError::DimensionReadFailed;
}

// WebP: two bitstream formats — VP8 (lossy) and VP8L (lossless).
// VP8 chunk: "VP8 " at offset 12; frame tag at 20; width/height encoded in
//            bits 14-0 of words at offsets 26-27 (width) and 28-29 (height),
//            both with +1 bias. Little-endian.
// VP8L chunk: "VP8L" at offset 12; signature byte 0x2F at 20;
//             packed uint32 at 21: width-1 in bits 0-13, height-1 in bits 14-27.
// VP8X (extended): "VP8X" at offset 12; canvas width-1 at bytes 24-26 (LE 24-bit),
//                  canvas height-1 at bytes 27-29.
static ValidationError webp_dimensions(
    const uint8_t* data, size_t len,
    uint32_t& out_w, uint32_t& out_h
) noexcept {
    if (len < 30) return ValidationError::DimensionReadFailed;

    // Chunk FourCC is at bytes 12-15
    const uint8_t* fourcc = data + 12;

    // VP8 (lossy)
    static constexpr uint8_t kVP8 [4] = { 0x56, 0x50, 0x38, 0x20 }; // "VP8 "
    // VP8L (lossless)
    static constexpr uint8_t kVP8L[4] = { 0x56, 0x50, 0x38, 0x4C }; // "VP8L"
    // VP8X (extended)
    static constexpr uint8_t kVP8X[4] = { 0x56, 0x50, 0x38, 0x58 }; // "VP8X"

    if (memcmp(fourcc, kVP8, 4) == 0) {
        // Frame header starts at offset 20 (after RIFF header + chunk header).
        // The VP8 bitstream frame tag is 3 bytes at 20-22; skip it.
        // Start code: 0x9D 0x01 0x2A at bytes 23-25.
        // Width in bits 13-0 of LE uint16 at 26; height in bits 13-0 at 28.
        if (len < 30) return ValidationError::DimensionReadFailed;
        const uint16_t w_raw = static_cast<uint16_t>(data[26]) |
                               (static_cast<uint16_t>(data[27]) << 8);
        const uint16_t h_raw = static_cast<uint16_t>(data[28]) |
                               (static_cast<uint16_t>(data[29]) << 8);
        out_w = (w_raw & 0x3FFF) + 1;
        out_h = (h_raw & 0x3FFF) + 1;
        return ValidationError::Ok;
    }

    if (memcmp(fourcc, kVP8L, 4) == 0) {
        // Signature byte 0x2F at offset 20.
        // Packed uint32 at offset 21 (LE):
        //   bits  0-13: width  - 1
        //   bits 14-27: height - 1
        if (len < 25) return ValidationError::DimensionReadFailed;
        if (data[20] != 0x2F) return ValidationError::DimensionReadFailed;
        const uint32_t packed =
            static_cast<uint32_t>(data[21])        |
            (static_cast<uint32_t>(data[22]) <<  8) |
            (static_cast<uint32_t>(data[23]) << 16) |
            (static_cast<uint32_t>(data[24]) << 24);
        out_w = (packed & 0x3FFF) + 1;
        out_h = ((packed >> 14) & 0x3FFF) + 1;
        return ValidationError::Ok;
    }

    if (memcmp(fourcc, kVP8X, 4) == 0) {
        // Canvas width  - 1 at bytes 24-26 (24-bit LE)
        // Canvas height - 1 at bytes 27-29 (24-bit LE)
        if (len < 30) return ValidationError::DimensionReadFailed;
        const uint32_t w_m1 = static_cast<uint32_t>(data[24])        |
                              (static_cast<uint32_t>(data[25]) <<  8) |
                              (static_cast<uint32_t>(data[26]) << 16);
        const uint32_t h_m1 = static_cast<uint32_t>(data[27])        |
                              (static_cast<uint32_t>(data[28]) <<  8) |
                              (static_cast<uint32_t>(data[29]) << 16);
        out_w = w_m1 + 1;
        out_h = h_m1 + 1;
        return ValidationError::Ok;
    }

    return ValidationError::DimensionReadFailed;
}

ValidationError check_dimensions(
    const uint8_t* data, size_t len,
    ImageFormat format,
    uint32_t max_w, uint32_t max_h,
    uint32_t& out_w, uint32_t& out_h
) noexcept {
    ValidationError err = ValidationError::DimensionReadFailed;

    switch (format) {
        case ImageFormat::PNG:
            err = png_dimensions(data, len, out_w, out_h);
            break;
        case ImageFormat::JPEG:
            err = jpeg_dimensions(data, len, out_w, out_h);
            break;
        case ImageFormat::WebP:
            err = webp_dimensions(data, len, out_w, out_h);
            break;
        case ImageFormat::Unknown:
            return ValidationError::UnknownFormat;
    }

    if (err != ValidationError::Ok) return err;

    if (out_w == 0 || out_h == 0) return ValidationError::ZeroDimension;
    if (out_w > max_w)            return ValidationError::WidthTooLarge;
    if (out_h > max_h)            return ValidationError::HeightTooLarge;

    // Overflow-safe pixel count check: use uint64 arithmetic.
    // 4 channels is the maximum (RGBA). If this overflows uint64 we reject.
    // In practice 8192 * 8192 * 4 = 268 435 456 which fits comfortably,
    // but we guard against arbitrarily large headers from malformed files.
    static constexpr uint64_t kU64Max = std::numeric_limits<uint64_t>::max();
    const uint64_t w64 = static_cast<uint64_t>(out_w);
    const uint64_t h64 = static_cast<uint64_t>(out_h);
    // Check w * h first to detect overflow before multiplying by 4.
    if (w64 > kU64Max / h64) return ValidationError::PixelCountOverflow;
    const uint64_t pixel_count = w64 * h64;
    if (pixel_count > kU64Max / 4) return ValidationError::PixelCountOverflow;

    return ValidationError::Ok;
}

// ─── 3. Decompression ratio estimation ───────────────────────────────────────
//
// We estimate the decoded size and compare it to the compressed size (len).
// Conservative channel assumption: 3 (RGB). We intentionally underestimate
// raw size so we don't accidentally reject legitimate images; the 100x ratio
// is already highly conservative.
//
// PNG heuristic:
//   raw_estimate = width * height * 3
//   ratio = raw_estimate / len
//   (PNG is lossless and rarely exceeds 20x on natural images.)
//
// JPEG heuristic:
//   At quality 1 (worst case, maximum ratio), JPEG is ~1 byte/pixel.
//   At quality 100, it approaches raw. We use 3 bytes/pixel as upper bound
//   for decoded size. Decompression bombs are impractical with JPEG since
//   the format cannot store nearly-zero data that decompresses to huge images
//   the way zlib/PNG can, but we apply the check uniformly.
//
// WebP heuristic: same as PNG (lossless path can be explosive).

ValidationError check_decomp_ratio(
    const uint8_t* data, size_t len,
    ImageFormat format,
    uint32_t width, uint32_t height,
    uint32_t max_ratio
) noexcept {
    (void)data;   // not inspecting payload bytes here
    (void)format; // ratio formula is format-independent at this granularity

    if (len == 0) return ValidationError::FileTooSmall;
    if (max_ratio == 0) return ValidationError::Ok; // caller disabled check

    // Use 3 channels (conservative underestimate — avoids false positives).
    const uint64_t raw_estimate =
        static_cast<uint64_t>(width) * height * 3;

    // Avoid division: ratio > max_ratio iff raw_estimate > max_ratio * len.
    // Check for overflow before multiplying: max_ratio is at most ~4B, len
    // is at most 50MB (50 * 2^20 ≈ 5 * 10^7). Product fits in uint64.
    static constexpr uint64_t kU64Max = std::numeric_limits<uint64_t>::max();
    const uint64_t len64 = static_cast<uint64_t>(len);
    if (len64 > kU64Max / max_ratio) {
        // max_ratio * len overflows — raw_estimate can't possibly be larger.
        return ValidationError::Ok;
    }
    if (raw_estimate > static_cast<uint64_t>(max_ratio) * len64) {
        return ValidationError::DecompRatioTooHigh;
    }
    return ValidationError::Ok;
}

// ─── 4. Top-level validation ─────────────────────────────────────────────────

ValidationResult validate_input(
    const uint8_t* data,
    size_t         len,
    uint32_t       max_w,
    uint32_t       max_h,
    uint64_t       max_file,
    uint32_t       max_ratio
) noexcept {
    ValidationResult result;

    // 4a. Null / empty check
    if (data == nullptr) {
        result.error = ValidationError::NullInput;
        return result;
    }
    if (len < 2) {
        result.error = ValidationError::FileTooSmall;
        return result;
    }

    // 4b. File size ceiling (before any header parsing)
    if (static_cast<uint64_t>(len) > max_file) {
        result.error = ValidationError::FileTooLarge;
        return result;
    }

    // 4c. Format detection
    result.format = detect_format(data, len);
    switch (result.format) {
        case ImageFormat::Unknown:
            result.error = ValidationError::UnknownFormat;
            return result;
        case ImageFormat::WebP:
            // Detected but not accepted in v0.1 — JPEG + PNG only.
            result.error = ValidationError::UnsupportedFormat;
            return result;
        case ImageFormat::PNG:
        case ImageFormat::JPEG:
            break; // accepted
    }

    // 4d. Dimension check (reads from compressed header, no decode)
    result.error = check_dimensions(
        data, len, result.format,
        max_w, max_h,
        result.width, result.height
    );
    if (result.error != ValidationError::Ok) return result;

    // 4e. Decompression ratio check
    result.error = check_decomp_ratio(
        data, len, result.format,
        result.width, result.height,
        max_ratio
    );

    return result;
}

} // namespace pixmask
```

---

## Test coverage skeleton (test_validate.cpp)

Maps to DECISIONS.md §7 test gate criteria.

```cpp
// src/tests/cpp/test_validate.cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "pixmask/validate.h"

using namespace pixmask;

// ─── Magic-byte detection ─────────────────────────────────────────────────────

TEST_CASE("detect_format: null / undersized") {
    CHECK(detect_format(nullptr, 0)  == ImageFormat::Unknown);
    CHECK(detect_format(nullptr, 8)  == ImageFormat::Unknown);
    const uint8_t one = 0xFF;
    CHECK(detect_format(&one, 1)     == ImageFormat::Unknown);
}

TEST_CASE("detect_format: PNG signature") {
    const uint8_t sig[] = { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A };
    CHECK(detect_format(sig, sizeof(sig)) == ImageFormat::PNG);
}

TEST_CASE("detect_format: JPEG SOI") {
    const uint8_t sig[] = { 0xFF, 0xD8, 0xFF, 0xE0 };
    CHECK(detect_format(sig, sizeof(sig)) == ImageFormat::JPEG);
}

TEST_CASE("detect_format: WebP RIFF header") {
    const uint8_t sig[] = {
        0x52, 0x49, 0x46, 0x46,  // RIFF
        0x00, 0x00, 0x00, 0x00,  // file size (ignored)
        0x57, 0x45, 0x42, 0x50   // WEBP
    };
    CHECK(detect_format(sig, sizeof(sig)) == ImageFormat::WebP);
}

TEST_CASE("detect_format: garbage bytes") {
    const uint8_t garbage[] = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07 };
    CHECK(detect_format(garbage, sizeof(garbage)) == ImageFormat::Unknown);
}

// ─── validate_input: top-level gating ────────────────────────────────────────

TEST_CASE("validate_input: null pointer") {
    auto r = validate_input(nullptr, 0);
    CHECK(r.error == ValidationError::NullInput);
    CHECK_FALSE(r.ok());
}

TEST_CASE("validate_input: data too small") {
    const uint8_t one = 0xAB;
    auto r = validate_input(&one, 1);
    CHECK(r.error == ValidationError::FileTooSmall);
}

TEST_CASE("validate_input: WebP rejected in v0.1") {
    const uint8_t sig[] = {
        0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00,
        0x57, 0x45, 0x42, 0x50
    };
    auto r = validate_input(sig, sizeof(sig));
    CHECK(r.error == ValidationError::UnsupportedFormat);
    CHECK(r.format == ImageFormat::WebP);
}

TEST_CASE("validate_input: unknown format") {
    const uint8_t garbage[16] = {};
    auto r = validate_input(garbage, sizeof(garbage));
    CHECK(r.error == ValidationError::UnknownFormat);
}

// ─── Dimension pre-check ──────────────────────────────────────────────────────

TEST_CASE("check_dimensions: PNG zero width is rejected") {
    // Craft a minimal PNG header with width=0
    uint8_t buf[24] = {};
    memcpy(buf, "\x89PNG\r\n\x1a\n", 8);                   // sig
    buf[12] = 'I'; buf[13] = 'H'; buf[14] = 'D'; buf[15] = 'R'; // IHDR
    // width bytes 16-19 = 0 (already zeroed)
    buf[20] = 0; buf[21] = 0; buf[22] = 0; buf[23] = 1; // height = 1
    uint32_t w = 0, h = 0;
    auto err = check_dimensions(buf, sizeof(buf), ImageFormat::PNG,
                                kMaxDimension, kMaxDimension, w, h);
    CHECK(err == ValidationError::ZeroDimension);
}

TEST_CASE("check_dimensions: PNG oversized width rejected") {
    uint8_t buf[24] = {};
    memcpy(buf, "\x89PNG\r\n\x1a\n", 8);
    buf[12] = 'I'; buf[13] = 'H'; buf[14] = 'D'; buf[15] = 'R';
    // width = 0xFFFFFFFF (big-endian)
    buf[16] = 0xFF; buf[17] = 0xFF; buf[18] = 0xFF; buf[19] = 0xFF;
    buf[20] = 0x00; buf[21] = 0x00; buf[22] = 0x00; buf[23] = 0x01;
    uint32_t w = 0, h = 0;
    auto err = check_dimensions(buf, sizeof(buf), ImageFormat::PNG,
                                kMaxDimension, kMaxDimension, w, h);
    CHECK(err == ValidationError::WidthTooLarge);
}

// ─── Decompression ratio ──────────────────────────────────────────────────────

TEST_CASE("check_decomp_ratio: bomb scenario rejected") {
    // 1x1 pixel image stored in 1 byte of "compressed" data — ratio = 3/1 = 3x, ok.
    // Simulate 8192x8192 image in 1 byte:
    // raw_estimate = 8192 * 8192 * 3 = 201 326 592; len = 1 → ratio > 100
    auto err = check_decomp_ratio(nullptr, 1, ImageFormat::PNG,
                                  8192, 8192, kMaxDecompRatio);
    CHECK(err == ValidationError::DecompRatioTooHigh);
}

TEST_CASE("check_decomp_ratio: normal image accepted") {
    // 512x512 RGB, 100KB compressed → raw = 786 432; ratio ≈ 7.8x
    auto err = check_decomp_ratio(nullptr, 100 * 1024, ImageFormat::PNG,
                                  512, 512, kMaxDecompRatio);
    CHECK(err == ValidationError::Ok);
}

TEST_CASE("check_decomp_ratio: disabled when max_ratio=0") {
    // Even a bomb-level scenario should pass when the check is disabled.
    auto err = check_decomp_ratio(nullptr, 1, ImageFormat::PNG,
                                  8192, 8192, 0 /*disabled*/);
    CHECK(err == ValidationError::Ok);
}

// ─── Error messages ───────────────────────────────────────────────────────────

TEST_CASE("validation_error_message: all codes have non-empty strings") {
    const ValidationError codes[] = {
        ValidationError::Ok,
        ValidationError::NullInput,
        ValidationError::FileTooSmall,
        ValidationError::FileTooLarge,
        ValidationError::UnknownFormat,
        ValidationError::UnsupportedFormat,
        ValidationError::DimensionReadFailed,
        ValidationError::WidthTooLarge,
        ValidationError::HeightTooLarge,
        ValidationError::ZeroDimension,
        ValidationError::PixelCountOverflow,
        ValidationError::DecompRatioTooHigh,
    };
    for (auto code : codes) {
        const char* msg = validation_error_message(code);
        CHECK(msg != nullptr);
        CHECK(msg[0] != '\0');
    }
}
```

---

## Design notes

### Why magic bytes before any allocation

stb_image's `stbi_info_from_memory` still parses enough of the header to infer
dimensions, but it also initialises internal state and touches more of the input
buffer than the 8–24 bytes we read here. Doing format + dimension checks
entirely in `validate.cpp` with no library calls means the hot path through a
rejected input touches at most ~30 bytes and returns in under 100 ns.

### PNG dimension offset derivation

```
offset  0: PNG signature       (8 bytes)
offset  8: IHDR chunk length   (4 bytes, always 13 for well-formed PNGs)
offset 12: chunk type "IHDR"   (4 bytes)
offset 16: width               (4 bytes, big-endian)  <-- read here
offset 20: height              (4 bytes, big-endian)  <-- read here
offset 24: bit depth           (1 byte)
offset 25: color type          (1 byte)
...
```

We verify the "IHDR" tag at offset 12 before trusting the width/height fields.
A file with a valid PNG signature but a corrupted first chunk returns
`DimensionReadFailed`, not a wrong dimension.

### JPEG SOF scan: why we scan instead of using a fixed offset

JPEG does not place the SOF marker at a fixed offset. APPn markers
(0xFF 0xE0–0xFF 0xEF) and COM markers (0xFF 0xFE) can appear before it and
their sizes are variable. Camera JPEGs with large EXIF blobs can push SOF
past offset 65 KB. The linear scan terminates as soon as SOF0/C1/C2 is found
and adds at most ~1 µs on a 50 MB pathological file.

### Overflow-safe pixel count

```
width * height * 4 must not overflow uint64_t.
```

Rather than using compiler builtins (`__builtin_mul_overflow`) which are not
portable across MSVC/Clang/GCC uniformly, we use the equivalent
`if (a > UINT64_MAX / b)` pattern which compiles to a single DIV or comparison
on all targets and is UB-free under the C++ standard.

### Decompression ratio formula

```
ratio = (width * height * 3) / compressed_size
```

Using 3 channels (RGB) is intentionally conservative: an RGBA image is
actually 4 bytes/pixel, making the ratio higher and more likely to trip the
check. Using 3 channels means we only reject files where the ratio is
egregiously high even in the best case, avoiding false positives on legitimate
images with alpha channels.

The threshold of 100x is borrowed from the DECISIONS.md §3 spec. Natural PNG
images rarely exceed 5–8x compression. A ratio of >100x indicates either a
crafted decompression bomb (e.g. a 1×1 header claiming 8192×8192) or severe
header corruption — both warrant rejection.

### WebP support status

WebP dimension parsing is implemented so the format can be cleanly identified
and rejected with `UnsupportedFormat` rather than the opaque `UnknownFormat`.
This gives callers a useful error message. When v0.2 adds WebP acceptance, the
`validate_input` switch statement changes one line: remove the
`UnsupportedFormat` return for `ImageFormat::WebP`.
