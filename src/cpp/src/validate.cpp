#include "pixmask/validate.h"

#include <cstring>   // memcmp
#include <limits>

namespace pixmask {

// --- Magic-byte constants ----------------------------------------------------

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

// --- Error messages ----------------------------------------------------------

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

// --- Internal helpers --------------------------------------------------------

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

// --- 1. Format detection -----------------------------------------------------

ImageFormat detect_format(const uint8_t* data, size_t len) noexcept {
    if (data == nullptr || len < 2) {
        return ImageFormat::Unknown;
    }

    // JPEG: SOI = 0xFF 0xD8
    if (len >= 2 && std::memcmp(data, kJpegSig, 2) == 0) {
        return ImageFormat::JPEG;
    }

    // PNG: 8-byte signature
    if (len >= 8 && std::memcmp(data, kPngSig, 8) == 0) {
        return ImageFormat::PNG;
    }

    // WebP: RIFF????WEBP
    if (len >= kWebpMinLen &&
        std::memcmp(data,     kRiffSig, 4) == 0 &&
        std::memcmp(data + 8, kWebpSig, 4) == 0)
    {
        return ImageFormat::WebP;
    }

    return ImageFormat::Unknown;
}

// --- 2. Dimension extraction -------------------------------------------------

// PNG: IHDR is always the first chunk.
// Layout: sig(8) + chunk_len(4) + "IHDR"(4) + width(4) + height(4) + ...
// Minimum bytes required to reach height end: 8 + 4 + 4 + 4 + 4 = 24
static constexpr size_t kPngMinForDims = 24;

// JPEG: scan forward for SOF0 (0xFF 0xC0), SOF1 (0xFF 0xC1), or SOF2 (0xFF 0xC2).
// SOF structure: marker(2) + length(2) + precision(1) + height(2) + width(2)
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
    if (std::memcmp(data + 12, kIHDR, 4) != 0) {
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
            // Marker sync lost -- malformed JPEG
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

// WebP: VP8 (lossy), VP8L (lossless), VP8X (extended).
static ValidationError webp_dimensions(
    const uint8_t* data, size_t len,
    uint32_t& out_w, uint32_t& out_h
) noexcept {
    if (len < 30) return ValidationError::DimensionReadFailed;

    // Chunk FourCC is at bytes 12-15
    const uint8_t* fourcc = data + 12;

    static constexpr uint8_t kVP8 [4] = { 0x56, 0x50, 0x38, 0x20 }; // "VP8 "
    static constexpr uint8_t kVP8L[4] = { 0x56, 0x50, 0x38, 0x4C }; // "VP8L"
    static constexpr uint8_t kVP8X[4] = { 0x56, 0x50, 0x38, 0x58 }; // "VP8X"

    if (std::memcmp(fourcc, kVP8, 4) == 0) {
        // VP8 lossy: width/height at bytes 26-29, little-endian, 14-bit + scale
        if (len < 30) return ValidationError::DimensionReadFailed;
        const uint16_t w_raw = static_cast<uint16_t>(data[26]) |
                               (static_cast<uint16_t>(data[27]) << 8);
        const uint16_t h_raw = static_cast<uint16_t>(data[28]) |
                               (static_cast<uint16_t>(data[29]) << 8);
        out_w = (w_raw & 0x3FFF) + 1;
        out_h = (h_raw & 0x3FFF) + 1;
        return ValidationError::Ok;
    }

    if (std::memcmp(fourcc, kVP8L, 4) == 0) {
        // VP8L lossless: signature 0x2F at offset 20, packed uint32 at 21
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

    if (std::memcmp(fourcc, kVP8X, 4) == 0) {
        // VP8X extended: canvas width-1 at bytes 24-26 (24-bit LE),
        //                canvas height-1 at bytes 27-29 (24-bit LE)
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
    static constexpr uint64_t kU64Max = std::numeric_limits<uint64_t>::max();
    const uint64_t w64 = static_cast<uint64_t>(out_w);
    const uint64_t h64 = static_cast<uint64_t>(out_h);
    if (w64 > kU64Max / h64) return ValidationError::PixelCountOverflow;
    const uint64_t pixel_count = w64 * h64;
    if (pixel_count > kU64Max / 4) return ValidationError::PixelCountOverflow;

    return ValidationError::Ok;
}

// --- 3. Decompression ratio estimation ---------------------------------------

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

    // Use 3 channels (conservative underestimate -- avoids false positives).
    const uint64_t raw_estimate =
        static_cast<uint64_t>(width) * height * 3;

    // Avoid division: ratio > max_ratio iff raw_estimate > max_ratio * len.
    static constexpr uint64_t kU64Max = std::numeric_limits<uint64_t>::max();
    const uint64_t len64 = static_cast<uint64_t>(len);
    if (len64 > kU64Max / max_ratio) {
        // max_ratio * len overflows -- raw_estimate can't possibly be larger.
        return ValidationError::Ok;
    }
    if (raw_estimate > static_cast<uint64_t>(max_ratio) * len64) {
        return ValidationError::DecompRatioTooHigh;
    }
    return ValidationError::Ok;
}

// --- 4. Top-level validation -------------------------------------------------

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
            // Detected but not accepted in v0.1 -- JPEG + PNG only.
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
