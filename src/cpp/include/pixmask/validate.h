#pragma once

#include <cstddef>
#include <cstdint>

#include "pixmask/types.h"

namespace pixmask {

// --- Limits (must match SanitizeOptions defaults in types.h) -----------------

inline constexpr uint32_t kMaxDimension    = 8192;
inline constexpr uint64_t kMaxFileSizeBytes = 50ULL * 1024 * 1024; // 50 MB
inline constexpr uint32_t kMaxDecompRatio  = 100;
// Overflow-safe pixel count ceiling: 8192 * 8192 * 4 channels = 268 435 456
inline constexpr uint64_t kMaxRawBytes     =
    static_cast<uint64_t>(kMaxDimension) * kMaxDimension * 4;

// --- Format ------------------------------------------------------------------

enum class ImageFormat : uint8_t {
    Unknown = 0,
    PNG,
    JPEG,
    WebP,   // Detected but REJECTED in v0.1 (not in accepted set)
};

// --- Error codes -------------------------------------------------------------

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

// --- Validation result -------------------------------------------------------

struct ValidationResult {
    ValidationError error   = ValidationError::Ok;
    ImageFormat     format  = ImageFormat::Unknown;
    uint32_t        width   = 0;
    uint32_t        height  = 0;
    uint8_t         channels = 0;  // 0 = unknown at this stage

    [[nodiscard]] bool ok() const noexcept { return error == ValidationError::Ok; }
};

// --- Public entry point ------------------------------------------------------

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

// --- Sub-functions (usable from tests / fuzz targets) ------------------------

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
