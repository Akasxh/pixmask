// pixmask/types.h — Core types for the pixmask sanitization pipeline.
// See architecture/DECISIONS.md section 5 for authoritative spec.
#pragma once
#ifndef PIXMASK_TYPES_H
#define PIXMASK_TYPES_H

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace pixmask {

// ---------------------------------------------------------------------------
// ImageView — Non-owning view over a pixel buffer (owned by Arena).
// Trivially copyable. Cheap to pass by value through pipeline stages.
// ---------------------------------------------------------------------------
struct ImageView {
    uint8_t* data     = nullptr;
    uint32_t width    = 0;
    uint32_t height   = 0;
    uint32_t channels = 0;  // 1 (gray), 3 (RGB), 4 (RGBA)
    uint32_t stride   = 0;  // bytes per row; stride >= width * channels

    // Bytes in one row as stored (includes padding).
    [[nodiscard]] constexpr size_t row_bytes() const noexcept {
        return stride;
    }

    // Total bytes of the pixel buffer (height * stride).
    [[nodiscard]] constexpr size_t total_bytes() const noexcept {
        return static_cast<size_t>(height) * stride;
    }

    // Pointer to the start of row y. No bounds check.
    [[nodiscard]] uint8_t* row(uint32_t y) noexcept {
        return data + static_cast<size_t>(y) * stride;
    }
    [[nodiscard]] const uint8_t* row(uint32_t y) const noexcept {
        return data + static_cast<size_t>(y) * stride;
    }

    // Pointer to pixel (x, y). No bounds check.
    [[nodiscard]] uint8_t* pixel(uint32_t x, uint32_t y) noexcept {
        return row(y) + static_cast<size_t>(x) * channels;
    }
    [[nodiscard]] const uint8_t* pixel(uint32_t x, uint32_t y) const noexcept {
        return row(y) + static_cast<size_t>(x) * channels;
    }

    // True if this view references a plausibly valid image.
    [[nodiscard]] bool is_valid() const noexcept {
        return data != nullptr
            && width > 0
            && height > 0
            && (channels == 1 || channels == 3 || channels == 4)
            && stride >= width * channels;
    }
};

static_assert(std::is_trivially_copyable_v<ImageView>,
              "ImageView must be trivially copyable");

// Align stride to `alignment` bytes for SIMD-friendly row starts.
[[nodiscard]] inline uint32_t aligned_stride(uint32_t width,
                                             uint32_t channels,
                                             uint32_t alignment = 64) noexcept {
    uint32_t raw = width * channels;
    return (raw + alignment - 1) & ~(alignment - 1);
}

// ---------------------------------------------------------------------------
// SanitizeOptions — Per-call configuration with sane defaults.
// ---------------------------------------------------------------------------
struct SanitizeOptions {
    uint8_t  bit_depth        = 5;             // 1-8
    uint8_t  median_radius    = 1;             // kernel = 2r+1; 1 = 3x3
    uint8_t  jpeg_quality_lo  = 70;
    uint8_t  jpeg_quality_hi  = 85;
    uint32_t max_width        = 8192;
    uint32_t max_height       = 8192;
    uint64_t max_file_bytes   = 50ULL << 20;   // 50 MB
    uint32_t max_decomp_ratio = 100;
};

// ---------------------------------------------------------------------------
// SanitizeError — Error codes returned in SanitizeResult.
// ---------------------------------------------------------------------------
enum class SanitizeError : uint32_t {
    Ok                 = 0,
    BadMagicBytes      = 1,
    DimensionsTooLarge = 2,
    FileTooLarge       = 3,
    DecompRatioBreach  = 4,
    UnsupportedFormat  = 5,
    DecodeFailed       = 6,
    EncodeFailed       = 7,
    OomFailed          = 8,
};

// ---------------------------------------------------------------------------
// SanitizeResult — Output of the sanitization pipeline.
// ---------------------------------------------------------------------------
struct SanitizeResult {
    ImageView     image;
    bool          success       = false;
    SanitizeError error_code    = SanitizeError::Ok;
    const char*   error_message = nullptr;
};

} // namespace pixmask

#endif // PIXMASK_TYPES_H
