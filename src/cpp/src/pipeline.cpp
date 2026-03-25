// pipeline.cpp — Orchestrates the pixmask sanitization pipeline.
// Stage order: validate → decode → bit-depth → median → JPEG roundtrip.
#include "pixmask/pipeline.h"
#include "pixmask/validate.h"
#include "pixmask/decode.h"
#include "pixmask/bitdepth.h"
#include "pixmask/median.h"
#include "pixmask/jpeg_roundtrip.h"

namespace pixmask {

// ---------------------------------------------------------------------------
// Construction / move
// ---------------------------------------------------------------------------

Pipeline::Pipeline(const SanitizeOptions& opts)
    : opts_(opts)
    , arena_(Arena::kDefaultBlockSize)
{}

Pipeline::~Pipeline() = default;

Pipeline::Pipeline(Pipeline&& other) noexcept
    : opts_(other.opts_)
    , arena_(std::move(other.arena_))
{}

Pipeline& Pipeline::operator=(Pipeline&& other) noexcept {
    if (this != &other) {
        opts_  = other.opts_;
        arena_ = std::move(other.arena_);
    }
    return *this;
}

// ---------------------------------------------------------------------------
// Helper: build a failure result from a SanitizeError + literal message.
// ---------------------------------------------------------------------------

static SanitizeResult make_error(SanitizeError code, const char* msg) noexcept {
    SanitizeResult r{};
    r.success       = false;
    r.error_code    = code;
    r.error_message = msg;
    return r;
}

// ---------------------------------------------------------------------------
// Map ValidationError → SanitizeError for the caller.
// ---------------------------------------------------------------------------

static SanitizeError map_validation_error(ValidationError ve) noexcept {
    switch (ve) {
        case ValidationError::Ok:
            return SanitizeError::Ok;
        case ValidationError::NullInput:
        case ValidationError::FileTooSmall:
        case ValidationError::UnknownFormat:
            return SanitizeError::BadMagicBytes;
        case ValidationError::FileTooLarge:
            return SanitizeError::FileTooLarge;
        case ValidationError::UnsupportedFormat:
            return SanitizeError::UnsupportedFormat;
        case ValidationError::WidthTooLarge:
        case ValidationError::HeightTooLarge:
        case ValidationError::ZeroDimension:
        case ValidationError::DimensionReadFailed:
            return SanitizeError::DimensionsTooLarge;
        case ValidationError::PixelCountOverflow:
        case ValidationError::DecompRatioTooHigh:
            return SanitizeError::DecompRatioBreach;
        default:
            return SanitizeError::BadMagicBytes;
    }
}

// ---------------------------------------------------------------------------
// Pipeline::sanitize — the main entry point.
// ---------------------------------------------------------------------------

SanitizeResult Pipeline::sanitize(const uint8_t* data, size_t len) {
    // Stage 0: Reset arena — all previous pointers invalidated.
    arena_.reset();

    // Stage 0: Validate input (magic bytes, dimensions, file size, decomp ratio).
    ValidationResult vr = validate_input(
        data, len,
        opts_.max_width, opts_.max_height,
        opts_.max_file_bytes, opts_.max_decomp_ratio
    );
    if (!vr.ok()) {
        return make_error(
            map_validation_error(vr.error),
            validation_error_message(vr.error)
        );
    }

    // Stage 1: Decode raw bytes → pixel buffer in arena.
    ImageView image = decode_image(data, len, arena_);
    if (!image.is_valid()) {
        return make_error(SanitizeError::DecodeFailed,
                          "image decoding failed");
    }

    // Stage 3: Bit-depth reduction (in-place).
    reduce_bit_depth(image, opts_.bit_depth);

    // Stage 4: Median 3x3 filter (allocates new buffer from arena).
    image = median_filter_3x3(image, arena_);
    if (!image.is_valid()) {
        return make_error(SanitizeError::OomFailed,
                          "median filter allocation failed");
    }

    // Stage 5: JPEG roundtrip (encode + decode, allocates from arena).
    image = jpeg_roundtrip(image, arena_,
                           opts_.jpeg_quality_lo,
                           opts_.jpeg_quality_hi);
    if (!image.is_valid()) {
        return make_error(SanitizeError::EncodeFailed,
                          "JPEG roundtrip failed");
    }

    // Success — image is owned by arena_, valid until next sanitize() call.
    SanitizeResult r{};
    r.image         = image;
    r.success       = true;
    r.error_code    = SanitizeError::Ok;
    r.error_message = nullptr;
    return r;
}

// ---------------------------------------------------------------------------
// Free function convenience wrapper.
// ---------------------------------------------------------------------------

SanitizeResult sanitize(const uint8_t* data, size_t len,
                        const SanitizeOptions& opts) {
    Pipeline pipeline(opts);
    return pipeline.sanitize(data, len);
}

} // namespace pixmask
