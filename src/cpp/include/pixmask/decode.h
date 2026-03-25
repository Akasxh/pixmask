#pragma once

#include <cstddef>
#include <cstdint>

#include "pixmask/arena.h"
#include "pixmask/types.h"

namespace pixmask {

// Decode raw image bytes (JPEG or PNG) into an RGB pixel buffer.
// Pixels are stored in `arena`; the returned ImageView points to arena memory.
// Forces 3-channel (RGB) output regardless of source format.
// Stride is width*3 aligned up to 64 bytes for SIMD.
// On failure, returns an ImageView with data == nullptr.
// `error_out` (if non-null) receives a static error string on failure.
ImageView decode_image(
    const uint8_t* data,
    size_t         len,
    Arena&         arena,
    const char**   error_out = nullptr
) noexcept;

} // namespace pixmask
