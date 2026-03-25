// pixmask/jpeg_roundtrip.h — Stage 5: JPEG encode+decode roundtrip.
// Destroys steganographic payloads and adversarial perturbations via
// lossy DCT quantization with a randomized quality factor.
#pragma once
#ifndef PIXMASK_JPEG_ROUNDTRIP_H
#define PIXMASK_JPEG_ROUNDTRIP_H

#include "pixmask/types.h"
#include "pixmask/arena.h"

namespace pixmask {

// JPEG encode then decode roundtrip with randomized quality factor.
// Quality is randomly chosen from [quality_lo, quality_hi] using OS entropy.
// Allocates output from arena.
//
// On failure, returns an ImageView with data == nullptr and sets *err_out
// to a static error string (do not free). err_out may be nullptr if the
// caller doesn't need the message.
ImageView jpeg_roundtrip(const ImageView& input, Arena& arena,
                         uint8_t quality_lo = 70, uint8_t quality_hi = 85,
                         const char** err_out = nullptr) noexcept;

} // namespace pixmask

#endif // PIXMASK_JPEG_ROUNDTRIP_H
