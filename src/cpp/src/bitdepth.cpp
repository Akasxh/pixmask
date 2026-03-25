// Bit-depth reduction — Highway dynamic dispatch file.
// Re-includes bitdepth-inl.h once per enabled SIMD target.

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "bitdepth-inl.h"
#include "hwy/foreach_target.h"  // re-includes this file per target
#include "bitdepth-inl.h"

// ---- Compiled exactly once ----
#if HWY_ONCE

#include "pixmask/bitdepth.h"

namespace pixmask {

HWY_EXPORT(ReduceBitDepthRow);

void reduce_bit_depth(ImageView& image, uint8_t bits) {
    if (bits >= 8 || bits == 0) {
        return;  // 8-bit is identity, 0 is invalid — both no-ops
    }

    const size_t row_pixels =
        static_cast<size_t>(image.width) * image.channels;

    for (uint32_t y = 0; y < image.height; ++y) {
        uint8_t* row = image.data + static_cast<size_t>(y) * image.stride;
        HWY_DYNAMIC_DISPATCH(ReduceBitDepthRow)(row, row_pixels, bits);
    }
}

}  // namespace pixmask

#endif  // HWY_ONCE
