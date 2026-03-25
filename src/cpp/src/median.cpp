// 3x3 median filter — Highway dynamic dispatch file.
// Re-includes median-inl.h once per enabled SIMD target.

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "median-inl.h"
#include "hwy/foreach_target.h"  // re-includes this file per target
#include "median-inl.h"

// ---- Compiled exactly once ----
#if HWY_ONCE

#include "pixmask/median.h"

namespace pixmask {

HWY_EXPORT(Median3x3GrayImpl);
HWY_EXPORT(Median3x3MultiImpl);

ImageView median_filter_3x3(const ImageView& input, Arena& arena) {
    if (!input.is_valid()) {
        return {};
    }

    // Allocate output buffer with 64-byte aligned stride.
    const uint32_t out_stride = aligned_stride(input.width, input.channels, 64);
    const size_t total = static_cast<size_t>(input.height) * out_stride;
    auto* out_data = arena.allocate_array<uint8_t>(total);

    ImageView output;
    output.data     = out_data;
    output.width    = input.width;
    output.height   = input.height;
    output.channels = input.channels;
    output.stride   = out_stride;

    if (input.channels == 1) {
        HWY_DYNAMIC_DISPATCH(Median3x3GrayImpl)(
            input.data, output.data,
            input.width, input.height,
            input.stride, output.stride);
    } else {
        // RGB (3) or RGBA (4): per-channel scalar path.
        HWY_DYNAMIC_DISPATCH(Median3x3MultiImpl)(
            input.data, output.data,
            input.width, input.height,
            input.channels,
            input.stride, output.stride);
    }

    return output;
}

}  // namespace pixmask

#endif  // HWY_ONCE
