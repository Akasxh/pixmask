// Highway SIMD implementation for bit-depth reduction.
// This file uses the -inl.h pattern: it is re-included once per SIMD target
// by foreach_target.h via the dispatch file (bitdepth.cpp).

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace pixmask {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Process a contiguous row of uint8 pixels, applying bit-depth mask in-place.
// kShift = 8 - target_bits (compile-time constant for SIMD).
template <int kShift>
void ReduceBitDepthRowImpl(uint8_t* HWY_RESTRICT row, size_t count) {
    const hn::ScalableTag<uint8_t> d;
    const size_t N = hn::Lanes(d);
    const auto mask_vec = hn::Set(d, static_cast<uint8_t>(0xFF << kShift));

    size_t i = 0;
    // Main SIMD loop: full vectors
    for (; i + N <= count; i += N) {
        auto v = hn::LoadU(d, row + i);
        hn::StoreU(hn::And(v, mask_vec), d, row + i);
    }

    // Tail: scalar fallback to avoid OOB reads with SIMD loads.
    // MaskedLoad on AVX2 still reads a full 32-byte vector internally,
    // which is an OOB read if the buffer is shorter than the vector width.
    const uint8_t scalar_mask = static_cast<uint8_t>(0xFF << kShift);
    for (; i < count; ++i) {
        row[i] &= scalar_mask;
    }
}

// Dispatch entry point called per-row. Routes to the correct template
// instantiation based on runtime target_bits value.
void ReduceBitDepthRow(uint8_t* HWY_RESTRICT row, size_t count,
                       int target_bits) {
    const int shift = 8 - target_bits;
    switch (shift) {
        case 0: /* 8-bit: no-op */ break;
        case 1: ReduceBitDepthRowImpl<1>(row, count); break;
        case 2: ReduceBitDepthRowImpl<2>(row, count); break;
        case 3: ReduceBitDepthRowImpl<3>(row, count); break;
        case 4: ReduceBitDepthRowImpl<4>(row, count); break;
        case 5: ReduceBitDepthRowImpl<5>(row, count); break;
        case 6: ReduceBitDepthRowImpl<6>(row, count); break;
        case 7: ReduceBitDepthRowImpl<7>(row, count); break;
        default: break;
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace pixmask
HWY_AFTER_NAMESPACE();
