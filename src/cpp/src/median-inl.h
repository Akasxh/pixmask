// Highway SIMD implementation for 3x3 median filter.
// This file uses the -inl.h pattern: it is re-included once per SIMD target
// by foreach_target.h via the dispatch file (median.cpp).
//
// The 19-step Bose-Nelson partial sorting network extracts the median of 9
// elements without fully sorting them. Each compare-and-swap (CAS) maps to
// Min + Max on SIMD lanes, processing N pixels per CAS pair.

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace pixmask {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// -------------------------------------------------------------------------
// CAS (compare-and-swap) for Highway vectors.
// After HWY_CAS(a, b): a = min(a, b), b = max(a, b).
// -------------------------------------------------------------------------
#define HWY_CAS(a, b)                 \
    do {                              \
        auto _lo = hn::Min((a), (b)); \
        auto _hi = hn::Max((a), (b)); \
        (a) = _lo;                    \
        (b) = _hi;                    \
    } while (0)

// -------------------------------------------------------------------------
// Scalar CAS for border/remainder fallback.
// -------------------------------------------------------------------------
static HWY_INLINE void ScalarCAS(uint8_t& a, uint8_t& b) {
    const uint8_t lo = a < b ? a : b;
    const uint8_t hi = a < b ? b : a;
    a = lo;
    b = hi;
}

// -------------------------------------------------------------------------
// Scalar median of 9 elements via 19-step partial sorting network.
// Only p4 is guaranteed correct (the median) after these 19 CAS operations.
// -------------------------------------------------------------------------
static HWY_INLINE uint8_t Median9Scalar(uint8_t p0, uint8_t p1, uint8_t p2,
                                         uint8_t p3, uint8_t p4, uint8_t p5,
                                         uint8_t p6, uint8_t p7, uint8_t p8) {
    ScalarCAS(p1, p2); ScalarCAS(p4, p5); ScalarCAS(p7, p8);
    ScalarCAS(p0, p1); ScalarCAS(p3, p4); ScalarCAS(p6, p7);
    ScalarCAS(p1, p2); ScalarCAS(p4, p5); ScalarCAS(p7, p8);
    ScalarCAS(p0, p3); ScalarCAS(p5, p8); ScalarCAS(p4, p7);
    ScalarCAS(p3, p6); ScalarCAS(p1, p4); ScalarCAS(p2, p5);
    ScalarCAS(p4, p7);
    ScalarCAS(p4, p2); ScalarCAS(p6, p4); ScalarCAS(p4, p2);
    return p4;
}

// -------------------------------------------------------------------------
// SIMD 3x3 median kernel for single-channel: processes interior pixels
// of one row. Caller provides row pointers at column 0; MedianRow reads
// offsets [i, i+1, i+2] where i iterates 0..(width-3).
//
// dst receives (width - 2) interior pixels starting at column 1.
// -------------------------------------------------------------------------
template <class D>
HWY_INLINE void MedianRow(D d,
                           const uint8_t* HWY_RESTRICT row_above,
                           const uint8_t* HWY_RESTRICT row_curr,
                           const uint8_t* HWY_RESTRICT row_below,
                           uint8_t* HWY_RESTRICT dst,
                           size_t n) {
    using V = hn::Vec<D>;
    const size_t N = hn::Lanes(d);

    size_t i = 0;
    for (; i + N <= n; i += N) {
        V p0 = hn::LoadU(d, row_above + i);
        V p1 = hn::LoadU(d, row_above + i + 1);
        V p2 = hn::LoadU(d, row_above + i + 2);
        V p3 = hn::LoadU(d, row_curr  + i);
        V p4 = hn::LoadU(d, row_curr  + i + 1);
        V p5 = hn::LoadU(d, row_curr  + i + 2);
        V p6 = hn::LoadU(d, row_below + i);
        V p7 = hn::LoadU(d, row_below + i + 1);
        V p8 = hn::LoadU(d, row_below + i + 2);

        // 19-step Bose-Nelson partial sorting network.
        HWY_CAS(p1, p2); HWY_CAS(p4, p5); HWY_CAS(p7, p8);
        HWY_CAS(p0, p1); HWY_CAS(p3, p4); HWY_CAS(p6, p7);
        HWY_CAS(p1, p2); HWY_CAS(p4, p5); HWY_CAS(p7, p8);
        HWY_CAS(p0, p3); HWY_CAS(p5, p8); HWY_CAS(p4, p7);
        HWY_CAS(p3, p6); HWY_CAS(p1, p4); HWY_CAS(p2, p5);
        HWY_CAS(p4, p7);
        HWY_CAS(p4, p2); HWY_CAS(p6, p4); HWY_CAS(p4, p2);

        hn::StoreU(p4, d, dst + i);
    }

    // Scalar remainder for < N trailing pixels.
    for (; i < n; ++i) {
        dst[i] = Median9Scalar(
            row_above[i],     row_above[i + 1], row_above[i + 2],
            row_curr[i],      row_curr[i + 1],  row_curr[i + 2],
            row_below[i],     row_below[i + 1], row_below[i + 2]);
    }
}

// -------------------------------------------------------------------------
// Full single-channel 3x3 median filter (per SIMD target).
// Border: replicate (clamp-to-edge).
// -------------------------------------------------------------------------
void Median3x3GrayImpl(const uint8_t* HWY_RESTRICT src,
                        uint8_t* HWY_RESTRICT dst,
                        uint32_t width, uint32_t height,
                        uint32_t src_stride, uint32_t dst_stride) {
    const hn::ScalableTag<uint8_t> d;

    for (uint32_t y = 0; y < height; ++y) {
        const uint32_t y0 = y == 0          ? 0          : y - 1;
        const uint32_t y2 = y == height - 1 ? height - 1 : y + 1;

        const uint8_t* ra = src + static_cast<size_t>(y0) * src_stride;
        const uint8_t* rc = src + static_cast<size_t>(y)  * src_stride;
        const uint8_t* rb = src + static_cast<size_t>(y2) * src_stride;
        uint8_t*       dr = dst + static_cast<size_t>(y)  * dst_stride;

        // Left border pixel (x=0): replicate left column.
        dr[0] = Median9Scalar(
            ra[0], ra[0], (width > 1 ? ra[1] : ra[0]),
            rc[0], rc[0], (width > 1 ? rc[1] : rc[0]),
            rb[0], rb[0], (width > 1 ? rb[1] : rb[0]));

        // Interior pixels via SIMD.
        if (width > 2) {
            MedianRow(d, ra, rc, rb, dr + 1, static_cast<size_t>(width - 2));
        }

        // Right border pixel (x = width-1): replicate right column.
        if (width > 1) {
            const uint32_t xr = width - 1;
            dr[xr] = Median9Scalar(
                ra[xr - 1], ra[xr], ra[xr],
                rc[xr - 1], rc[xr], rc[xr],
                rb[xr - 1], rb[xr], rb[xr]);
        }
    }
}

// -------------------------------------------------------------------------
// Multi-channel (RGB/RGBA) 3x3 median filter — scalar per-channel.
// For interleaved layouts, SIMD on non-contiguous channel data is not
// efficient without deinterleaving. v0.1 uses scalar; v0.2 may add
// planar SIMD path.
// -------------------------------------------------------------------------
void Median3x3MultiImpl(const uint8_t* HWY_RESTRICT src,
                         uint8_t* HWY_RESTRICT dst,
                         uint32_t width, uint32_t height,
                         uint32_t channels,
                         uint32_t src_stride, uint32_t dst_stride) {
    for (uint32_t y = 0; y < height; ++y) {
        const uint32_t y0 = y == 0          ? 0          : y - 1;
        const uint32_t y2 = y == height - 1 ? height - 1 : y + 1;

        const uint8_t* ra = src + static_cast<size_t>(y0) * src_stride;
        const uint8_t* rc = src + static_cast<size_t>(y)  * src_stride;
        const uint8_t* rb = src + static_cast<size_t>(y2) * src_stride;
        uint8_t*       dr = dst + static_cast<size_t>(y)  * dst_stride;

        for (uint32_t x = 0; x < width; ++x) {
            const uint32_t xl = x == 0         ? 0         : x - 1;
            const uint32_t xr = x == width - 1 ? width - 1 : x + 1;

            for (uint32_t c = 0; c < channels; ++c) {
                dr[x * channels + c] = Median9Scalar(
                    ra[xl * channels + c], ra[x * channels + c], ra[xr * channels + c],
                    rc[xl * channels + c], rc[x * channels + c], rc[xr * channels + c],
                    rb[xl * channels + c], rb[x * channels + c], rb[xr * channels + c]);
            }
        }
    }
}

#undef HWY_CAS

}  // namespace HWY_NAMESPACE
}  // namespace pixmask
HWY_AFTER_NAMESPACE();
