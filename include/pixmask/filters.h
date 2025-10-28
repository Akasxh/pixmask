#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "pixmask/image.h"

namespace pixmask {

constexpr float kCubicBSplinePole = static_cast<float>(std::sqrt(3.0) - 2.0);
constexpr float kCubicBSplinePrefilterTolerance = 1.0e-5f;

struct CubicPhase {
    std::vector<std::size_t> indices;
    std::vector<float> weights;
};

std::vector<CubicPhase> build_cubic_weight_table(std::size_t src_size,
                                                 std::size_t dst_size) noexcept;

Image box_blur(const Image &input);
Image sharpen(const Image &input);
Image b_spline_prefilter(const Image &input);
Image resample_cubic(const Image &input, std::size_t new_width, std::size_t new_height);
Image dct8x8_hf_attenuate(const Image &input, int quality);

} // namespace pixmask

#if defined(_WIN32)
#    define PIXMASK_CAPI_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#    define PIXMASK_CAPI_EXPORT __attribute__((visibility("default"))) __attribute__((used))
#else
#    define PIXMASK_CAPI_EXPORT
#endif

extern "C" {

PIXMASK_CAPI_EXPORT void pixmask_cubic_b_spline_prefilter(const float *src,
                                                          float *dst,
                                                          std::size_t width,
                                                          std::size_t height,
                                                          std::size_t channels) noexcept;

PIXMASK_CAPI_EXPORT void pixmask_cubic_resample(const float *src,
                                                float *dst,
                                                std::size_t src_width,
                                                std::size_t src_height,
                                                std::size_t channels,
                                                std::size_t dst_width,
                                                std::size_t dst_height) noexcept;

PIXMASK_CAPI_EXPORT void pixmask_dct8x8_hf_attenuate(const float *src,
                                                     float *dst,
                                                     std::size_t width,
                                                     std::size_t height,
                                                     std::size_t channels,
                                                     int quality) noexcept;

}

#undef PIXMASK_CAPI_EXPORT
