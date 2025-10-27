#include "common/pixel_ops.h"
#include "common/thread_pool.h"
#include "pixmask/api.h"
#include "pixmask/image.h"
#include "pixmask/sr_weights.h"
#include "pixmask/version.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <vector>

namespace {

using pixmask::parallel_for;

constexpr std::size_t kInputChannels = pixmask::sr::kInputChannels;
constexpr std::size_t kStage1Channels = pixmask::sr::kConv1OutChannels;
constexpr std::size_t kStage2Channels = pixmask::sr::kConv2OutChannels;
constexpr std::size_t kStage3Channels = pixmask::sr::kConv3OutChannels;
constexpr std::size_t kUpscaleFactor = pixmask::sr::kUpscaleFactor;

std::size_t mirror_index(std::ptrdiff_t idx, std::size_t length) noexcept {
    if (length <= 1) {
        return 0;
    }
    const std::ptrdiff_t period = static_cast<std::ptrdiff_t>(length * 2 - 2);
    std::ptrdiff_t value = idx % period;
    if (value < 0) {
        value += period;
    }
    if (value >= static_cast<std::ptrdiff_t>(length)) {
        value = period - value;
    }
    return static_cast<std::size_t>(value);
}

void convolve3x3(const float *input,
                 float *output,
                 std::size_t width,
                 std::size_t height,
                 std::size_t in_channels,
                 std::size_t out_channels,
                 const float *weights,
                 const float *bias,
                 bool relu) {
    if (input == nullptr || output == nullptr || width == 0 || height == 0) {
        return;
    }

    const std::size_t in_row_stride = width * in_channels;
    const std::size_t out_row_stride = width * out_channels;

    parallel_for(0, height, [&](std::size_t y) {
        std::array<std::size_t, 3> y_indices = {
            mirror_index(static_cast<std::ptrdiff_t>(y) - 1, height),
            mirror_index(static_cast<std::ptrdiff_t>(y), height),
            mirror_index(static_cast<std::ptrdiff_t>(y) + 1, height),
        };

        for (std::size_t x = 0; x < width; ++x) {
            std::array<std::size_t, 3> x_indices = {
                mirror_index(static_cast<std::ptrdiff_t>(x) - 1, width),
                mirror_index(static_cast<std::ptrdiff_t>(x), width),
                mirror_index(static_cast<std::ptrdiff_t>(x) + 1, width),
            };

            const std::size_t out_base = y * out_row_stride + x * out_channels;
            for (std::size_t oc = 0; oc < out_channels; ++oc) {
                float acc = bias[oc];
                const float *kernel = weights + (oc * 3 * 3 * in_channels);
                for (std::size_t ky = 0; ky < 3; ++ky) {
                    const std::size_t sy = y_indices[ky];
                    const float *row = input + sy * in_row_stride;
                    for (std::size_t kx = 0; kx < 3; ++kx) {
                        const std::size_t sx = x_indices[kx];
                        const float *pixel = row + sx * in_channels;
                        for (std::size_t ic = 0; ic < in_channels; ++ic) {
                            acc += kernel[ic] * pixel[ic];
                        }
                        kernel += in_channels;
                    }
                }

                if (relu && acc < 0.0f) {
                    acc = 0.0f;
                }
                output[out_base + oc] = acc;
            }
        }
    });
}

void pixel_shuffle_r2(const float *input,
                      float *output,
                      std::size_t width,
                      std::size_t height,
                      std::size_t channels) {
    if (input == nullptr || output == nullptr || width == 0 || height == 0) {
        return;
    }

    const std::size_t in_channels = channels * kUpscaleFactor * kUpscaleFactor;
    const std::size_t out_width = width * kUpscaleFactor;
    const std::size_t out_height = height * kUpscaleFactor;

    parallel_for(0, height, [&](std::size_t y) {
        for (std::size_t x = 0; x < width; ++x) {
            const float *in_pixel = input + (y * width + x) * in_channels;
            for (std::size_t c = 0; c < channels; ++c) {
                const std::size_t base = c * kUpscaleFactor * kUpscaleFactor;
                for (std::size_t sub = 0; sub < kUpscaleFactor * kUpscaleFactor; ++sub) {
                    const std::size_t oy = y * kUpscaleFactor + (sub / kUpscaleFactor);
                    const std::size_t ox = x * kUpscaleFactor + (sub % kUpscaleFactor);
                    float value = in_pixel[base + sub];
                    value = std::clamp(value, 0.0f, 1.0f);
                    output[(oy * out_width + ox) * channels + c] = value;
                }
            }
        }
    });
}

void sr_lite_forward(const float *input, float *output, std::size_t width, std::size_t height) {
    if (input == nullptr || output == nullptr || width == 0 || height == 0) {
        return;
    }

    std::vector<float> stage1(width * height * kStage1Channels, 0.0f);
    std::vector<float> stage2(width * height * kStage2Channels, 0.0f);
    std::vector<float> stage3(width * height * kStage3Channels, 0.0f);

    convolve3x3(input,
                stage1.data(),
                width,
                height,
                kInputChannels,
                kStage1Channels,
                pixmask::sr::kConv1Weights.data(),
                pixmask::sr::kConv1Bias.data(),
                true);

    convolve3x3(stage1.data(),
                stage2.data(),
                width,
                height,
                kStage1Channels,
                kStage2Channels,
                pixmask::sr::kConv2Weights.data(),
                pixmask::sr::kConv2Bias.data(),
                true);

    convolve3x3(stage2.data(),
                stage3.data(),
                width,
                height,
                kStage2Channels,
                kStage3Channels,
                pixmask::sr::kConv3Weights.data(),
                pixmask::sr::kConv3Bias.data(),
                false);

    pixel_shuffle_r2(stage3.data(), output, width, height, kInputChannels);
}

} // namespace

namespace pixmask {

void initialize() {}

std::string version_string() {
    std::ostringstream oss;
    oss << version_major() << '.' << version_minor() << '.' << version_patch();
    return oss.str();
}

bool sr_lite_refine(const CpuImage &input, const CpuImage &output) {
    if (!validate_image(input) || !validate_image(output)) {
        return false;
    }

    if (input.width == 0 || input.height == 0) {
        return false;
    }

    if (output.width != input.width * kUpscaleFactor || output.height != input.height * kUpscaleFactor) {
        return false;
    }

    const auto supports_type = [](PixelType type) noexcept {
        switch (type) {
        case PixelType::U8_RGB:
        case PixelType::U8_RGBA:
        case PixelType::F32_RGB:
            return true;
        default:
            return false;
        }
    };

    if (!supports_type(input.type) || !supports_type(output.type)) {
        return false;
    }

    const std::size_t src_stride = input.width * kInputChannels * sizeof(float);
    std::vector<float> low_res(input.width * input.height * kInputChannels, 0.0f);
    CpuImage low_res_view(PixelType::F32_RGB, input.width, input.height, src_stride, low_res.data());
    if (!convert_image(input, low_res_view)) {
        return false;
    }

    const std::size_t out_width = output.width;
    const std::size_t out_height = output.height;
    const std::size_t dst_stride = out_width * kInputChannels * sizeof(float);
    std::vector<float> high_res(out_width * out_height * kInputChannels, 0.0f);

    sr_lite_forward(low_res.data(), high_res.data(), input.width, input.height);

    CpuImage high_res_view(PixelType::F32_RGB, out_width, out_height, dst_stride, high_res.data());
    if (!convert_image(high_res_view, output)) {
        return false;
    }

    return true;
}

} // namespace pixmask

extern "C" {

bool pixmask_sr_lite(const pixmask::CpuImage *input, const pixmask::CpuImage *output) {
    if (input == nullptr || output == nullptr) {
        return false;
    }
    return pixmask::sr_lite_refine(*input, *output);
}

}
