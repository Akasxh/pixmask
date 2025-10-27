#include "common/pixel_ops.h"
#include "pixmask/api.h"
#include "pixmask/filters.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace pixmask {
namespace {

bool to_float_image(const CpuImage &src, Image &dst) {
    if (!validate_image(src)) {
        return false;
    }

    dst.width = src.width;
    dst.height = src.height;
    const std::size_t channels = 3;
    const std::size_t expected = dst.width * dst.height * channels;
    dst.pixels.assign(expected, 0.0f);

    CpuImage float_view(PixelType::F32_RGB,
                        dst.width,
                        dst.height,
                        dst.width * channels * sizeof(float),
                        dst.pixels.data());

    return convert_image(src, float_view);
}

bool from_float_image(const Image &src, const CpuImage &dst) {
    if (src.width != dst.width || src.height != dst.height) {
        return false;
    }
    if (src.pixels.empty()) {
        return false;
    }

    CpuImage float_view(PixelType::F32_RGB,
                        src.width,
                        src.height,
                        src.width * 3 * sizeof(float),
                        const_cast<float *>(src.pixels.data()));

    return convert_image(float_view, dst);
}

std::size_t scaled_dimension(std::size_t value, double scale) noexcept {
    const double scaled = static_cast<double>(value) * scale;
    const std::size_t rounded = static_cast<std::size_t>(std::llround(scaled));
    return rounded > 0 ? rounded : 1;
}

} // namespace

void quantize_bitdepth(Image &image, std::uint32_t bits) noexcept;

float sanitize_pixel(float value) noexcept {
    return std::clamp(value, 0.0f, 1.0f);
}

Image sanitize_image(const Image &input) {
    Image output = input;
    for (float &pixel : output.pixels) {
        pixel = sanitize_pixel(pixel);
    }
    return output;
}

bool sanitize(const CpuImage &input, const CpuImage &output) {
    if (!validate_image(input) || !validate_image(output)) {
        return false;
    }

    const auto supported_type = [](PixelType type) noexcept {
        return type == PixelType::U8_RGB || type == PixelType::F32_RGB;
    };

    if (!supported_type(input.type) || !supported_type(output.type)) {
        return false;
    }

    if (input.width == 0 || input.height == 0) {
        return false;
    }

    if (input.width != output.width || input.height != output.height) {
        return false;
    }

    if ((output.width % 2u) != 0u || (output.height % 2u) != 0u) {
        return false;
    }

    Image working;
    if (!to_float_image(input, working)) {
        return false;
    }

    const std::size_t down_width = scaled_dimension(working.width, 0.25);
    const std::size_t down_height = scaled_dimension(working.height, 0.25);

    Image low_res = resample_cubic(working, down_width, down_height);
    if (low_res.pixels.empty()) {
        return false;
    }

    quantize_bitdepth(low_res, 6);

    Image filtered = dct8x8_hf_attenuate(low_res, 60);
    if (filtered.pixels.empty()) {
        return false;
    }

    const std::size_t filtered_elements = filtered.pixels.size();
    for (std::size_t i = 0; i < filtered_elements; ++i) {
        const float mixed = 0.4f * filtered.pixels[i] + 0.6f * low_res.pixels[i];
        filtered.pixels[i] = sanitize_pixel(mixed);
    }

    Image upscaled = resample_cubic(filtered, output.width, output.height);
    if (upscaled.pixels.empty()) {
        return false;
    }

    const std::size_t sr_width = output.width / 2;
    const std::size_t sr_height = output.height / 2;
    if (sr_width == 0 || sr_height == 0) {
        return false;
    }

    Image sr_input = resample_cubic(filtered, sr_width, sr_height);
    if (sr_input.pixels.empty()) {
        return false;
    }

    std::vector<float> sr_output(output.width * output.height * 3, 0.0f);

    CpuImage sr_input_view(PixelType::F32_RGB,
                           sr_width,
                           sr_height,
                           sr_width * 3 * sizeof(float),
                           sr_input.pixels.data());
    CpuImage sr_output_view(PixelType::F32_RGB,
                            output.width,
                            output.height,
                            output.width * 3 * sizeof(float),
                            sr_output.data());

    if (!sr_lite_refine(sr_input_view, sr_output_view)) {
        return false;
    }

    const std::size_t total_elements = output.width * output.height * 3;
    constexpr float sr_weight = 0.15f;
    constexpr float upscaled_weight = 0.35f;
    constexpr float original_weight = 1.0f - sr_weight - upscaled_weight;
    for (std::size_t i = 0; i < total_elements; ++i) {
        const float blended = sr_weight * sr_output[i] +
                              upscaled_weight * upscaled.pixels[i] +
                              original_weight * working.pixels[i];
        sr_output[i] = sanitize_pixel(blended);
    }

    Image final_image;
    final_image.width = output.width;
    final_image.height = output.height;
    final_image.pixels = std::move(sr_output);

    if (!from_float_image(final_image, output)) {
        return false;
    }

    return true;
}

} // namespace pixmask

extern "C" {

bool pixmask_sanitize(const pixmask::CpuImage *input, const pixmask::CpuImage *output) {
    if (input == nullptr || output == nullptr) {
        return false;
    }
    return pixmask::sanitize(*input, *output);
}

}
