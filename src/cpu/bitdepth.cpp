#include "pixmask/image.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr std::array<std::array<std::uint8_t, 8>, 8> kBayer8x8 = {{
    {{0, 48, 12, 60, 3, 51, 15, 63}},
    {{32, 16, 44, 28, 35, 19, 47, 31}},
    {{8, 56, 4, 52, 11, 59, 7, 55}},
    {{40, 24, 36, 20, 43, 27, 39, 23}},
    {{2, 50, 14, 62, 1, 49, 13, 61}},
    {{34, 18, 46, 30, 33, 17, 45, 29}},
    {{10, 58, 6, 54, 9, 57, 5, 53}},
    {{42, 26, 38, 22, 41, 25, 37, 21}},
}};

constexpr float kInvBayerScale = 1.0f / 64.0f;

std::uint32_t sanitize_bits(std::uint32_t bits) noexcept {
    if (bits == 0) {
        bits = 6;
    }
    if (bits < 1) {
        bits = 1;
    }
    if (bits > 8) {
        bits = 8;
    }
    return bits;
}

float clamp01(float value) noexcept {
    if (value < 0.0f) {
        return 0.0f;
    }
    if (value > 1.0f) {
        return 1.0f;
    }
    return value;
}

void quantize_in_place(float *data,
                       std::size_t width,
                       std::size_t height,
                       std::size_t channels,
                       std::uint32_t bits) noexcept {
    if (data == nullptr || width == 0 || height == 0 || channels == 0) {
        return;
    }

    const std::uint32_t sanitized_bits = sanitize_bits(bits);
    const std::uint32_t levels = 1u << sanitized_bits;
    const int max_level = static_cast<int>(levels - 1u);
    const float inv_levels_minus_one = max_level > 0 ? 1.0f / static_cast<float>(max_level) : 0.0f;

    for (std::size_t y = 0; y < height; ++y) {
        const std::size_t row_offset = y * width * channels;
        const std::uint32_t matrix_y = static_cast<std::uint32_t>(y & 7u);
        for (std::size_t x = 0; x < width; ++x) {
            const std::uint32_t matrix_x = static_cast<std::uint32_t>(x & 7u);
            const float threshold =
                (static_cast<float>(kBayer8x8[matrix_y][matrix_x]) + 0.5f) * kInvBayerScale;
            for (std::size_t c = 0; c < channels; ++c) {
                const std::size_t idx = row_offset + x * channels + c;
                const float clamped = clamp01(data[idx]);
                const float scaled = clamped * static_cast<float>(levels);
                const float biased = scaled + threshold - 0.5f;
                int quantized = static_cast<int>(std::floor(biased));
                if (quantized < 0) {
                    quantized = 0;
                } else if (quantized > max_level) {
                    quantized = max_level;
                }
                data[idx] = max_level > 0 ? static_cast<float>(quantized) * inv_levels_minus_one : 0.0f;
            }
        }
    }
}

std::size_t infer_channels(const pixmask::Image &image) noexcept {
    if (image.width == 0 || image.height == 0) {
        return 0;
    }
    const std::size_t pixels = image.width * image.height;
    if (pixels == 0) {
        return 0;
    }
    if (image.pixels.empty() || (image.pixels.size() % pixels) != 0) {
        return 0;
    }
    return image.pixels.size() / pixels;
}

} // namespace

namespace pixmask {

void quantize_bitdepth(Image &image, std::uint32_t bits) noexcept {
    const std::size_t channels = infer_channels(image);
    if (channels == 0) {
        return;
    }
    quantize_in_place(image.pixels.data(), image.width, image.height, channels, bits);
}

void convert_to_float(Image &image) {
    (void)image;
}

} // namespace pixmask

extern "C" {

void pixmask_quantize_bitdepth(float *data,
                               std::size_t width,
                               std::size_t height,
                               std::size_t channels,
                               std::uint32_t bits) noexcept {
    quantize_in_place(data, width, height, channels, bits);
}

}
