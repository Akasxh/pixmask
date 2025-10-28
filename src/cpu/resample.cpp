#include "common/pixel_ops.h"
#include "common/thread_pool.h"
#include "pixmask/filters.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace pixmask {
namespace {

constexpr float kCubicParameter = -0.5f; // Catmull-Rom

inline float cubic_kernel(float x) noexcept {
    x = std::abs(x);
    const float x2 = x * x;
    const float x3 = x2 * x;
    if (x < 1.0f) {
        return (kCubicParameter + 2.0f) * x3 - (kCubicParameter + 3.0f) * x2 + 1.0f;
    }
    if (x < 2.0f) {
        return kCubicParameter * x3 - 5.0f * kCubicParameter * x2 + 8.0f * kCubicParameter * x -
               4.0f * kCubicParameter;
    }
    return 0.0f;
}

inline std::size_t mirror_index(std::ptrdiff_t idx, std::size_t length) noexcept {
    if (length == 0) {
        return 0;
    }
    if (length == 1) {
        return 0;
    }

    const std::ptrdiff_t period = static_cast<std::ptrdiff_t>(length * 2 - 2);
    if (period == 0) {
        return 0;
    }

    idx %= period;
    if (idx < 0) {
        idx += period;
    }
    if (idx >= static_cast<std::ptrdiff_t>(length)) {
        idx = period - idx;
    }
    return static_cast<std::size_t>(idx);
}

inline std::size_t compute_channels(const Image &image) noexcept {
    if (image.width == 0 || image.height == 0) {
        return 0;
    }
    const std::size_t count = image.width * image.height;
    if (count == 0) {
        return 0;
    }
    if (image.pixels.size() % count != 0) {
        return 0;
    }
    return image.pixels.size() / count;
}

bool copy_cpu_to_image(const CpuImage &src, Image &dst) {
    if (!validate_image(src)) {
        return false;
    }

    dst.width = src.width;
    dst.height = src.height;
    const std::size_t channels = 3;
    dst.pixels.assign(dst.width * dst.height * channels, 0.0f);

    switch (src.type) {
    case PixelType::U8_RGB: {
        const auto *row_bytes = static_cast<const std::uint8_t *>(src.data);
        for (std::size_t y = 0; y < src.height; ++y) {
            const auto *src_row = row_bytes + y * src.stride_bytes;
            float *dst_row = dst.pixels.data() + y * dst.width * channels;
            for (std::size_t x = 0; x < src.width; ++x) {
                const std::size_t base = x * channels;
                dst_row[base + 0] = static_cast<float>(src_row[base + 0]) * kInv255;
                dst_row[base + 1] = static_cast<float>(src_row[base + 1]) * kInv255;
                dst_row[base + 2] = static_cast<float>(src_row[base + 2]) * kInv255;
            }
        }
        return true;
    }
    case PixelType::U8_RGBA: {
        const auto *row_bytes = static_cast<const std::uint8_t *>(src.data);
        for (std::size_t y = 0; y < src.height; ++y) {
            const auto *src_row = row_bytes + y * src.stride_bytes;
            float *dst_row = dst.pixels.data() + y * dst.width * channels;
            for (std::size_t x = 0; x < src.width; ++x) {
                const std::size_t rgba_index = x * 4;
                const std::size_t rgb_index = x * channels;
                dst_row[rgb_index + 0] = static_cast<float>(src_row[rgba_index + 0]) * kInv255;
                dst_row[rgb_index + 1] = static_cast<float>(src_row[rgba_index + 1]) * kInv255;
                dst_row[rgb_index + 2] = static_cast<float>(src_row[rgba_index + 2]) * kInv255;
            }
        }
        return true;
    }
    case PixelType::F32_RGB: {
        const auto *row_bytes = static_cast<const std::uint8_t *>(src.data);
        for (std::size_t y = 0; y < src.height; ++y) {
            const auto *src_row = reinterpret_cast<const float *>(row_bytes + y * src.stride_bytes);
            float *dst_row = dst.pixels.data() + y * dst.width * channels;
            std::copy_n(src_row, dst.width * channels, dst_row);
        }
        return true;
    }
    default:
        break;
    }

    return false;
}

bool copy_image_to_cpu(const Image &src, const CpuImage &dst) {
    if (dst.width != src.width || dst.height != src.height) {
        return false;
    }

    if (!validate_image(dst)) {
        return false;
    }

    const std::size_t channels = 3;
    switch (dst.type) {
    case PixelType::U8_RGB: {
        auto *row_bytes = static_cast<std::uint8_t *>(dst.data);
        for (std::size_t y = 0; y < dst.height; ++y) {
            const float *src_row = src.pixels.data() + y * src.width * channels;
            auto *dst_row = row_bytes + y * dst.stride_bytes;
            for (std::size_t x = 0; x < dst.width; ++x) {
                const std::size_t base = x * channels;
                dst_row[base + 0] = float_to_u8(src_row[base + 0]);
                dst_row[base + 1] = float_to_u8(src_row[base + 1]);
                dst_row[base + 2] = float_to_u8(src_row[base + 2]);
            }
        }
        return true;
    }
    case PixelType::U8_RGBA: {
        auto *row_bytes = static_cast<std::uint8_t *>(dst.data);
        for (std::size_t y = 0; y < dst.height; ++y) {
            const float *src_row = src.pixels.data() + y * src.width * channels;
            auto *dst_row = row_bytes + y * dst.stride_bytes;
            for (std::size_t x = 0; x < dst.width; ++x) {
                const std::size_t rgb_index = x * channels;
                const std::size_t rgba_index = x * 4;
                dst_row[rgba_index + 0] = float_to_u8(src_row[rgb_index + 0]);
                dst_row[rgba_index + 1] = float_to_u8(src_row[rgb_index + 1]);
                dst_row[rgba_index + 2] = float_to_u8(src_row[rgb_index + 2]);
                dst_row[rgba_index + 3] = 255u;
            }
        }
        return true;
    }
    case PixelType::F32_RGB: {
        auto *row_bytes = static_cast<std::uint8_t *>(dst.data);
        for (std::size_t y = 0; y < dst.height; ++y) {
            const float *src_row = src.pixels.data() + y * src.width * channels;
            auto *dst_row = reinterpret_cast<float *>(row_bytes + y * dst.stride_bytes);
            std::copy_n(src_row, dst.width * channels, dst_row);
        }
        return true;
    }
    default:
        break;
    }

    return false;
}

} // namespace

std::vector<CubicPhase> build_cubic_weight_table(std::size_t src_size,
                                                 std::size_t dst_size) noexcept {
    std::vector<CubicPhase> table;
    if (src_size == 0 || dst_size == 0) {
        return table;
    }

    const float scale = static_cast<float>(dst_size) / static_cast<float>(src_size);
    const float inv_scale = static_cast<float>(src_size) / static_cast<float>(dst_size);
    const bool downscale = scale < 1.0f;
    for (std::size_t i = 0; i < dst_size; ++i) {
        CubicPhase phase;
        if (downscale) {
            const float start = static_cast<float>(i) * inv_scale;
            const float end = start + inv_scale;
            float current = start;
            std::int32_t idx = static_cast<std::int32_t>(std::floor(current));
            float weight_sum = 0.0f;
            while (current < end) {
                const float next_edge = static_cast<float>(idx + 1);
                const float next = std::min(end, next_edge);
                const float coverage = next - current;
                if (coverage > 0.0f) {
                    const std::size_t mapped =
                        mirror_index(static_cast<std::ptrdiff_t>(idx), src_size);
                    phase.indices.push_back(mapped);
                    const float weight = coverage * scale;
                    phase.weights.push_back(weight);
                    weight_sum += weight;
                }
                current = next;
                ++idx;
            }
            if (weight_sum != 0.0f) {
                const float inv_sum = 1.0f / weight_sum;
                for (float &weight : phase.weights) {
                    weight *= inv_sum;
                }
            } else {
                const std::size_t mapped = mirror_index(static_cast<std::ptrdiff_t>(std::llround(start)), src_size);
                phase.indices.push_back(mapped);
                phase.weights.push_back(1.0f);
            }
        } else {
            const float src_pos = (static_cast<float>(i) + 0.5f) * inv_scale - 0.5f;
            const std::int32_t base = static_cast<std::int32_t>(std::floor(src_pos)) - 1;
            float weight_sum = 0.0f;
            for (int tap = 0; tap < 4; ++tap) {
                const std::int32_t idx = base + tap;
                const float distance = src_pos - static_cast<float>(idx);
                const float weight = cubic_kernel(distance);
                if (weight == 0.0f) {
                    continue;
                }
                const std::size_t mapped =
                    mirror_index(static_cast<std::ptrdiff_t>(idx), src_size);
                phase.indices.push_back(mapped);
                phase.weights.push_back(weight);
                weight_sum += weight;
            }
            if (weight_sum != 0.0f) {
                const float inv_sum = 1.0f / weight_sum;
                for (float &weight : phase.weights) {
                    weight *= inv_sum;
                }
            } else {
                const std::size_t mapped = mirror_index(static_cast<std::ptrdiff_t>(std::llround(src_pos)), src_size);
                phase.indices.push_back(mapped);
                phase.weights.push_back(1.0f);
            }
        }

        table.push_back(std::move(phase));
    }

    return table;
}

Image box_blur(const Image &input) {
    return sanitize_image(input);
}

Image sharpen(const Image &input) {
    return sanitize_image(input);
}

Image resample_cubic(const Image &input, std::size_t new_width, std::size_t new_height) {
    Image output;
    output.width = new_width;
    output.height = new_height;

    if (new_width == 0 || new_height == 0) {
        return output;
    }

    const std::size_t channels = compute_channels(input);
    if (channels == 0) {
        return output;
    }

    output.pixels.resize(new_width * new_height * channels, 0.0f);

    if (input.width == 0 || input.height == 0) {
        return output;
    }

    const std::vector<CubicPhase> horizontal_weights =
        build_cubic_weight_table(input.width, new_width);
    const std::vector<CubicPhase> vertical_weights =
        build_cubic_weight_table(input.height, new_height);

    if (horizontal_weights.empty() || vertical_weights.empty()) {
        return output;
    }

    std::vector<float> intermediate(new_width * input.height * channels, 0.0f);

    const float *src_data = input.pixels.data();
    float *intermediate_data = intermediate.data();
    parallel_for(0, input.height, [&](std::size_t y) {
        const float *src_row = src_data + y * input.width * channels;
        float *dst_row = intermediate_data + y * new_width * channels;
        for (std::size_t x = 0; x < new_width; ++x) {
            const CubicPhase &phase = horizontal_weights[x];
            const std::size_t tap_count = phase.indices.size();
            float *dst_pixel = dst_row + x * channels;
            for (std::size_t c = 0; c < channels; ++c) {
                float accum = 0.0f;
                for (std::size_t tap = 0; tap < tap_count; ++tap) {
                    const std::size_t src_idx = phase.indices[tap];
                    const float weight = phase.weights[tap];
                    const float value = src_row[src_idx * channels + c];
                    accum += weight * value;
                }
                dst_pixel[c] = accum;
            }
        }
    });

    float *dst_data = output.pixels.data();
    parallel_for(0, new_height, [&](std::size_t y) {
        const CubicPhase &phase = vertical_weights[y];
        const std::size_t tap_count = phase.indices.size();

        float *dst_row = dst_data + y * new_width * channels;
        for (std::size_t x = 0; x < new_width; ++x) {
            for (std::size_t c = 0; c < channels; ++c) {
                float accum = 0.0f;
                for (std::size_t tap = 0; tap < tap_count; ++tap) {
                    const std::size_t sample_y = phase.indices[tap];
                    const float weight = phase.weights[tap];
                    const float value = intermediate[sample_y * new_width * channels + x * channels + c];
                    accum += weight * value;
                }
                dst_row[x * channels + c] = std::clamp(accum, 0.0f, 1.0f);
            }
        }
    });

    return output;
}

enum class ResampleMode : int {
    Cubic = 0,
};

bool resize(const CpuImage &src,
            const CpuImage &dst,
            float scale_x,
            float scale_y,
            ResampleMode mode) {
    if (!validate_image(src) || !validate_image(dst)) {
        return false;
    }
    if (scale_x <= 0.0f || scale_y <= 0.0f) {
        return false;
    }
    if (mode != ResampleMode::Cubic) {
        return false;
    }

    const double expected_width = static_cast<double>(src.width) * static_cast<double>(scale_x);
    const double expected_height = static_cast<double>(src.height) * static_cast<double>(scale_y);
    if (expected_width <= 0.0 || expected_height <= 0.0) {
        return false;
    }

    const std::size_t target_width = static_cast<std::size_t>(std::llround(expected_width));
    const std::size_t target_height = static_cast<std::size_t>(std::llround(expected_height));
    if (target_width == 0 || target_height == 0) {
        return false;
    }
    if (dst.width != target_width || dst.height != target_height) {
        return false;
    }

    Image src_image;
    if (!copy_cpu_to_image(src, src_image)) {
        return false;
    }

    Image resized = resample_cubic(src_image, target_width, target_height);
    if (resized.pixels.empty()) {
        return false;
    }

    return copy_image_to_cpu(resized, dst);
}

} // namespace pixmask

extern "C" {

void pixmask_cubic_resample(const float *src,
                            float *dst,
                            std::size_t src_width,
                            std::size_t src_height,
                            std::size_t channels,
                            std::size_t dst_width,
                            std::size_t dst_height) noexcept {
    if (src == nullptr || dst == nullptr) {
        return;
    }
    if (src_width == 0 || src_height == 0 || channels == 0 || dst_width == 0 || dst_height == 0) {
        return;
    }

    const std::size_t src_elements = src_width * src_height * channels;
    const std::size_t dst_elements = dst_width * dst_height * channels;

    pixmask::Image input;
    input.width = src_width;
    input.height = src_height;
    input.pixels.assign(src, src + src_elements);

    pixmask::Image result = pixmask::resample_cubic(input, dst_width, dst_height);
    if (result.pixels.empty()) {
        return;
    }

    const std::size_t to_copy = std::min<std::size_t>(dst_elements, result.pixels.size());
    std::copy_n(result.pixels.data(), to_copy, dst);
}

bool pixmask_resize(const pixmask::CpuImage *src,
                    const pixmask::CpuImage *dst,
                    float scale_x,
                    float scale_y,
                    int mode) noexcept {
    if (src == nullptr || dst == nullptr) {
        return false;
    }
    if (mode != static_cast<int>(pixmask::ResampleMode::Cubic)) {
        return false;
    }
    return pixmask::resize(*src, *dst, scale_x, scale_y, pixmask::ResampleMode::Cubic);
}

}
