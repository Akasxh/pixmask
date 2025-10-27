#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "pixmask/image.h"

namespace pixmask {

inline constexpr float kInv255 = 1.0f / 255.0f;

inline bool validate_image(const CpuImage& image) noexcept {
    const std::size_t bpp = bytes_per_pixel(image.type);
    if (bpp == 0) {
        return false;
    }
    if (image.width == 0 || image.height == 0) {
        return false;
    }
    if (image.stride_bytes < image.row_bytes()) {
        return false;
    }
    if (image.data == nullptr) {
        return false;
    }
    const std::size_t channel_size = bytes_per_channel(image.type);
    if (channel_size == 0 || (image.stride_bytes % channel_size) != 0) {
        return false;
    }
    return true;
}

inline std::uint8_t float_to_u8(float value) noexcept {
    value = std::clamp(value, 0.0f, 1.0f);
    const float scaled = std::round(value * 255.0f);
    const float clamped = std::clamp(scaled, 0.0f, 255.0f);
    return static_cast<std::uint8_t>(clamped);
}

inline void copy_image_bytes(const CpuImage& src, const CpuImage& dst) noexcept {
    const auto* src_base = reinterpret_cast<const std::uint8_t*>(src.data);
    auto* dst_base = reinterpret_cast<std::uint8_t*>(dst.data);
    const std::size_t row_bytes = src.row_bytes();
    for (std::size_t y = 0; y < src.height; ++y) {
        std::memcpy(dst_base + y * dst.stride_bytes, src_base + y * src.stride_bytes, row_bytes);
    }
}

inline bool convert_image(const CpuImage& src, const CpuImage& dst) noexcept {
    if (!validate_image(src) || !validate_image(dst)) {
        return false;
    }
    if (src.width != dst.width || src.height != dst.height) {
        return false;
    }

    if (src.type == dst.type) {
        copy_image_bytes(src, dst);
        return true;
    }

    const auto* src_bytes = reinterpret_cast<const std::uint8_t*>(src.data);
    auto* dst_bytes = reinterpret_cast<std::uint8_t*>(dst.data);

    switch (src.type) {
    case PixelType::U8_RGB: {
        if (dst.type != PixelType::F32_RGB) {
            return false;
        }
        for (std::size_t y = 0; y < src.height; ++y) {
            const auto* src_row = src_bytes + y * src.stride_bytes;
            auto* dst_row = reinterpret_cast<float*>(dst_bytes + y * dst.stride_bytes);
            for (std::size_t x = 0; x < src.width; ++x) {
                const std::size_t src_index = x * 3;
                const std::size_t dst_index = x * 3;
                dst_row[dst_index + 0] = static_cast<float>(src_row[src_index + 0]) * kInv255;
                dst_row[dst_index + 1] = static_cast<float>(src_row[src_index + 1]) * kInv255;
                dst_row[dst_index + 2] = static_cast<float>(src_row[src_index + 2]) * kInv255;
            }
        }
        return true;
    }
    case PixelType::U8_RGBA: {
        if (dst.type != PixelType::F32_RGB) {
            return false;
        }
        for (std::size_t y = 0; y < src.height; ++y) {
            const auto* src_row = src_bytes + y * src.stride_bytes;
            auto* dst_row = reinterpret_cast<float*>(dst_bytes + y * dst.stride_bytes);
            for (std::size_t x = 0; x < src.width; ++x) {
                const std::size_t src_index = x * 4;
                const std::size_t dst_index = x * 3;
                dst_row[dst_index + 0] = static_cast<float>(src_row[src_index + 0]) * kInv255;
                dst_row[dst_index + 1] = static_cast<float>(src_row[src_index + 1]) * kInv255;
                dst_row[dst_index + 2] = static_cast<float>(src_row[src_index + 2]) * kInv255;
            }
        }
        return true;
    }
    case PixelType::F32_RGB: {
        switch (dst.type) {
        case PixelType::U8_RGB: {
            for (std::size_t y = 0; y < src.height; ++y) {
                const auto* src_row = reinterpret_cast<const float*>(src_bytes + y * src.stride_bytes);
                auto* dst_row = dst_bytes + y * dst.stride_bytes;
                for (std::size_t x = 0; x < src.width; ++x) {
                    const std::size_t src_index = x * 3;
                    const std::size_t dst_index = x * 3;
                    dst_row[dst_index + 0] = float_to_u8(src_row[src_index + 0]);
                    dst_row[dst_index + 1] = float_to_u8(src_row[src_index + 1]);
                    dst_row[dst_index + 2] = float_to_u8(src_row[src_index + 2]);
                }
            }
            return true;
        }
        case PixelType::U8_RGBA: {
            for (std::size_t y = 0; y < src.height; ++y) {
                const auto* src_row = reinterpret_cast<const float*>(src_bytes + y * src.stride_bytes);
                auto* dst_row = dst_bytes + y * dst.stride_bytes;
                for (std::size_t x = 0; x < src.width; ++x) {
                    const std::size_t src_index = x * 3;
                    const std::size_t dst_index = x * 4;
                    dst_row[dst_index + 0] = float_to_u8(src_row[src_index + 0]);
                    dst_row[dst_index + 1] = float_to_u8(src_row[src_index + 1]);
                    dst_row[dst_index + 2] = float_to_u8(src_row[src_index + 2]);
                    dst_row[dst_index + 3] = 255u;
                }
            }
            return true;
        }
        default:
            break;
        }
        break;
    }
    default:
        break;
    }

    return false;
}

float sanitize_pixel(float value) noexcept;
Image sanitize_image(const Image &input);

} // namespace pixmask

#if defined(_WIN32)
#    define PIXMASK_CAPI_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#    define PIXMASK_CAPI_EXPORT __attribute__((visibility("default"))) __attribute__((used))
#else
#    define PIXMASK_CAPI_EXPORT
#endif

extern "C" {

PIXMASK_CAPI_EXPORT inline bool pixmask_validate_image(const pixmask::CpuImage* image) noexcept {
    if (image == nullptr) {
        return false;
    }
    return pixmask::validate_image(*image);
}

PIXMASK_CAPI_EXPORT inline bool pixmask_convert_image(const pixmask::CpuImage* src, const pixmask::CpuImage* dst) noexcept {
    if (src == nullptr || dst == nullptr) {
        return false;
    }
    return pixmask::convert_image(*src, *dst);
}

}

#undef PIXMASK_CAPI_EXPORT
