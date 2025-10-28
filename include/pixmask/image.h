#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace pixmask {

enum class PixelType : std::uint32_t {
    U8_RGB = 0,
    U8_RGBA = 1,
    F32_RGB = 2,
};

constexpr std::size_t pixel_channels(PixelType type) noexcept {
    switch (type) {
    case PixelType::U8_RGB:
    case PixelType::F32_RGB:
        return 3;
    case PixelType::U8_RGBA:
        return 4;
    }
    return 0;
}

constexpr std::size_t bytes_per_channel(PixelType type) noexcept {
    switch (type) {
    case PixelType::U8_RGB:
    case PixelType::U8_RGBA:
        return 1;
    case PixelType::F32_RGB:
        return sizeof(float);
    }
    return 0;
}

constexpr std::size_t bytes_per_pixel(PixelType type) noexcept {
    return pixel_channels(type) * bytes_per_channel(type);
}

struct CpuImage {
    PixelType type = PixelType::U8_RGB;
    std::size_t width = 0;
    std::size_t height = 0;
    std::size_t stride_bytes = 0;
    void* data = nullptr;

    CpuImage() = default;

    CpuImage(PixelType pixel_type,
             std::size_t image_width,
             std::size_t image_height,
             std::size_t row_stride_bytes,
             void* buffer) noexcept
        : type(pixel_type)
        , width(image_width)
        , height(image_height)
        , stride_bytes(row_stride_bytes)
        , data(buffer) {}

    template <typename T>
    T* data_as() noexcept {
        return static_cast<T*>(data);
    }

    template <typename T>
    const T* data_as() const noexcept {
        return static_cast<const T*>(data);
    }

    std::size_t row_bytes() const noexcept {
        return width * bytes_per_pixel(type);
    }

    bool is_contiguous() const noexcept {
        return stride_bytes == row_bytes();
    }
};

struct Image {
    std::size_t width = 0;
    std::size_t height = 0;
    std::vector<float> pixels;
};

} // namespace pixmask
