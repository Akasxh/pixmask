#include "pixmask/filters.h"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace pixmask {
namespace {

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

inline void prefilter_line(float *line, std::size_t length, std::size_t stride) {
    if (length == 0) {
        return;
    }
    if (length == 1) {
        return;
    }

    std::vector<float> diagonal(length, 4.0f);
    std::vector<float> upper(length - 1, 0.0f);
    std::vector<float> lower(length - 1, 0.0f);
    std::vector<float> rhs(length, 0.0f);

    rhs[0] = 6.0f * line[0];
    upper[0] = 2.0f;

    for (std::size_t i = 1; i + 1 < length; ++i) {
        rhs[i] = 6.0f * line[i * stride];
        lower[i - 1] = 1.0f;
        upper[i] = 1.0f;
    }

    rhs[length - 1] = 6.0f * line[(length - 1) * stride];
    lower[length - 2] = 2.0f;

    for (std::size_t i = 1; i < length; ++i) {
        const float factor = lower[i - 1] / diagonal[i - 1];
        diagonal[i] -= factor * upper[i - 1];
        rhs[i] -= factor * rhs[i - 1];
    }

    line[(length - 1) * stride] = rhs[length - 1] / diagonal[length - 1];
    for (std::size_t i = length - 1; i-- > 0;) {
        const float value = (rhs[i] - upper[i] * line[(i + 1) * stride]) / diagonal[i];
        line[i * stride] = value;
    }
}

} // namespace

Image b_spline_prefilter(const Image &input) {
    Image output = input;
    const std::size_t width = output.width;
    const std::size_t height = output.height;
    if (width == 0 || height == 0) {
        return output;
    }
    const std::size_t channels = compute_channels(output);
    if (channels == 0) {
        return output;
    }

    float *data = output.pixels.data();

    for (std::size_t y = 0; y < height; ++y) {
        float *row = data + y * width * channels;
        for (std::size_t c = 0; c < channels; ++c) {
            prefilter_line(row + c, width, channels);
        }
    }

    const std::size_t row_stride = width * channels;
    for (std::size_t x = 0; x < width; ++x) {
        for (std::size_t c = 0; c < channels; ++c) {
            prefilter_line(data + x * channels + c, height, row_stride);
        }
    }

    return output;
}

} // namespace pixmask

extern "C" {

void pixmask_cubic_b_spline_prefilter(const float *src,
                                      float *dst,
                                      std::size_t width,
                                      std::size_t height,
                                      std::size_t channels) noexcept {
    if (src == nullptr || dst == nullptr) {
        return;
    }
    if (width == 0 || height == 0 || channels == 0) {
        return;
    }

    const std::size_t total = width * height;
    if (total == 0) {
        return;
    }
    const std::size_t element_count = total * channels;

    pixmask::Image input;
    input.width = width;
    input.height = height;
    input.pixels.assign(src, src + element_count);

    pixmask::Image filtered = pixmask::b_spline_prefilter(input);
    const std::size_t to_copy = std::min<std::size_t>(element_count, filtered.pixels.size());
    std::copy_n(filtered.pixels.data(), to_copy, dst);
}

}
