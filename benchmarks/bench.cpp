#include "common/pixel_ops.h"
#include "pixmask/api.h"
#include "pixmask/filters.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;

std::size_t scaled_dimension(std::size_t value, double scale) {
    const double scaled = static_cast<double>(value) * scale;
    const std::size_t rounded = static_cast<std::size_t>(std::llround(scaled));
    return rounded > 0 ? rounded : 1;
}

struct StageTimer {
    std::string name;
    Duration elapsed{0.0};
};

} // namespace

namespace pixmask {

void quantize_bitdepth(Image &image, std::uint32_t bits) noexcept;

} // namespace pixmask

int main() {
    constexpr std::size_t width = 1024;
    constexpr std::size_t height = 1024;
    constexpr std::size_t channels = 3;

    std::vector<std::uint8_t> input_data(width * height * channels, 0);
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            const std::size_t idx = (y * width + x) * channels;
            input_data[idx + 0] = static_cast<std::uint8_t>((x + y) % 256);
            input_data[idx + 1] = static_cast<std::uint8_t>((x * 2 + y) % 256);
            input_data[idx + 2] = static_cast<std::uint8_t>((x + y * 2) % 256);
        }
    }

    std::vector<std::uint8_t> output_data(width * height * channels, 0);

    pixmask::CpuImage input_view(pixmask::PixelType::U8_RGB,
                                 width,
                                 height,
                                 width * channels,
                                 input_data.data());
    pixmask::CpuImage output_view(pixmask::PixelType::U8_RGB,
                                  width,
                                  height,
                                  width * channels,
                                  output_data.data());

    if (!pixmask::validate_image(input_view) || !pixmask::validate_image(output_view)) {
        std::cerr << "Invalid benchmark images" << std::endl;
        return 1;
    }

    std::vector<StageTimer> timers;

    const auto total_start = Clock::now();

    pixmask::Image working;
    timers.push_back({"to_float", Duration{0.0}});
    {
        const auto start = Clock::now();
        working.width = input_view.width;
        working.height = input_view.height;
        working.pixels.assign(width * height * channels, 0.0f);

        pixmask::CpuImage float_view(pixmask::PixelType::F32_RGB,
                                     width,
                                     height,
                                     width * channels * sizeof(float),
                                     working.pixels.data());
        if (!pixmask::convert_image(input_view, float_view)) {
            std::cerr << "convert_image(u8->f32) failed" << std::endl;
            return 1;
        }
        timers.back().elapsed = Clock::now() - start;
    }

    timers.push_back({"downscale", Duration{0.0}});
    pixmask::Image low_res;
    {
        const auto start = Clock::now();
        const std::size_t down_width = scaled_dimension(width, 0.25);
        const std::size_t down_height = scaled_dimension(height, 0.25);
        low_res = pixmask::resample_cubic(working, down_width, down_height);
        if (low_res.pixels.empty()) {
            std::cerr << "downscale failed" << std::endl;
            return 1;
        }
        timers.back().elapsed = Clock::now() - start;
    }

    timers.push_back({"quantize", Duration{0.0}});
    {
        const auto start = Clock::now();
        pixmask::quantize_bitdepth(low_res, 6);
        timers.back().elapsed = Clock::now() - start;
    }

    timers.push_back({"dct", Duration{0.0}});
    pixmask::Image filtered;
    {
        const auto start = Clock::now();
        filtered = pixmask::dct8x8_hf_attenuate(low_res, 60);
        if (filtered.pixels.empty()) {
            std::cerr << "dct stage failed" << std::endl;
            return 1;
        }
        timers.back().elapsed = Clock::now() - start;
    }

    timers.push_back({"blend_low", Duration{0.0}});
    {
        const auto start = Clock::now();
        const std::size_t count = filtered.pixels.size();
        for (std::size_t i = 0; i < count; ++i) {
            const float mixed = 0.4f * filtered.pixels[i] + 0.6f * low_res.pixels[i];
            filtered.pixels[i] = std::clamp(mixed, 0.0f, 1.0f);
        }
        timers.back().elapsed = Clock::now() - start;
    }

    timers.push_back({"upscale", Duration{0.0}});
    pixmask::Image upscaled;
    {
        const auto start = Clock::now();
        upscaled = pixmask::resample_cubic(filtered, width, height);
        if (upscaled.pixels.empty()) {
            std::cerr << "upscale failed" << std::endl;
            return 1;
        }
        timers.back().elapsed = Clock::now() - start;
    }

    const std::size_t sr_width = width / 2;
    const std::size_t sr_height = height / 2;

    timers.push_back({"sr_prep", Duration{0.0}});
    pixmask::Image sr_input;
    {
        const auto start = Clock::now();
        sr_input = pixmask::resample_cubic(filtered, sr_width, sr_height);
        if (sr_input.pixels.empty()) {
            std::cerr << "sr prep failed" << std::endl;
            return 1;
        }
        timers.back().elapsed = Clock::now() - start;
    }

    std::vector<float> sr_output(width * height * channels, 0.0f);
    timers.push_back({"sr_lite", Duration{0.0}});
    {
        const auto start = Clock::now();
        pixmask::CpuImage sr_in_view(pixmask::PixelType::F32_RGB,
                                     sr_width,
                                     sr_height,
                                     sr_width * channels * sizeof(float),
                                     sr_input.pixels.data());
        pixmask::CpuImage sr_out_view(pixmask::PixelType::F32_RGB,
                                      width,
                                      height,
                                      width * channels * sizeof(float),
                                      sr_output.data());
        if (!pixmask::sr_lite_refine(sr_in_view, sr_out_view)) {
            std::cerr << "sr_lite_refine failed" << std::endl;
            return 1;
        }
        timers.back().elapsed = Clock::now() - start;
    }

    timers.push_back({"blend_final", Duration{0.0}});
    {
        const auto start = Clock::now();
        const std::size_t total = width * height * channels;
        constexpr float sr_weight = 0.15f;
        constexpr float up_weight = 0.35f;
        constexpr float original_weight = 1.0f - sr_weight - up_weight;
        for (std::size_t i = 0; i < total; ++i) {
            const float blended = sr_weight * sr_output[i] +
                                  up_weight * upscaled.pixels[i] +
                                  original_weight * working.pixels[i];
            sr_output[i] = std::clamp(blended, 0.0f, 1.0f);
        }
        timers.back().elapsed = Clock::now() - start;
    }

    timers.push_back({"to_u8", Duration{0.0}});
    {
        const auto start = Clock::now();
        pixmask::CpuImage float_view(pixmask::PixelType::F32_RGB,
                                     width,
                                     height,
                                     width * channels * sizeof(float),
                                     sr_output.data());
        if (!pixmask::convert_image(float_view, output_view)) {
            std::cerr << "convert_image(f32->u8) failed" << std::endl;
            return 1;
        }
        timers.back().elapsed = Clock::now() - start;
    }

    const auto total_end = Clock::now();
    const Duration total_elapsed = total_end - total_start;

    std::cout << "pixmask benchmark (" << width << "x" << height << ")\n";
    for (const auto &timer : timers) {
        std::cout << std::left << std::setw(12) << timer.name << ": "
                  << std::fixed << std::setprecision(3) << timer.elapsed.count() << " ms\n";
    }
    std::cout << std::left << std::setw(12) << "total" << ": "
              << std::fixed << std::setprecision(3) << total_elapsed.count() << " ms\n";

    return 0;
}
