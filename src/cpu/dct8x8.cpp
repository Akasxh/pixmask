#include "common/thread_pool.h"
#include "pixmask/dct_tables.h"
#include "pixmask/filters.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace pixmask {
namespace {

constexpr float kPi = 3.14159265358979323846f;

inline std::array<std::array<float, 8>, 8> build_cos_table() {
    std::array<std::array<float, 8>, 8> table{};
    for (std::size_t u = 0; u < 8; ++u) {
        for (std::size_t x = 0; x < 8; ++x) {
            const float angle = (kPi / 8.0f) * static_cast<float>(u) *
                                (static_cast<float>(x) + 0.5f);
            table[u][x] = std::cos(angle);
        }
    }
    return table;
}

inline const std::array<std::array<float, 8>, 8> &cos_table() {
    static const std::array<std::array<float, 8>, 8> table = build_cos_table();
    return table;
}

inline std::array<float, 8> alpha_factors() {
    std::array<float, 8> alpha{};
    alpha[0] = 0.3535533905932738f; // sqrt(1/8)
    for (std::size_t i = 1; i < 8; ++i) {
        alpha[i] = 0.5f; // sqrt(2/8)
    }
    return alpha;
}

inline const std::array<float, 8> &alphas() {
    static const std::array<float, 8> alpha = alpha_factors();
    return alpha;
}

inline void fdct_1d(const float *in, float *out) {
    const auto &cos = cos_table();
    const auto &alpha = alphas();
    for (std::size_t u = 0; u < 8; ++u) {
        float sum = 0.0f;
        for (std::size_t x = 0; x < 8; ++x) {
            sum += in[x] * cos[u][x];
        }
        out[u] = sum * alpha[u];
    }
}

inline void idct_1d(const float *in, float *out) {
    const auto &cos = cos_table();
    const auto &alpha = alphas();
    for (std::size_t x = 0; x < 8; ++x) {
        float sum = 0.0f;
        for (std::size_t u = 0; u < 8; ++u) {
            sum += alpha[u] * in[u] * cos[u][x];
        }
        out[x] = sum;
    }
}

inline void forward_dct(float *block) {
    float tmp[64];
    float out_row[8];
    for (std::size_t y = 0; y < 8; ++y) {
        fdct_1d(block + y * 8, out_row);
        for (std::size_t x = 0; x < 8; ++x) {
            tmp[y * 8 + x] = out_row[x];
        }
    }

    float col_in[8];
    float col_out[8];
    for (std::size_t x = 0; x < 8; ++x) {
        for (std::size_t y = 0; y < 8; ++y) {
            col_in[y] = tmp[y * 8 + x];
        }
        fdct_1d(col_in, col_out);
        for (std::size_t y = 0; y < 8; ++y) {
            block[y * 8 + x] = col_out[y];
        }
    }
}

inline void inverse_dct(float *block) {
    float tmp[64];
    float col_out[8];
    float col_in[8];
    for (std::size_t x = 0; x < 8; ++x) {
        for (std::size_t y = 0; y < 8; ++y) {
            col_in[y] = block[y * 8 + x];
        }
        idct_1d(col_in, col_out);
        for (std::size_t y = 0; y < 8; ++y) {
            tmp[y * 8 + x] = col_out[y];
        }
    }

    float row_out[8];
    for (std::size_t y = 0; y < 8; ++y) {
        idct_1d(tmp + y * 8, row_out);
        for (std::size_t x = 0; x < 8; ++x) {
            block[y * 8 + x] = row_out[x];
        }
    }
}

inline std::array<float, 64> build_quality_table(int quality) {
    int q = std::max(1, std::min(quality, 100));
    if (q >= 100) {
        std::array<float, 64> identity{};
        identity.fill(1.0f);
        return identity;
    }

    int scaled = q;
    if (q < 50) {
        scaled = 5000 / q;
    } else {
        scaled = 200 - q * 2;
    }

    std::array<float, 64> table{};
    for (std::size_t i = 0; i < 64; ++i) {
        int value = (kQuantTableQ50[i] * scaled + 50) / 100;
        value = std::max(1, std::min(value, 255));
        table[i] = static_cast<float>(value);
    }
    table[0] = 1.0f; // preserve DC
    return table;
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

inline std::size_t clamp_index(std::size_t value, std::size_t limit) noexcept {
    if (value >= limit) {
        return limit - 1;
    }
    return value;
}

} // namespace

Image dct8x8_hf_attenuate(const Image &input, int quality) {
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

    const std::array<float, 64> quant_table = build_quality_table(quality);

    const std::size_t tiles_x = (width + 7) / 8;
    const std::size_t tiles_y = (height + 7) / 8;
    const std::size_t tile_count = tiles_x * tiles_y;

    const float *src = input.pixels.data();
    float *dst = output.pixels.data();

    parallel_for(0, tile_count, [&](std::size_t tile_index) {
        const std::size_t tile_y = tile_index / tiles_x;
        const std::size_t tile_x = tile_index % tiles_x;
        const std::size_t base_x = tile_x * 8;
        const std::size_t base_y = tile_y * 8;

        for (std::size_t c = 0; c < channels; ++c) {
            float block[64];
            for (std::size_t yy = 0; yy < 8; ++yy) {
                const std::size_t src_y = clamp_index(base_y + yy, height);
                for (std::size_t xx = 0; xx < 8; ++xx) {
                    const std::size_t src_x = clamp_index(base_x + xx, width);
                    const std::size_t idx = (src_y * width + src_x) * channels + c;
                    block[yy * 8 + xx] = src[idx];
                }
            }

            forward_dct(block);

            if (quality < 100) {
                for (std::size_t i = 0; i < 64; ++i) {
                    if (i == 0) {
                        continue;
                    }
                    const float q = quant_table[i];
                    const float scaled = block[i] / q;
                    const float quantized = std::nearbyint(scaled) * q;
                    block[i] = quantized;
                }
            }

            inverse_dct(block);

            for (std::size_t yy = 0; yy < 8; ++yy) {
                const std::size_t dst_y = base_y + yy;
                if (dst_y >= height) {
                    break;
                }
                for (std::size_t xx = 0; xx < 8; ++xx) {
                    const std::size_t dst_x = base_x + xx;
                    if (dst_x >= width) {
                        break;
                    }
                    const std::size_t idx = (dst_y * width + dst_x) * channels + c;
                    dst[idx] = block[yy * 8 + xx];
                }
            }
        }
    });

    return output;
}

} // namespace pixmask

extern "C" {

void pixmask_dct8x8_hf_attenuate(const float *src,
                                 float *dst,
                                 std::size_t width,
                                 std::size_t height,
                                 std::size_t channels,
                                 int quality) noexcept {
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

    pixmask::Image filtered = pixmask::dct8x8_hf_attenuate(input, quality);
    const std::size_t to_copy = std::min<std::size_t>(element_count, filtered.pixels.size());
    std::copy_n(filtered.pixels.data(), to_copy, dst);
}

}
