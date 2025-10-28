#pragma once

#include <array>
#include <cstddef>

namespace pixmask {
namespace sr {

inline constexpr std::size_t kInputChannels = 3;
inline constexpr std::size_t kUpscaleFactor = 2;
inline constexpr std::size_t kConv1OutChannels = 16;
inline constexpr std::size_t kConv2OutChannels = 16;
inline constexpr std::size_t kConv3OutChannels = kInputChannels * kUpscaleFactor * kUpscaleFactor;
inline constexpr std::size_t kKernelSize = 3;

constexpr std::size_t conv_weight_count(std::size_t out_channels, std::size_t in_channels) noexcept {
    return out_channels * kKernelSize * kKernelSize * in_channels;
}

constexpr std::size_t conv_index(std::size_t out_channel,
                                 std::size_t ky,
                                 std::size_t kx,
                                 std::size_t in_channel,
                                 std::size_t in_channels) noexcept {
    return (((out_channel * kKernelSize) + ky) * kKernelSize + kx) * in_channels + in_channel;
}

inline constexpr auto make_conv1_weights() {
    std::array<float, conv_weight_count(kConv1OutChannels, kInputChannels)> data{};
    for (std::size_t channel = 0; channel < kInputChannels; ++channel) {
        const std::size_t base = channel * 5;
        data[conv_index(base + 0, 1, 1, channel, kInputChannels)] = 1.0f; // center
        data[conv_index(base + 1, 0, 1, channel, kInputChannels)] = 1.0f; // up
        data[conv_index(base + 2, 2, 1, channel, kInputChannels)] = 1.0f; // down
        data[conv_index(base + 3, 1, 0, channel, kInputChannels)] = 1.0f; // left
        data[conv_index(base + 4, 1, 2, channel, kInputChannels)] = 1.0f; // right
    }

    // Luminance helper
    constexpr float inv3 = 1.0f / 3.0f;
    data[conv_index(15, 1, 1, 0, kInputChannels)] = inv3;
    data[conv_index(15, 1, 1, 1, kInputChannels)] = inv3;
    data[conv_index(15, 1, 1, 2, kInputChannels)] = inv3;

    return data;
}

inline constexpr auto make_conv2_weights() {
    std::array<float, conv_weight_count(kConv2OutChannels, kConv1OutChannels)> data{};
    for (std::size_t channel = 0; channel < kConv2OutChannels; ++channel) {
        data[conv_index(channel, 1, 1, channel, kConv1OutChannels)] = 1.0f;
    }
    return data;
}

inline constexpr auto make_conv3_weights() {
    std::array<float, conv_weight_count(kConv3OutChannels, kConv2OutChannels)> data{};

    constexpr float kMain = 1.2f;
    constexpr float kStrong = -0.1f;
    constexpr float kWeak = -0.05f;
    constexpr float kLumaBlend = 0.05f;

    constexpr int strong_pairs[4][2] = {
        {1, 3}, // top-left emphasises up + left
        {1, 4}, // top-right emphasises up + right
        {2, 3}, // bottom-left emphasises down + left
        {2, 4}, // bottom-right emphasises down + right
    };

    constexpr int weak_pairs[4][2] = {
        {2, 4}, // remaining neighbors for TL
        {2, 3}, // remaining neighbors for TR
        {1, 4}, // remaining neighbors for BL
        {1, 3}, // remaining neighbors for BR
    };

    for (std::size_t channel = 0; channel < kInputChannels; ++channel) {
        const std::size_t feature_base = channel * 5;
        for (std::size_t orientation = 0; orientation < 4; ++orientation) {
            const std::size_t out_channel = channel * 4 + orientation;
            const std::size_t center_index = conv_index(out_channel, 1, 1, feature_base + 0, kConv2OutChannels);
            data[center_index] = kMain;

            for (int idx : strong_pairs[orientation]) {
                const std::size_t feature = feature_base + static_cast<std::size_t>(idx);
                data[conv_index(out_channel, 1, 1, feature, kConv2OutChannels)] = kStrong;
            }

            for (int idx : weak_pairs[orientation]) {
                const std::size_t feature = feature_base + static_cast<std::size_t>(idx);
                data[conv_index(out_channel, 1, 1, feature, kConv2OutChannels)] = kWeak;
            }

            // Blend a touch of shared luminance to maintain global brightness.
            data[conv_index(out_channel, 1, 1, 15, kConv2OutChannels)] = kLumaBlend;
        }
    }

    return data;
}

inline constexpr std::array<float, conv_weight_count(kConv1OutChannels, kInputChannels)> kConv1Weights =
    make_conv1_weights();
inline constexpr std::array<float, kConv1OutChannels> kConv1Bias{};

inline constexpr std::array<float, conv_weight_count(kConv2OutChannels, kConv1OutChannels)> kConv2Weights =
    make_conv2_weights();
inline constexpr std::array<float, kConv2OutChannels> kConv2Bias{};

inline constexpr std::array<float, conv_weight_count(kConv3OutChannels, kConv2OutChannels)> kConv3Weights =
    make_conv3_weights();
inline constexpr std::array<float, kConv3OutChannels> kConv3Bias{};

} // namespace sr
} // namespace pixmask
