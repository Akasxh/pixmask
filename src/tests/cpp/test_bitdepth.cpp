#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "pixmask/bitdepth.h"
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

// Helper: create a contiguous ImageView from a flat buffer.
static pixmask::ImageView make_image(uint8_t* data, uint32_t width,
                                     uint32_t height, uint32_t channels) {
    return pixmask::ImageView{data, width, height, channels,
                              width * channels};
}

// Helper: compute expected value for a single pixel at a given bit depth.
static uint8_t expected_pixel(uint8_t value, uint8_t bits) {
    const uint8_t mask =
        static_cast<uint8_t>((0xFF << (8 - bits)) & 0xFF);
    return value & mask;
}

// ---------------------------------------------------------------------------
// Test all bit depths 1-8
// ---------------------------------------------------------------------------

TEST_CASE("reduce_bit_depth: all depths 1-8 on full range") {
    for (uint8_t bits = 1; bits <= 8; ++bits) {
        CAPTURE(bits);
        std::vector<uint8_t> pixels(256);
        std::iota(pixels.begin(), pixels.end(), static_cast<uint8_t>(0));
        auto image = make_image(pixels.data(), 256, 1, 1);

        pixmask::reduce_bit_depth(image, bits);

        for (int i = 0; i < 256; ++i) {
            CAPTURE(i);
            CHECK(pixels[static_cast<size_t>(i)] ==
                  expected_pixel(static_cast<uint8_t>(i), bits));
        }
    }
}

// ---------------------------------------------------------------------------
// Edge values: 0, 1, 127, 128, 254, 255
// ---------------------------------------------------------------------------

TEST_CASE("reduce_bit_depth: edge values") {
    const std::vector<uint8_t> edge_values = {0, 1, 127, 128, 254, 255};

    for (uint8_t bits = 1; bits <= 7; ++bits) {
        CAPTURE(bits);
        std::vector<uint8_t> pixels(edge_values);
        auto image = make_image(pixels.data(),
                                static_cast<uint32_t>(pixels.size()), 1, 1);

        pixmask::reduce_bit_depth(image, bits);

        for (size_t j = 0; j < edge_values.size(); ++j) {
            CAPTURE(edge_values[j]);
            CHECK(pixels[j] == expected_pixel(edge_values[j], bits));
        }
    }
}

// ---------------------------------------------------------------------------
// Specific formula verification: bit_depth=5
// ---------------------------------------------------------------------------

TEST_CASE("reduce_bit_depth: bit_depth=5 specific values") {
    // 5-bit mask = 0xF8 = 11111000
    std::vector<uint8_t> pixels = {255, 128, 7, 0, 248, 127};
    auto image = make_image(pixels.data(),
                            static_cast<uint32_t>(pixels.size()), 1, 1);

    pixmask::reduce_bit_depth(image, 5);

    CHECK(pixels[0] == 248);  // 255 & 0xF8
    CHECK(pixels[1] == 128);  // 128 & 0xF8
    CHECK(pixels[2] == 0);    // 7   & 0xF8
    CHECK(pixels[3] == 0);    // 0   & 0xF8
    CHECK(pixels[4] == 248);  // 248 & 0xF8
    CHECK(pixels[5] == 120);  // 127 & 0xF8
}

// ---------------------------------------------------------------------------
// 8-bit depth is identity (no-op)
// ---------------------------------------------------------------------------

TEST_CASE("reduce_bit_depth: 8-bit is identity") {
    std::vector<uint8_t> pixels(256);
    std::iota(pixels.begin(), pixels.end(), static_cast<uint8_t>(0));
    const auto original = pixels;
    auto image = make_image(pixels.data(), 256, 1, 1);

    pixmask::reduce_bit_depth(image, 8);

    CHECK(pixels == original);
}

// ---------------------------------------------------------------------------
// Full image: multi-row, multi-channel
// ---------------------------------------------------------------------------

TEST_CASE("reduce_bit_depth: full RGB image") {
    constexpr uint32_t W = 64;
    constexpr uint32_t H = 48;
    constexpr uint32_t C = 3;
    std::vector<uint8_t> pixels(W * H * C);

    // Fill with a pattern
    for (size_t i = 0; i < pixels.size(); ++i) {
        pixels[i] = static_cast<uint8_t>(i & 0xFF);
    }
    const auto original = pixels;
    auto image = make_image(pixels.data(), W, H, C);

    pixmask::reduce_bit_depth(image, 4);

    // 4-bit mask = 0xF0
    for (size_t i = 0; i < pixels.size(); ++i) {
        CHECK(pixels[i] == (original[i] & 0xF0));
    }
}

// ---------------------------------------------------------------------------
// Non-aligned width: 13 pixels (tests SIMD tail handling)
// ---------------------------------------------------------------------------

TEST_CASE("reduce_bit_depth: non-aligned width 13 pixels") {
    constexpr uint32_t W = 13;
    constexpr uint32_t H = 3;
    constexpr uint32_t C = 3;
    // Total bytes per row: 13 * 3 = 39 — not a power of 2
    std::vector<uint8_t> pixels(W * H * C, 0xFF);
    auto image = make_image(pixels.data(), W, H, C);

    pixmask::reduce_bit_depth(image, 4);

    // 4-bit mask = 0xF0; 0xFF & 0xF0 = 0xF0
    for (uint8_t v : pixels) {
        CHECK(v == 0xF0);
    }
}

// ---------------------------------------------------------------------------
// Non-aligned width with varied data (ensures tail correctness, not just 0xFF)
// ---------------------------------------------------------------------------

TEST_CASE("reduce_bit_depth: non-aligned width 13 varied data") {
    constexpr uint32_t W = 13;
    constexpr uint32_t H = 2;
    constexpr uint32_t C = 1;
    std::vector<uint8_t> pixels(W * H * C);
    for (size_t i = 0; i < pixels.size(); ++i) {
        pixels[i] = static_cast<uint8_t>(i * 17 + 3);  // arbitrary pattern
    }
    const auto original = pixels;
    auto image = make_image(pixels.data(), W, H, C);

    pixmask::reduce_bit_depth(image, 6);

    // 6-bit mask = 0xFC
    for (size_t i = 0; i < pixels.size(); ++i) {
        CHECK(pixels[i] == (original[i] & 0xFC));
    }
}

// ---------------------------------------------------------------------------
// Stride > width * channels (padding between rows)
// ---------------------------------------------------------------------------

TEST_CASE("reduce_bit_depth: respects stride with padding") {
    constexpr uint32_t W = 4;
    constexpr uint32_t H = 3;
    constexpr uint32_t C = 3;
    constexpr uint32_t row_bytes = W * C;        // 12
    constexpr uint32_t stride = 16;              // 4 bytes padding per row
    std::vector<uint8_t> buffer(stride * H, 0xAA);  // fill with sentinel

    // Write pixel data into the non-padded portion of each row
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < row_bytes; ++x) {
            buffer[y * stride + x] = 0xFF;
        }
    }

    pixmask::ImageView image{buffer.data(), W, H, C, stride};
    pixmask::reduce_bit_depth(image, 5);

    // 5-bit mask = 0xF8; 0xFF & 0xF8 = 0xF8
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < row_bytes; ++x) {
            CHECK(buffer[y * stride + x] == 0xF8);
        }
        // Padding bytes should be UNTOUCHED (still 0xAA)
        for (uint32_t x = row_bytes; x < stride; ++x) {
            CHECK(buffer[y * stride + x] == 0xAA);
        }
    }
}

// ---------------------------------------------------------------------------
// 1-bit depth: most aggressive reduction
// ---------------------------------------------------------------------------

TEST_CASE("reduce_bit_depth: 1-bit reduction") {
    // 1-bit mask = 0x80 = 10000000
    std::vector<uint8_t> pixels = {0, 1, 127, 128, 200, 255};
    auto image = make_image(pixels.data(),
                            static_cast<uint32_t>(pixels.size()), 1, 1);

    pixmask::reduce_bit_depth(image, 1);

    CHECK(pixels[0] == 0);    // 0   & 0x80
    CHECK(pixels[1] == 0);    // 1   & 0x80
    CHECK(pixels[2] == 0);    // 127 & 0x80
    CHECK(pixels[3] == 128);  // 128 & 0x80
    CHECK(pixels[4] == 128);  // 200 & 0x80
    CHECK(pixels[5] == 128);  // 255 & 0x80
}
