// Unit tests for 3x3 median filter (Stage 4).
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "pixmask/median.h"

#include <algorithm>
#include <cstring>
#include <vector>

// Helper: create an ImageView backed by a vector with 64-byte aligned stride.
static pixmask::ImageView make_image(std::vector<uint8_t>& buf,
                                     uint32_t w, uint32_t h, uint32_t ch) {
    uint32_t stride = pixmask::aligned_stride(w, ch, 64);
    buf.assign(static_cast<size_t>(h) * stride, 0);
    pixmask::ImageView v;
    v.data     = buf.data();
    v.width    = w;
    v.height   = h;
    v.channels = ch;
    v.stride   = stride;
    return v;
}

// Reference median: brute-force nth_element on the 3x3 neighborhood.
static uint8_t ref_median_at(const pixmask::ImageView& img,
                             uint32_t x, uint32_t y, uint32_t c) {
    uint8_t vals[9];
    int idx = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        uint32_t yy = static_cast<uint32_t>(
            std::clamp(static_cast<int>(y) + dy, 0, static_cast<int>(img.height) - 1));
        for (int dx = -1; dx <= 1; ++dx) {
            uint32_t xx = static_cast<uint32_t>(
                std::clamp(static_cast<int>(x) + dx, 0, static_cast<int>(img.width) - 1));
            vals[idx++] = img.pixel(xx, yy)[c];
        }
    }
    std::nth_element(vals, vals + 4, vals + 9);
    return vals[4];
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

TEST_CASE("median: output dimensions match input") {
    pixmask::Arena arena(1 << 20);
    std::vector<uint8_t> buf;
    auto img = make_image(buf, 32, 24, 3);
    std::memset(img.data, 128, img.total_bytes());

    auto out = pixmask::median_filter_3x3(img, arena);
    CHECK(out.is_valid());
    CHECK(out.width    == img.width);
    CHECK(out.height   == img.height);
    CHECK(out.channels == img.channels);
    CHECK(out.stride   >= out.width * out.channels);
    // Stride should be 64-byte aligned.
    CHECK((out.stride % 64) == 0);
}

TEST_CASE("median: uniform image unchanged") {
    pixmask::Arena arena(1 << 20);
    std::vector<uint8_t> buf;
    auto img = make_image(buf, 64, 64, 1);
    // Fill with a constant value.
    for (uint32_t y = 0; y < img.height; ++y) {
        std::memset(img.row(y), 42, img.width);
    }

    auto out = pixmask::median_filter_3x3(img, arena);
    REQUIRE(out.is_valid());
    for (uint32_t y = 0; y < out.height; ++y) {
        for (uint32_t x = 0; x < out.width; ++x) {
            CHECK(out.pixel(x, y)[0] == 42);
        }
    }
}

TEST_CASE("median: single impulse noise suppressed") {
    pixmask::Arena arena(1 << 20);
    std::vector<uint8_t> buf;
    auto img = make_image(buf, 5, 5, 1);
    // All zeros except center pixel = 255.
    img.pixel(2, 2)[0] = 255;

    auto out = pixmask::median_filter_3x3(img, arena);
    REQUIRE(out.is_valid());
    // Center pixel neighborhood: 8 zeros, 1 x 255 -> median = 0.
    CHECK(out.pixel(2, 2)[0] == 0);
}

TEST_CASE("median: known 3x3 block") {
    pixmask::Arena arena(1 << 20);
    std::vector<uint8_t> buf;
    auto img = make_image(buf, 3, 3, 1);
    // Row-major: {100, 50, 200, 30, 180, 70, 90, 120, 10}
    const uint8_t pixels[] = {100, 50, 200, 30, 180, 70, 90, 120, 10};
    for (uint32_t y = 0; y < 3; ++y) {
        std::memcpy(img.row(y), pixels + y * 3, 3);
    }

    auto out = pixmask::median_filter_3x3(img, arena);
    REQUIRE(out.is_valid());
    // Center pixel sees all 9 values. Sorted: 10,30,50,70,90,100,120,180,200
    // Median (index 4) = 90.
    CHECK(out.pixel(1, 1)[0] == 90);
}

TEST_CASE("median: non-square image (7x5)") {
    pixmask::Arena arena(1 << 20);
    std::vector<uint8_t> buf;
    auto img = make_image(buf, 7, 5, 1);
    // Fill with pseudo-random data.
    for (uint32_t y = 0; y < img.height; ++y) {
        for (uint32_t x = 0; x < img.width; ++x) {
            img.pixel(x, y)[0] = static_cast<uint8_t>((x * 7 + y * 13 + 17) & 0xFF);
        }
    }

    auto out = pixmask::median_filter_3x3(img, arena);
    REQUIRE(out.is_valid());
    CHECK(out.width  == 7);
    CHECK(out.height == 5);

    // Verify all pixels match the reference implementation.
    for (uint32_t y = 0; y < out.height; ++y) {
        for (uint32_t x = 0; x < out.width; ++x) {
            uint8_t expected = ref_median_at(img, x, y, 0);
            CHECK(out.pixel(x, y)[0] == expected);
        }
    }
}

TEST_CASE("median: 1x1 image edge case") {
    pixmask::Arena arena(1 << 20);
    std::vector<uint8_t> buf;
    auto img = make_image(buf, 1, 1, 1);
    img.pixel(0, 0)[0] = 128;

    auto out = pixmask::median_filter_3x3(img, arena);
    REQUIRE(out.is_valid());
    // 1x1: all 9 neighbors are the same pixel (replicate border).
    CHECK(out.pixel(0, 0)[0] == 128);
}

TEST_CASE("median: RGB uniform image") {
    pixmask::Arena arena(1 << 20);
    std::vector<uint8_t> buf;
    auto img = make_image(buf, 16, 16, 3);
    for (uint32_t y = 0; y < img.height; ++y) {
        for (uint32_t x = 0; x < img.width; ++x) {
            img.pixel(x, y)[0] = 100;
            img.pixel(x, y)[1] = 150;
            img.pixel(x, y)[2] = 200;
        }
    }

    auto out = pixmask::median_filter_3x3(img, arena);
    REQUIRE(out.is_valid());
    for (uint32_t y = 0; y < out.height; ++y) {
        for (uint32_t x = 0; x < out.width; ++x) {
            CHECK(out.pixel(x, y)[0] == 100);
            CHECK(out.pixel(x, y)[1] == 150);
            CHECK(out.pixel(x, y)[2] == 200);
        }
    }
}

TEST_CASE("median: RGB matches reference per-channel") {
    pixmask::Arena arena(1 << 20);
    std::vector<uint8_t> buf;
    auto img = make_image(buf, 7, 5, 3);
    // Fill with pseudo-random per-channel data.
    for (uint32_t y = 0; y < img.height; ++y) {
        for (uint32_t x = 0; x < img.width; ++x) {
            img.pixel(x, y)[0] = static_cast<uint8_t>((x * 7 + y * 13 + 3)  & 0xFF);
            img.pixel(x, y)[1] = static_cast<uint8_t>((x * 11 + y * 5 + 19) & 0xFF);
            img.pixel(x, y)[2] = static_cast<uint8_t>((x * 3 + y * 17 + 41) & 0xFF);
        }
    }

    auto out = pixmask::median_filter_3x3(img, arena);
    REQUIRE(out.is_valid());
    for (uint32_t y = 0; y < out.height; ++y) {
        for (uint32_t x = 0; x < out.width; ++x) {
            for (uint32_t c = 0; c < 3; ++c) {
                uint8_t expected = ref_median_at(img, x, y, c);
                INFO("x=" << x << " y=" << y << " c=" << c);
                CHECK(out.pixel(x, y)[c] == expected);
            }
        }
    }
}

TEST_CASE("median: SIMD width crossing (width=33 grayscale)") {
    pixmask::Arena arena(1 << 20);
    std::vector<uint8_t> buf;
    auto img = make_image(buf, 33, 5, 1);
    for (uint32_t y = 0; y < img.height; ++y) {
        for (uint32_t x = 0; x < img.width; ++x) {
            img.pixel(x, y)[0] = static_cast<uint8_t>((x * 7 + y * 13 + 17) & 0xFF);
        }
    }

    auto out = pixmask::median_filter_3x3(img, arena);
    REQUIRE(out.is_valid());
    // Verify all pixels match the reference (exercises remainder loop).
    for (uint32_t y = 0; y < out.height; ++y) {
        for (uint32_t x = 0; x < out.width; ++x) {
            uint8_t expected = ref_median_at(img, x, y, 0);
            INFO("x=" << x << " y=" << y);
            CHECK(out.pixel(x, y)[0] == expected);
        }
    }
}

TEST_CASE("median: output is separate from input (not in-place)") {
    pixmask::Arena arena(1 << 20);
    std::vector<uint8_t> buf;
    auto img = make_image(buf, 8, 8, 1);
    for (uint32_t y = 0; y < img.height; ++y) {
        for (uint32_t x = 0; x < img.width; ++x) {
            img.pixel(x, y)[0] = static_cast<uint8_t>(x + y);
        }
    }

    auto out = pixmask::median_filter_3x3(img, arena);
    REQUIRE(out.is_valid());
    // Output data pointer must differ from input data pointer.
    CHECK(out.data != img.data);
}

TEST_CASE("median: 2x2 image") {
    pixmask::Arena arena(1 << 20);
    std::vector<uint8_t> buf;
    auto img = make_image(buf, 2, 2, 1);
    img.pixel(0, 0)[0] = 10;
    img.pixel(1, 0)[0] = 20;
    img.pixel(0, 1)[0] = 30;
    img.pixel(1, 1)[0] = 40;

    auto out = pixmask::median_filter_3x3(img, arena);
    REQUIRE(out.is_valid());
    // Verify each pixel against reference.
    for (uint32_t y = 0; y < 2; ++y) {
        for (uint32_t x = 0; x < 2; ++x) {
            uint8_t expected = ref_median_at(img, x, y, 0);
            CHECK(out.pixel(x, y)[0] == expected);
        }
    }
}
