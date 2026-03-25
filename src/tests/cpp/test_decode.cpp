// Unit tests for Stage 1: decode_image()
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "pixmask/decode.h"
#include "pixmask/arena.h"
#include "pixmask/types.h"

#include <cstdint>
#include <cstring>
#include <vector>

// We need stb_image_write to generate test images in-memory.
// STB_IMAGE_WRITE_IMPLEMENTATION lives in jpeg_roundtrip.cpp (via pixmask_core);
// only include the header here for declarations.
#include "stb_image_write.h"

// ---------------------------------------------------------------------------
// Helper: accumulate stb_image_write output into a vector
// ---------------------------------------------------------------------------
namespace {

void write_callback(void* context, void* data, int size) {
    auto* buf = static_cast<std::vector<uint8_t>*>(context);
    const auto* bytes = static_cast<const uint8_t*>(data);
    buf->insert(buf->end(), bytes, bytes + size);
}

// Generate a minimal JPEG from an RGB pixel buffer.
std::vector<uint8_t> make_jpeg(const uint8_t* pixels, int w, int h, int quality = 80) {
    std::vector<uint8_t> out;
    out.reserve(static_cast<size_t>(w * h));
    stbi_write_jpg_to_func(write_callback, &out, w, h, 3, pixels, quality);
    return out;
}

// Generate a minimal PNG from an RGB pixel buffer.
std::vector<uint8_t> make_png(const uint8_t* pixels, int w, int h, int channels = 3) {
    std::vector<uint8_t> out;
    out.reserve(static_cast<size_t>(w * h * channels * 2));
    stbi_write_png_to_func(write_callback, &out, w, h, channels, pixels, w * channels);
    return out;
}

// Fill a pixel buffer with a simple gradient.
std::vector<uint8_t> gradient_rgb(int w, int h) {
    std::vector<uint8_t> pixels(static_cast<size_t>(w * h * 3));
    for (int i = 0; i < w * h * 3; ++i) {
        pixels[static_cast<size_t>(i)] = static_cast<uint8_t>((i * 7) % 256);
    }
    return pixels;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_CASE("decode valid JPEG") {
    constexpr int W = 16, H = 16;
    auto pixels = gradient_rgb(W, H);
    auto jpeg = make_jpeg(pixels.data(), W, H);
    REQUIRE(!jpeg.empty());

    pixmask::Arena arena(1 << 20);  // 1 MB
    const char* err = nullptr;
    auto view = pixmask::decode_image(jpeg.data(), jpeg.size(), arena, &err);

    CHECK(err == nullptr);
    REQUIRE(view.is_valid());
    CHECK(view.width == W);
    CHECK(view.height == H);
    CHECK(view.channels == 3);
    CHECK(view.stride >= view.width * view.channels);
    // Stride must be 64-byte aligned.
    CHECK((view.stride % 64) == 0);
}

TEST_CASE("decode valid PNG") {
    constexpr int W = 8, H = 8;
    auto pixels = gradient_rgb(W, H);
    auto png = make_png(pixels.data(), W, H);
    REQUIRE(!png.empty());

    pixmask::Arena arena(1 << 20);
    const char* err = nullptr;
    auto view = pixmask::decode_image(png.data(), png.size(), arena, &err);

    CHECK(err == nullptr);
    REQUIRE(view.is_valid());
    CHECK(view.width == W);
    CHECK(view.height == H);
    CHECK(view.channels == 3);
    CHECK((view.stride % 64) == 0);

    // PNG is lossless: decoded pixels (within the tightly-packed region)
    // must exactly match the source.
    const uint32_t src_stride = W * 3;
    for (int row = 0; row < H; ++row) {
        const uint8_t* src_row = pixels.data() + row * src_stride;
        const uint8_t* dec_row = view.row(static_cast<uint32_t>(row));
        CHECK(std::memcmp(src_row, dec_row, src_stride) == 0);
    }
}

TEST_CASE("decode rejects corrupt data") {
    // Random garbage that is not a valid image.
    std::vector<uint8_t> garbage = {0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03};

    pixmask::Arena arena(1 << 20);
    const char* err = nullptr;
    auto view = pixmask::decode_image(garbage.data(), garbage.size(), arena, &err);

    CHECK_FALSE(view.is_valid());
    CHECK(view.data == nullptr);
    CHECK(err != nullptr);  // error message should be set
}

TEST_CASE("decode rejects null input") {
    pixmask::Arena arena(1 << 20);
    const char* err = nullptr;
    auto view = pixmask::decode_image(nullptr, 0, arena, &err);

    CHECK_FALSE(view.is_valid());
    CHECK(view.data == nullptr);
    CHECK(err != nullptr);
}

TEST_CASE("decode rejects empty buffer") {
    pixmask::Arena arena(1 << 20);
    uint8_t dummy = 0;
    const char* err = nullptr;
    auto view = pixmask::decode_image(&dummy, 0, arena, &err);

    CHECK_FALSE(view.is_valid());
    CHECK(err != nullptr);
}

TEST_CASE("output is always 3 channels regardless of input format") {
    SUBCASE("RGBA PNG input produces RGB output") {
        constexpr int W = 4, H = 4;
        // Build an RGBA pixel buffer.
        std::vector<uint8_t> rgba(static_cast<size_t>(W * H * 4));
        for (size_t i = 0; i < rgba.size(); i += 4) {
            rgba[i + 0] = 200;  // R
            rgba[i + 1] = 100;  // G
            rgba[i + 2] = 50;   // B
            rgba[i + 3] = 255;  // A (opaque)
        }
        auto png = make_png(rgba.data(), W, H, 4);
        REQUIRE(!png.empty());

        pixmask::Arena arena(1 << 20);
        const char* err = nullptr;
        auto view = pixmask::decode_image(png.data(), png.size(), arena, &err);

        CHECK(err == nullptr);
        REQUIRE(view.is_valid());
        CHECK(view.channels == 3);
        CHECK(view.width == W);
        CHECK(view.height == H);

        // Check the first pixel's RGB values match the source.
        CHECK(view.data[0] == 200);  // R
        CHECK(view.data[1] == 100);  // G
        CHECK(view.data[2] == 50);   // B
    }

    SUBCASE("grayscale JPEG input produces RGB output") {
        // stb_image_write doesn't have a direct grayscale JPEG writer,
        // but we can make a "gray" image as RGB with R=G=B and verify
        // decode still gives 3 channels.
        constexpr int W = 8, H = 8;
        std::vector<uint8_t> gray_as_rgb(static_cast<size_t>(W * H * 3));
        for (size_t i = 0; i < gray_as_rgb.size(); i += 3) {
            auto val = static_cast<uint8_t>((i / 3) * 3 % 256);
            gray_as_rgb[i + 0] = val;
            gray_as_rgb[i + 1] = val;
            gray_as_rgb[i + 2] = val;
        }
        auto jpeg = make_jpeg(gray_as_rgb.data(), W, H);
        REQUIRE(!jpeg.empty());

        pixmask::Arena arena(1 << 20);
        auto view = pixmask::decode_image(jpeg.data(), jpeg.size(), arena);

        REQUIRE(view.is_valid());
        CHECK(view.channels == 3);
    }
}

TEST_CASE("stride is 64-byte aligned for various widths") {
    // Test several widths to exercise alignment padding.
    const int widths[] = {1, 7, 21, 32, 64, 100, 128, 200};

    for (int w : widths) {
        CAPTURE(w);
        constexpr int H = 2;
        auto pixels = gradient_rgb(w, H);
        auto png = make_png(pixels.data(), w, H);
        REQUIRE(!png.empty());

        pixmask::Arena arena(1 << 20);
        auto view = pixmask::decode_image(png.data(), png.size(), arena);

        REQUIRE(view.is_valid());
        CHECK(view.stride >= view.width * view.channels);
        CHECK((view.stride % 64) == 0);
    }
}

TEST_CASE("decode truncated JPEG fails gracefully") {
    constexpr int W = 16, H = 16;
    auto pixels = gradient_rgb(W, H);
    auto jpeg = make_jpeg(pixels.data(), W, H);
    REQUIRE(jpeg.size() > 20);

    // Truncate to half the data.
    size_t truncated_len = jpeg.size() / 2;

    pixmask::Arena arena(1 << 20);
    const char* err = nullptr;
    auto view = pixmask::decode_image(jpeg.data(), truncated_len, arena, &err);

    // stb_image may or may not decode a truncated JPEG (it tries to be lenient).
    // If it fails, data must be null and error set.
    // If it succeeds, channels must still be 3.
    if (!view.is_valid()) {
        CHECK(view.data == nullptr);
        // err might or might not be set depending on stb behavior
    } else {
        CHECK(view.channels == 3);
    }
}
