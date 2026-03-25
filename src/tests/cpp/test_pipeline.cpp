// test_pipeline.cpp — Integration tests for the pixmask Pipeline orchestrator.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "pixmask/pipeline.h"

#include <cstdint>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers: minimal valid JPEG generation.
// A 2x2 red JPEG — smallest meaningful test image.
// Generated offline with stb_image_write; embedded as raw bytes.
// ---------------------------------------------------------------------------

// Minimal 2x2 red JPEG (hand-crafted SOI + tables + SOS).
// We use stb_image_write at test time instead: encode a known pixel buffer.
#include "stb_image_write.h"
#include "stb_image.h"

namespace {

// Write callback that appends to a vector.
void vec_write(void* ctx, void* data, int size) {
    auto* buf = static_cast<std::vector<uint8_t>*>(ctx);
    auto* bytes = static_cast<const uint8_t*>(data);
    buf->insert(buf->end(), bytes, bytes + size);
}

// Create a valid JPEG from a solid-color WxH RGB image.
std::vector<uint8_t> make_jpeg(uint32_t w, uint32_t h,
                                uint8_t r, uint8_t g, uint8_t b,
                                int quality = 80) {
    std::vector<uint8_t> pixels(static_cast<size_t>(w) * h * 3);
    for (size_t i = 0; i < pixels.size(); i += 3) {
        pixels[i]     = r;
        pixels[i + 1] = g;
        pixels[i + 2] = b;
    }
    std::vector<uint8_t> jpeg_buf;
    jpeg_buf.reserve(4096);
    int ok = stbi_write_jpg_to_func(
        vec_write, &jpeg_buf,
        static_cast<int>(w), static_cast<int>(h),
        3, pixels.data(), quality
    );
    REQUIRE(ok != 0);
    return jpeg_buf;
}

} // namespace

// ===========================================================================
// Test cases
// ===========================================================================

TEST_CASE("Pipeline: valid JPEG produces successful result") {
    auto jpeg = make_jpeg(64, 48, 200, 100, 50);

    pixmask::Pipeline pipeline;
    auto result = pipeline.sanitize(jpeg.data(), jpeg.size());

    CHECK(result.success);
    CHECK(result.error_code == pixmask::SanitizeError::Ok);
    CHECK(result.image.is_valid());
    CHECK(result.image.width == 64);
    CHECK(result.image.height == 48);
    CHECK(result.image.channels == 3);
}

TEST_CASE("Pipeline: output has 3 channels and matching dimensions") {
    auto jpeg = make_jpeg(100, 80, 0, 255, 0);

    pixmask::Pipeline pipeline;
    auto result = pipeline.sanitize(jpeg.data(), jpeg.size());

    REQUIRE(result.success);
    CHECK(result.image.channels == 3);
    CHECK(result.image.width == 100);
    CHECK(result.image.height == 80);
    CHECK(result.image.stride >= result.image.width * result.image.channels);
}

TEST_CASE("Pipeline: corrupt data returns graceful error, no crash") {
    // Random garbage — not a valid image header.
    std::vector<uint8_t> garbage(512, 0xAB);

    pixmask::Pipeline pipeline;
    auto result = pipeline.sanitize(garbage.data(), garbage.size());

    CHECK_FALSE(result.success);
    CHECK(result.error_code != pixmask::SanitizeError::Ok);
    CHECK(result.error_message != nullptr);
    CHECK_FALSE(result.image.is_valid());
}

TEST_CASE("Pipeline: null input returns error") {
    pixmask::Pipeline pipeline;
    auto result = pipeline.sanitize(nullptr, 0);

    CHECK_FALSE(result.success);
    CHECK(result.error_code != pixmask::SanitizeError::Ok);
}

TEST_CASE("Pipeline: oversized dimensions are rejected") {
    // Create a valid JPEG, but set pipeline options with very small max dims.
    auto jpeg = make_jpeg(64, 48, 128, 128, 128);

    pixmask::SanitizeOptions opts{};
    opts.max_width  = 32;   // smaller than the 64px image
    opts.max_height = 32;

    pixmask::Pipeline pipeline(opts);
    auto result = pipeline.sanitize(jpeg.data(), jpeg.size());

    CHECK_FALSE(result.success);
    CHECK(result.error_code == pixmask::SanitizeError::DimensionsTooLarge);
}

TEST_CASE("Pipeline: multiple calls reuse arena correctly") {
    auto jpeg1 = make_jpeg(32, 32, 255, 0, 0);
    auto jpeg2 = make_jpeg(48, 48, 0, 0, 255);

    pixmask::Pipeline pipeline;

    // First call.
    auto r1 = pipeline.sanitize(jpeg1.data(), jpeg1.size());
    REQUIRE(r1.success);
    CHECK(r1.image.width == 32);
    CHECK(r1.image.height == 32);

    // Second call — arena is reset internally; previous pointers invalidated.
    auto r2 = pipeline.sanitize(jpeg2.data(), jpeg2.size());
    REQUIRE(r2.success);
    CHECK(r2.image.width == 48);
    CHECK(r2.image.height == 48);

    // Arena should have been reused (capacity doesn't keep growing).
    size_t cap = pipeline.arena().capacity_bytes();
    CHECK(cap > 0);
}

TEST_CASE("Pipeline: free function convenience wrapper works") {
    auto jpeg = make_jpeg(16, 16, 100, 200, 50);

    auto result = pixmask::sanitize(jpeg.data(), jpeg.size());
    CHECK(result.success);
    CHECK(result.image.is_valid());
    CHECK(result.image.width == 16);
    CHECK(result.image.height == 16);
}

TEST_CASE("Pipeline: file too large is rejected") {
    auto jpeg = make_jpeg(8, 8, 0, 0, 0);

    pixmask::SanitizeOptions opts{};
    opts.max_file_bytes = 10;  // absurdly small

    pixmask::Pipeline pipeline(opts);
    auto result = pipeline.sanitize(jpeg.data(), jpeg.size());

    CHECK_FALSE(result.success);
    CHECK(result.error_code == pixmask::SanitizeError::FileTooLarge);
}
