#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "pixmask/validate.h"

#include <cstring>
#include <vector>

using namespace pixmask;

// =============================================================================
// Helper: build a minimal valid PNG header with given dimensions
// =============================================================================
static std::vector<uint8_t> make_png_header(uint32_t w, uint32_t h) {
    std::vector<uint8_t> buf(24, 0);
    // PNG signature
    const uint8_t sig[8] = { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A };
    std::memcpy(buf.data(), sig, 8);
    // IHDR chunk type at offset 12
    buf[12] = 'I'; buf[13] = 'H'; buf[14] = 'D'; buf[15] = 'R';
    // Width (big-endian) at offset 16
    buf[16] = static_cast<uint8_t>((w >> 24) & 0xFF);
    buf[17] = static_cast<uint8_t>((w >> 16) & 0xFF);
    buf[18] = static_cast<uint8_t>((w >>  8) & 0xFF);
    buf[19] = static_cast<uint8_t>((w      ) & 0xFF);
    // Height (big-endian) at offset 20
    buf[20] = static_cast<uint8_t>((h >> 24) & 0xFF);
    buf[21] = static_cast<uint8_t>((h >> 16) & 0xFF);
    buf[22] = static_cast<uint8_t>((h >>  8) & 0xFF);
    buf[23] = static_cast<uint8_t>((h      ) & 0xFF);
    return buf;
}

// =============================================================================
// Helper: build a minimal JPEG with SOI + APP0 + SOF0 carrying dimensions
// =============================================================================
static std::vector<uint8_t> make_jpeg_header(uint16_t w, uint16_t h) {
    // SOI(2) + APP0 marker(2) + len(2) + 5 bytes JFIF id + SOF0 marker(2)
    //        + len(2) + precision(1) + height(2) + width(2) + ncomp(1)
    //        + padding to satisfy the parser's conservative bounds check.
    std::vector<uint8_t> buf;
    // SOI
    buf.push_back(0xFF); buf.push_back(0xD8);
    // APP0 marker
    buf.push_back(0xFF); buf.push_back(0xE0);
    // APP0 length = 7 (2 len bytes + 5 payload)
    buf.push_back(0x00); buf.push_back(0x07);
    // 5 dummy payload bytes
    for (int i = 0; i < 5; ++i) buf.push_back(0x00);
    // SOF0 marker
    buf.push_back(0xFF); buf.push_back(0xC0);
    // SOF0 length = 8 (2 len + 1 precision + 2 height + 2 width + 1 ncomp)
    buf.push_back(0x00); buf.push_back(0x08);
    // precision
    buf.push_back(0x08);
    // height (big-endian)
    buf.push_back(static_cast<uint8_t>((h >> 8) & 0xFF));
    buf.push_back(static_cast<uint8_t>((h     ) & 0xFF));
    // width (big-endian)
    buf.push_back(static_cast<uint8_t>((w >> 8) & 0xFF));
    buf.push_back(static_cast<uint8_t>((w     ) & 0xFF));
    // number of components
    buf.push_back(0x03);
    // Padding: the JPEG parser checks pos + 2 + 7 <= len (conservative),
    // so we need at least 22 bytes total. Add padding to be safe.
    buf.push_back(0x00);
    buf.push_back(0x00);
    return buf;
}

// =============================================================================
// Helper: build a minimal WebP VP8 (lossy) header with given dimensions
// =============================================================================
static std::vector<uint8_t> make_webp_vp8_header(uint16_t w, uint16_t h) {
    std::vector<uint8_t> buf(30, 0);
    // RIFF
    buf[0] = 'R'; buf[1] = 'I'; buf[2] = 'F'; buf[3] = 'F';
    // file size (dummy)
    // WEBP
    buf[8] = 'W'; buf[9] = 'E'; buf[10] = 'B'; buf[11] = 'P';
    // VP8 chunk
    buf[12] = 'V'; buf[13] = 'P'; buf[14] = '8'; buf[15] = ' ';
    // VP8 start code at 23-25
    buf[23] = 0x9D; buf[24] = 0x01; buf[25] = 0x2A;
    // Width (LE, 14-bit) at 26-27. Value is (w-1) stored in lower 14 bits.
    uint16_t w_raw = static_cast<uint16_t>(w - 1);
    buf[26] = static_cast<uint8_t>(w_raw & 0xFF);
    buf[27] = static_cast<uint8_t>((w_raw >> 8) & 0xFF);
    // Height (LE, 14-bit) at 28-29
    uint16_t h_raw = static_cast<uint16_t>(h - 1);
    buf[28] = static_cast<uint8_t>(h_raw & 0xFF);
    buf[29] = static_cast<uint8_t>((h_raw >> 8) & 0xFF);
    return buf;
}

// =============================================================================
// Magic-byte detection
// =============================================================================

TEST_CASE("detect_format: null / undersized") {
    CHECK(detect_format(nullptr, 0)  == ImageFormat::Unknown);
    CHECK(detect_format(nullptr, 8)  == ImageFormat::Unknown);
    const uint8_t one = 0xFF;
    CHECK(detect_format(&one, 1)     == ImageFormat::Unknown);
}

TEST_CASE("detect_format: PNG signature") {
    const uint8_t sig[] = { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A };
    CHECK(detect_format(sig, sizeof(sig)) == ImageFormat::PNG);
}

TEST_CASE("detect_format: JPEG SOI") {
    const uint8_t sig[] = { 0xFF, 0xD8, 0xFF, 0xE0 };
    CHECK(detect_format(sig, sizeof(sig)) == ImageFormat::JPEG);
}

TEST_CASE("detect_format: WebP RIFF header") {
    const uint8_t sig[] = {
        0x52, 0x49, 0x46, 0x46,  // RIFF
        0x00, 0x00, 0x00, 0x00,  // file size (ignored)
        0x57, 0x45, 0x42, 0x50   // WEBP
    };
    CHECK(detect_format(sig, sizeof(sig)) == ImageFormat::WebP);
}

TEST_CASE("detect_format: garbage bytes") {
    const uint8_t garbage[] = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07 };
    CHECK(detect_format(garbage, sizeof(garbage)) == ImageFormat::Unknown);
}

// =============================================================================
// validate_input: top-level gating
// =============================================================================

TEST_CASE("validate_input: null pointer") {
    auto r = validate_input(nullptr, 0);
    CHECK(r.error == ValidationError::NullInput);
    CHECK_FALSE(r.ok());
}

TEST_CASE("validate_input: data too small") {
    const uint8_t one = 0xAB;
    auto r = validate_input(&one, 1);
    CHECK(r.error == ValidationError::FileTooSmall);
}

TEST_CASE("validate_input: WebP rejected in v0.1") {
    const uint8_t sig[] = {
        0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00,
        0x57, 0x45, 0x42, 0x50
    };
    auto r = validate_input(sig, sizeof(sig));
    CHECK(r.error == ValidationError::UnsupportedFormat);
    CHECK(r.format == ImageFormat::WebP);
}

TEST_CASE("validate_input: unknown format") {
    const uint8_t garbage[16] = {};
    auto r = validate_input(garbage, sizeof(garbage));
    CHECK(r.error == ValidationError::UnknownFormat);
}

TEST_CASE("validate_input: file too large") {
    // Use a valid PNG header but claim size exceeds limit
    auto buf = make_png_header(100, 100);
    auto r = validate_input(buf.data(), buf.size(),
                            kMaxDimension, kMaxDimension,
                            /*max_file=*/10, // 10 bytes -- smaller than header
                            kMaxDecompRatio);
    CHECK(r.error == ValidationError::FileTooLarge);
}

// =============================================================================
// PNG dimension extraction
// =============================================================================

TEST_CASE("check_dimensions: PNG valid dimensions") {
    auto buf = make_png_header(1920, 1080);
    uint32_t w = 0, h = 0;
    auto err = check_dimensions(buf.data(), buf.size(), ImageFormat::PNG,
                                kMaxDimension, kMaxDimension, w, h);
    CHECK(err == ValidationError::Ok);
    CHECK(w == 1920);
    CHECK(h == 1080);
}

TEST_CASE("check_dimensions: PNG zero width rejected") {
    auto buf = make_png_header(0, 1);
    uint32_t w = 0, h = 0;
    auto err = check_dimensions(buf.data(), buf.size(), ImageFormat::PNG,
                                kMaxDimension, kMaxDimension, w, h);
    CHECK(err == ValidationError::ZeroDimension);
}

TEST_CASE("check_dimensions: PNG zero height rejected") {
    auto buf = make_png_header(1, 0);
    uint32_t w = 0, h = 0;
    auto err = check_dimensions(buf.data(), buf.size(), ImageFormat::PNG,
                                kMaxDimension, kMaxDimension, w, h);
    CHECK(err == ValidationError::ZeroDimension);
}

TEST_CASE("check_dimensions: PNG oversized width (8193) rejected") {
    auto buf = make_png_header(8193, 100);
    uint32_t w = 0, h = 0;
    auto err = check_dimensions(buf.data(), buf.size(), ImageFormat::PNG,
                                kMaxDimension, kMaxDimension, w, h);
    CHECK(err == ValidationError::WidthTooLarge);
}

TEST_CASE("check_dimensions: PNG oversized height (8193) rejected") {
    auto buf = make_png_header(100, 8193);
    uint32_t w = 0, h = 0;
    auto err = check_dimensions(buf.data(), buf.size(), ImageFormat::PNG,
                                kMaxDimension, kMaxDimension, w, h);
    CHECK(err == ValidationError::HeightTooLarge);
}

TEST_CASE("check_dimensions: PNG max dimension (8192) accepted") {
    auto buf = make_png_header(8192, 8192);
    uint32_t w = 0, h = 0;
    auto err = check_dimensions(buf.data(), buf.size(), ImageFormat::PNG,
                                kMaxDimension, kMaxDimension, w, h);
    CHECK(err == ValidationError::Ok);
    CHECK(w == 8192);
    CHECK(h == 8192);
}

TEST_CASE("check_dimensions: PNG truncated header") {
    auto buf = make_png_header(100, 100);
    buf.resize(16); // chop off dimension bytes
    uint32_t w = 0, h = 0;
    auto err = check_dimensions(buf.data(), buf.size(), ImageFormat::PNG,
                                kMaxDimension, kMaxDimension, w, h);
    CHECK(err == ValidationError::DimensionReadFailed);
}

TEST_CASE("check_dimensions: PNG corrupt IHDR tag") {
    auto buf = make_png_header(100, 100);
    // Corrupt the IHDR chunk type
    buf[12] = 'X'; buf[13] = 'X'; buf[14] = 'X'; buf[15] = 'X';
    uint32_t w = 0, h = 0;
    auto err = check_dimensions(buf.data(), buf.size(), ImageFormat::PNG,
                                kMaxDimension, kMaxDimension, w, h);
    CHECK(err == ValidationError::DimensionReadFailed);
}

// =============================================================================
// JPEG dimension extraction
// =============================================================================

TEST_CASE("check_dimensions: JPEG valid dimensions") {
    auto buf = make_jpeg_header(640, 480);
    uint32_t w = 0, h = 0;
    auto err = check_dimensions(buf.data(), buf.size(), ImageFormat::JPEG,
                                kMaxDimension, kMaxDimension, w, h);
    CHECK(err == ValidationError::Ok);
    CHECK(w == 640);
    CHECK(h == 480);
}

TEST_CASE("check_dimensions: JPEG oversized width rejected") {
    auto buf = make_jpeg_header(8193, 100);
    uint32_t w = 0, h = 0;
    auto err = check_dimensions(buf.data(), buf.size(), ImageFormat::JPEG,
                                kMaxDimension, kMaxDimension, w, h);
    CHECK(err == ValidationError::WidthTooLarge);
}

TEST_CASE("check_dimensions: JPEG zero dimensions rejected") {
    auto buf = make_jpeg_header(0, 480);
    uint32_t w = 0, h = 0;
    auto err = check_dimensions(buf.data(), buf.size(), ImageFormat::JPEG,
                                kMaxDimension, kMaxDimension, w, h);
    CHECK(err == ValidationError::ZeroDimension);
}

TEST_CASE("check_dimensions: JPEG truncated before SOF") {
    // SOI + partial APP0 -- no SOF marker reachable
    const uint8_t trunc[] = { 0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x04, 0x00, 0x00 };
    uint32_t w = 0, h = 0;
    auto err = check_dimensions(trunc, sizeof(trunc), ImageFormat::JPEG,
                                kMaxDimension, kMaxDimension, w, h);
    CHECK(err == ValidationError::DimensionReadFailed);
}

// =============================================================================
// WebP dimension extraction (VP8 lossy)
// =============================================================================

TEST_CASE("check_dimensions: WebP VP8 valid dimensions") {
    auto buf = make_webp_vp8_header(800, 600);
    uint32_t w = 0, h = 0;
    auto err = check_dimensions(buf.data(), buf.size(), ImageFormat::WebP,
                                kMaxDimension, kMaxDimension, w, h);
    CHECK(err == ValidationError::Ok);
    CHECK(w == 800);
    CHECK(h == 600);
}

TEST_CASE("check_dimensions: WebP truncated") {
    auto buf = make_webp_vp8_header(100, 100);
    buf.resize(20); // chop off VP8 frame data
    uint32_t w = 0, h = 0;
    auto err = check_dimensions(buf.data(), buf.size(), ImageFormat::WebP,
                                kMaxDimension, kMaxDimension, w, h);
    CHECK(err == ValidationError::DimensionReadFailed);
}

// =============================================================================
// Decompression ratio
// =============================================================================

TEST_CASE("check_decomp_ratio: bomb scenario rejected") {
    // 8192x8192 image in 1 byte: raw = 201326592, ratio > 100
    auto err = check_decomp_ratio(nullptr, 1, ImageFormat::PNG,
                                  8192, 8192, kMaxDecompRatio);
    CHECK(err == ValidationError::DecompRatioTooHigh);
}

TEST_CASE("check_decomp_ratio: normal image accepted") {
    // 512x512 RGB, 100KB compressed -> raw = 786432; ratio ~7.8x
    auto err = check_decomp_ratio(nullptr, 100 * 1024, ImageFormat::PNG,
                                  512, 512, kMaxDecompRatio);
    CHECK(err == ValidationError::Ok);
}

TEST_CASE("check_decomp_ratio: disabled when max_ratio=0") {
    auto err = check_decomp_ratio(nullptr, 1, ImageFormat::PNG,
                                  8192, 8192, 0);
    CHECK(err == ValidationError::Ok);
}

TEST_CASE("check_decomp_ratio: zero-length file rejected") {
    auto err = check_decomp_ratio(nullptr, 0, ImageFormat::PNG,
                                  100, 100, kMaxDecompRatio);
    CHECK(err == ValidationError::FileTooSmall);
}

// =============================================================================
// Full validate_input integration: valid PNG passes all checks
// =============================================================================

TEST_CASE("validate_input: valid PNG header passes") {
    // Build a PNG header with reasonable dimensions and enough "file size"
    // to not trigger the decomp ratio. We pad to simulate a reasonable file.
    auto hdr = make_png_header(512, 512);
    // raw_estimate = 512*512*3 = 786432. Need len >= 786432/100 = 7865 bytes
    std::vector<uint8_t> buf(8000, 0);
    std::memcpy(buf.data(), hdr.data(), hdr.size());
    auto r = validate_input(buf.data(), buf.size());
    CHECK(r.ok());
    CHECK(r.format == ImageFormat::PNG);
    CHECK(r.width == 512);
    CHECK(r.height == 512);
}

TEST_CASE("validate_input: valid JPEG header passes") {
    auto hdr = make_jpeg_header(640, 480);
    // raw_estimate = 640*480*3 = 921600. Need len >= 921600/100 = 9216
    std::vector<uint8_t> buf(10000, 0);
    std::memcpy(buf.data(), hdr.data(), hdr.size());
    // Fix: the JPEG parser needs valid marker structure; pad doesn't matter
    // since SOF0 was found in hdr already. Just copy the header.
    auto r = validate_input(buf.data(), buf.size());
    CHECK(r.ok());
    CHECK(r.format == ImageFormat::JPEG);
    CHECK(r.width == 640);
    CHECK(r.height == 480);
}

// =============================================================================
// Error messages
// =============================================================================

TEST_CASE("validation_error_message: all codes have non-empty strings") {
    const ValidationError codes[] = {
        ValidationError::Ok,
        ValidationError::NullInput,
        ValidationError::FileTooSmall,
        ValidationError::FileTooLarge,
        ValidationError::UnknownFormat,
        ValidationError::UnsupportedFormat,
        ValidationError::DimensionReadFailed,
        ValidationError::WidthTooLarge,
        ValidationError::HeightTooLarge,
        ValidationError::ZeroDimension,
        ValidationError::PixelCountOverflow,
        ValidationError::DecompRatioTooHigh,
    };
    for (auto code : codes) {
        const char* msg = validation_error_message(code);
        CHECK(msg != nullptr);
        CHECK(msg[0] != '\0');
    }
}
