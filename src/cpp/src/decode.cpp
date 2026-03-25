// Stage 1: Image decode via stb_image.
// STB_IMAGE_IMPLEMENTATION is defined here and ONLY here.

// Format restrictions (STBI_ONLY_JPEG, STBI_ONLY_PNG, STBI_NO_*) are set
// via target_compile_definitions in CMakeLists.txt. Do NOT redefine here.

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "pixmask/decode.h"

#include <cstring>
#include <new>  // std::bad_alloc

namespace pixmask {

static constexpr uint32_t kChannelsRGB = 3;

ImageView decode_image(
    const uint8_t* data,
    size_t         len,
    Arena&         arena,
    const char**   error_out
) noexcept {
    auto fail = [&](const char* msg) -> ImageView {
        if (error_out) *error_out = msg;
        return ImageView{};
    };

    if (data == nullptr || len == 0) {
        return fail("decode_image: null or empty input");
    }

    if (len > static_cast<size_t>(INT32_MAX)) {
        return fail("decode_image: input too large for stb_image (>2GB)");
    }

    int w = 0, h = 0, channels_in_file = 0;
    stbi_uc* pixels = stbi_load_from_memory(
        data,
        static_cast<int>(len),
        &w, &h, &channels_in_file,
        static_cast<int>(kChannelsRGB)  // force RGB output
    );

    if (pixels == nullptr) {
        const char* reason = stbi_failure_reason();
        return fail(reason ? reason : "decode_image: unknown stb_image error");
    }

    if (w <= 0 || h <= 0) {
        stbi_image_free(pixels);
        return fail("decode_image: decoded zero or negative dimensions");
    }

    const auto width  = static_cast<uint32_t>(w);
    const auto height = static_cast<uint32_t>(h);
    const uint32_t stride = aligned_stride(width, kChannelsRGB);
    const size_t buf_size = static_cast<size_t>(stride) * height;

    // Arena::allocate() throws std::bad_alloc on OOM.
    // Catch it to maintain noexcept contract and free stb's buffer.
    uint8_t* out = nullptr;
    try {
        out = static_cast<uint8_t*>(arena.allocate(buf_size));
    } catch (const std::bad_alloc&) {
        stbi_image_free(pixels);
        return fail("decode_image: arena allocation failed");
    }

    // Copy row-by-row: stb output is tightly packed (width*3 per row),
    // but our arena buffer uses aligned stride.
    const uint32_t src_stride = width * kChannelsRGB;
    if (stride == src_stride) {
        std::memcpy(out, pixels, static_cast<size_t>(src_stride) * height);
    } else {
        for (uint32_t row = 0; row < height; ++row) {
            std::memcpy(
                out + static_cast<size_t>(row) * stride,
                pixels + static_cast<size_t>(row) * src_stride,
                src_stride
            );
        }
    }

    stbi_image_free(pixels);

    if (error_out) *error_out = nullptr;

    return ImageView{
        .data     = out,
        .width    = width,
        .height   = height,
        .channels = kChannelsRGB,
        .stride   = stride,
    };
}

} // namespace pixmask
