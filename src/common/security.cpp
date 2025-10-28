#include "pixmask/security.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string_view>

namespace pixmask {

namespace {

constexpr std::size_t kMegapixel = 1'000'000u;

bool contains_signature(const std::uint8_t *data, std::size_t size, std::string_view needle) {
    if (needle.empty() || data == nullptr || size < needle.size()) {
        return false;
    }
    const auto *begin = reinterpret_cast<const char *>(data);
    const auto *end = begin + size;
    return std::search(begin, end, needle.begin(), needle.end()) != end;
}

} // namespace

bool exceeds_pixel_cap(std::size_t width, std::size_t height, double cap_megapixels) {
    if (width == 0 || height == 0) {
        return false;
    }

    if (!std::isfinite(cap_megapixels)) {
        return cap_megapixels < 0.0;
    }

    if (cap_megapixels <= 0.0) {
        return true;
    }

    const long double limit_pixels = static_cast<long double>(cap_megapixels) *
                                     static_cast<long double>(kMegapixel);
    if (limit_pixels <= 0.0L) {
        return true;
    }

    const long double total_pixels = static_cast<long double>(width) *
                                     static_cast<long double>(height);
    return total_pixels > limit_pixels;
}

bool suspicious_polyglot_bytes(const std::uint8_t *data, std::size_t size) {
    if (data == nullptr || size == 0) {
        return false;
    }

    constexpr std::array<std::string_view, 8> signatures = {
        std::string_view{"%PDF-"},
        std::string_view{"PK\x03\x04"},
        std::string_view{"7zXZ"},
        std::string_view{"Rar!"},
        std::string_view{"<?xml"},
        std::string_view{"<!DOCTYPE"},
        std::string_view{"MZ"},
        std::string_view{"ELF"},
    };

    for (const auto &sig : signatures) {
        if (contains_signature(data, size, sig)) {
            return true;
        }
    }

    return false;
}

} // namespace pixmask

