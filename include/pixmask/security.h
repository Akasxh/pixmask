#pragma once

#include <cstddef>
#include <cstdint>

namespace pixmask {

//! Return true if the image dimensions exceed the configured megapixel cap.
bool exceeds_pixel_cap(std::size_t width, std::size_t height, double cap_megapixels);

//! Detect common polyglot signatures within an arbitrary byte buffer.
bool suspicious_polyglot_bytes(const std::uint8_t *data, std::size_t size);

} // namespace pixmask

