#pragma once

#include <cstdint>

#include "pixmask/types.h"

namespace pixmask {

// Reduce bit depth in-place. bits must be 1-8.
// Formula: pixel = (pixel >> (8 - bits)) << (8 - bits)
//        = pixel & (0xFF << (8 - bits))
void reduce_bit_depth(ImageView& image, uint8_t bits);

} // namespace pixmask
