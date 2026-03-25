// pixmask/median.h — 3x3 median filter (Stage 4) using Bose-Nelson sorting network.
// See architecture/CPP_MEDIAN_REFERENCE.md for design rationale.
#pragma once

#include "pixmask/arena.h"
#include "pixmask/types.h"

namespace pixmask {

// Apply 3x3 median filter. Allocates output from arena.
// Border pixels use replicate padding (clamp-to-edge).
// The filter operates per-channel: for multi-channel images,
// each channel is filtered independently.
// Returns a new ImageView with stride aligned to 64 bytes.
ImageView median_filter_3x3(const ImageView& input, Arena& arena);

} // namespace pixmask
