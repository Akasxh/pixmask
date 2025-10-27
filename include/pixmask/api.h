#pragma once

#include <string>

#include "pixmask/image.h"

namespace pixmask {

//! Initialize any global state required by the pixmask runtime.
void initialize();

//! Retrieve a semantic version string for the library.
std::string version_string();

//! Run the fixed-weight SR-lite refinement stage.
bool sr_lite_refine(const CpuImage &input, const CpuImage &output);

//! Execute the full sanitize pipeline on the provided image buffers.
bool sanitize(const CpuImage &input, const CpuImage &output);

} // namespace pixmask
