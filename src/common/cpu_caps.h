#pragma once

#include <cstddef>

namespace pixmask {

/// Check whether AVX2 instructions are available on the current host.
bool has_avx2() noexcept;

/// Check whether NEON instructions are available on the current host.
bool has_neon() noexcept;

/// Return the number of hardware threads reported by the platform.
std::size_t hw_threads() noexcept;

} // namespace pixmask
