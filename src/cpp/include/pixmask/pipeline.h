// pixmask/pipeline.h — Orchestrator that chains all sanitization stages.
// See architecture/DECISIONS.md section 3 for pipeline spec.
#pragma once

#include <cstddef>
#include <cstdint>

#include "pixmask/types.h"
#include "pixmask/arena.h"

namespace pixmask {

class Pipeline {
public:
    explicit Pipeline(const SanitizeOptions& opts = {});
    ~Pipeline();

    Pipeline(Pipeline&&) noexcept;
    Pipeline& operator=(Pipeline&&) noexcept;
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    // Main entry point: sanitize raw image bytes.
    // Thread safety: NOT thread-safe. One Pipeline per thread.
    // All previously returned SanitizeResult.image pointers are
    // invalidated when sanitize() is called again (arena reset).
    SanitizeResult sanitize(const uint8_t* data, size_t len);

    // Expose arena for diagnostics / testing.
    [[nodiscard]] const Arena& arena() const noexcept { return arena_; }

private:
    SanitizeOptions opts_;
    Arena arena_;
};

// Convenience free function (creates a temporary Pipeline on the stack).
SanitizeResult sanitize(const uint8_t* data, size_t len,
                        const SanitizeOptions& opts = {});

} // namespace pixmask
