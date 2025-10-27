#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace pixmask {

/// Configure the number of worker threads used by the internal pool.
void set_threads(std::size_t worker_count);

/// Retrieve the number of worker threads currently active in the pool.
std::size_t thread_count() noexcept;

/// Low-level helper invoked by the templated \c parallel_for implementation.
void parallel_for_impl(
    std::size_t begin,
    std::size_t end,
    std::function<void(std::size_t, std::size_t)> chunk_fn);

/// Execute \p fn for every index in the half-open range [begin, end).
///
/// The range is internally chunked and scheduled over the persistent thread
/// pool. When only a single worker is active, the loop executes on the calling
/// thread without spawning additional tasks.
///
/// \tparam Fn Callable with signature compatible with void(std::size_t).
template <typename Fn>
void parallel_for(std::size_t begin, std::size_t end, Fn&& fn) {
    if (end <= begin) {
        return;
    }

    using FnType = std::decay_t<Fn>;
    auto callable = std::make_shared<FnType>(std::forward<Fn>(fn));

    parallel_for_impl(begin, end, [callable](std::size_t chunk_begin, std::size_t chunk_end) {
        for (std::size_t idx = chunk_begin; idx < chunk_end; ++idx) {
            (*callable)(idx);
        }
    });
}

} // namespace pixmask

extern "C" {

void pixmask_set_threads(std::size_t worker_count);

void pixmask_parallel_for(std::size_t begin,
                          std::size_t end,
                          void (*fn)(std::size_t, void*),
                          void* user_data);

std::size_t pixmask_thread_count();

}
