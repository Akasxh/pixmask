// pixmask/arena.h — Bump-pointer arena allocator for the sanitization pipeline.
// See architecture/CPP_ARENA_PIPELINE_REFERENCE.md for design rationale.
//
// Thread safety: NONE. One Arena per Pipeline; Pipelines are not shared across threads.
#pragma once
#ifndef PIXMASK_ARENA_H
#define PIXMASK_ARENA_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <new>       // std::bad_alloc, std::align_val_t
#include <utility>   // std::exchange

namespace pixmask {

class Arena {
public:
    static constexpr size_t kDefaultBlockSize = 32ULL << 20;  // 32 MB

    // Construct an arena with the given initial block size.
    // Immediately allocates one block of `block_size` bytes.
    // Throws std::bad_alloc if the system is out of memory.
    explicit Arena(size_t block_size = kDefaultBlockSize);

    ~Arena() noexcept;

    // Non-copyable.
    Arena(const Arena&)            = delete;
    Arena& operator=(const Arena&) = delete;

    // Move-only.
    Arena(Arena&& other) noexcept;
    Arena& operator=(Arena&& other) noexcept;

    // Allocate `bytes` with the given alignment (must be a power of 2).
    // Returns a non-null pointer. Throws std::bad_alloc on system OOM.
    // Default alignment = 64 (cache line + AVX-512 width).
    [[nodiscard]] void* allocate(size_t bytes, size_t alignment = 64);

    // Typed array allocation for trivially constructible types.
    template <typename T>
    [[nodiscard]] T* allocate_array(size_t count) {
        static_assert(std::is_trivially_constructible_v<T>,
                      "Arena only supports trivially constructible types");
        return static_cast<T*>(allocate(sizeof(T) * count, alignof(T) < 64 ? 64 : alignof(T)));
    }

    // Rewind all block offsets to zero. Does NOT free memory.
    // All previously returned pointers are invalidated after reset().
    void reset() noexcept;

    // Total bytes allocated across all blocks (capacity, not bytes in use).
    [[nodiscard]] size_t capacity_bytes() const noexcept;

    // Bytes currently in use across all blocks (sum of offsets).
    [[nodiscard]] size_t used_bytes() const noexcept;

private:
    struct Block {
        uint8_t* storage = nullptr;
        size_t   size    = 0;
        size_t   offset  = 0;
        Block*   next    = nullptr;
    };

    // Try to bump-allocate `bytes` with `alignment` from block `b`.
    // Returns aligned pointer on success, nullptr if insufficient space.
    static void* bump(Block* b, size_t bytes, size_t alignment) noexcept;

    // Allocate a new Block with `size` bytes of 64-byte-aligned storage.
    static Block* alloc_block(size_t size);

    // Free all blocks starting from `b`.
    static void free_chain(Block* b) noexcept;

    Block* head_       = nullptr;
    Block* current_    = nullptr;
    size_t block_size_ = kDefaultBlockSize;
};

} // namespace pixmask

#endif // PIXMASK_ARENA_H
