// arena.cpp — Arena allocator implementation.
#include "pixmask/arena.h"

namespace pixmask {

// ---------------------------------------------------------------------------
// Construction / Destruction / Move
// ---------------------------------------------------------------------------

Arena::Arena(size_t block_size)
    : block_size_(block_size)
{
    head_    = alloc_block(block_size_);
    current_ = head_;
}

Arena::~Arena() noexcept {
    free_chain(head_);
}

Arena::Arena(Arena&& other) noexcept
    : head_(std::exchange(other.head_, nullptr))
    , current_(std::exchange(other.current_, nullptr))
    , block_size_(other.block_size_)
{}

Arena& Arena::operator=(Arena&& other) noexcept {
    if (this != &other) {
        free_chain(head_);
        head_       = std::exchange(other.head_, nullptr);
        current_    = std::exchange(other.current_, nullptr);
        block_size_ = other.block_size_;
    }
    return *this;
}

// ---------------------------------------------------------------------------
// Core allocation
// ---------------------------------------------------------------------------

void* Arena::allocate(size_t bytes, size_t alignment) {
    assert((alignment & (alignment - 1)) == 0 && "alignment must be power of 2");
    assert(bytes > 0 && "zero-size allocation is not permitted");

    // Try the current block first.
    if (void* ptr = bump(current_, bytes, alignment)) {
        return ptr;
    }

    // Walk any existing overflow blocks (reachable after reset()).
    for (Block* b = current_->next; b != nullptr; b = b->next) {
        if (void* ptr = bump(b, bytes, alignment)) {
            current_ = b;
            return ptr;
        }
    }

    // All existing blocks exhausted — allocate a new one.
    // Size = max(block_size_, bytes + alignment) so the request always fits.
    size_t new_size = block_size_;
    if (bytes + alignment > new_size) {
        new_size = bytes + alignment;
    }

    Block* nb = alloc_block(new_size);

    // Append after current_ (not at end of chain) so we don't skip reusable blocks on next reset cycle.
    nb->next       = current_->next;
    current_->next = nb;
    current_       = nb;

    void* ptr = bump(nb, bytes, alignment);
    assert(ptr && "freshly allocated block must satisfy the request");
    return ptr;
}

void Arena::reset() noexcept {
    for (Block* b = head_; b != nullptr; b = b->next) {
        b->offset = 0;
    }
    current_ = head_;
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

size_t Arena::capacity_bytes() const noexcept {
    size_t total = 0;
    for (const Block* b = head_; b != nullptr; b = b->next) {
        total += b->size;
    }
    return total;
}

size_t Arena::used_bytes() const noexcept {
    size_t total = 0;
    for (const Block* b = head_; b != nullptr; b = b->next) {
        total += b->offset;
    }
    return total;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

void* Arena::bump(Block* b, size_t bytes, size_t alignment) noexcept {
    uintptr_t base    = reinterpret_cast<uintptr_t>(b->storage) + b->offset;
    uintptr_t aligned = (base + (alignment - 1)) & ~(alignment - 1);
    size_t    waste   = static_cast<size_t>(aligned - base);
    size_t    needed  = waste + bytes;

    if (b->offset + needed > b->size) {
        return nullptr;
    }

    b->offset += needed;
    return reinterpret_cast<void*>(aligned);
}

Arena::Block* Arena::alloc_block(size_t size) {
    auto* b    = new Block{};
    b->storage = static_cast<uint8_t*>(
        ::operator new(size, std::align_val_t{64})
    );
    b->size = size;
    return b;
}

void Arena::free_chain(Block* b) noexcept {
    while (b) {
        Block* next = b->next;
        ::operator delete(b->storage, std::align_val_t{64});
        delete b;
        b = next;
    }
}

} // namespace pixmask
