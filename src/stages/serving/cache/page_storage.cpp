#include "stages/serving/cache/detail/page_storage.hpp"

#include "stages/serving/cache/detail/validation.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace transformer_lab::stages::serving::cache_detail {

PageStorage::PageStorage(
    TransformerDimensions dimensions,
    ExecutionBackend backend,
    std::size_t block_size,
    std::size_t block_count,
    PageAllocation allocation
)
    : dimensions_(checked_dimensions(dimensions)),
      backend_(checked_backend(backend)),
      block_size_(checked_block_size(block_size)),
      block_count_(block_count),
      allocation_(allocation) {
    if (block_count_ == 0) {
        throw std::invalid_argument(
            "KV cache physical block count must be greater than zero"
        );
    }
    if (block_count_ - 1 >
        static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max()
        )) {
        throw std::overflow_error(
            "KV cache physical block IDs exceed uint32_t"
        );
    }

    const Tensor::Shape page_shape{
        block_count_,
        dimensions_.head_count,
        block_size_,
        head_width(),
    };
    key_pages_.reserve(dimensions_.block_count);
    value_pages_.reserve(dimensions_.block_count);
    for (std::size_t layer = 0;
         layer < dimensions_.block_count;
         ++layer) {
        key_pages_.push_back(Tensor::zeros(page_shape, backend_));
        value_pages_.push_back(Tensor::zeros(page_shape, backend_));
    }

    if (is_pooled()) {
        free_blocks_.reserve(block_count_);
        for (std::size_t index = block_count_; index > 0; --index) {
            free_blocks_.push_back(
                static_cast<std::uint32_t>(index - 1)
            );
        }
    }
}

const TransformerDimensions&
PageStorage::dimensions() const noexcept {
    return dimensions_;
}

ExecutionBackend PageStorage::backend() const noexcept {
    return backend_;
}

std::size_t PageStorage::block_size() const noexcept {
    return block_size_;
}

std::size_t PageStorage::block_count() const noexcept {
    return block_count_;
}

std::size_t PageStorage::head_width() const noexcept {
    return dimensions_.model_width / dimensions_.head_count;
}

Tensor& PageStorage::key_pages(std::size_t layer) {
    return key_pages_.at(layer);
}

Tensor& PageStorage::value_pages(std::size_t layer) {
    return value_pages_.at(layer);
}

std::uint32_t PageStorage::acquire_block() {
    if (!is_pooled()) {
        throw std::logic_error(
            "a fixed KV cache does not lease blocks"
        );
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (free_blocks_.empty()) {
        throw std::runtime_error(
            "paged KV cache pool is exhausted"
        );
    }
    const std::uint32_t result = free_blocks_.back();
    free_blocks_.pop_back();
    return result;
}

void PageStorage::release_block(std::uint32_t block) noexcept {
    if (!is_pooled()) {
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        free_blocks_.push_back(block);
    } catch (...) {
        // Destructors and cache reset are noexcept. All page IDs released
        // here were acquired from a vector with block_count capacity, so
        // this path is only defensive against an unexpected lock failure.
    }
}

std::size_t PageStorage::free_block_count() const noexcept {
    if (!is_pooled()) {
        return 0;
    }
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        return free_blocks_.size();
    } catch (...) {
        return 0;
    }
}

bool PageStorage::is_pooled() const noexcept {
    return allocation_ == PageAllocation::Pooled;
}

}  // namespace transformer_lab::stages::serving::cache_detail
