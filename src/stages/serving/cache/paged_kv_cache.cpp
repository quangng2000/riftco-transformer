#include "riftco_transformer/stages/serving/kv_cache.hpp"

#include "stages/serving/cache/detail/page_storage.hpp"
#include "stages/serving/cache/detail/page_table_cache.hpp"
#include "stages/serving/cache/detail/validation.hpp"

#include <cstddef>
#include <memory>

namespace riftco_transformer::stages::serving {

struct PagedKvCachePool::SharedState {
    std::shared_ptr<cache_detail::PageStorage> storage;
    std::size_t blocks_per_cache;
};

PagedKvCachePool::PagedKvCachePool(
    TransformerDimensions dimensions,
    ExecutionBackend backend,
    std::size_t block_size,
    std::size_t block_count
)
    : state_(nullptr) {
    dimensions = cache_detail::checked_dimensions(dimensions);
    backend = cache_detail::checked_backend(backend);
    block_size = cache_detail::checked_block_size(block_size);
    const std::size_t blocks_per_cache =
        cache_detail::divide_rounding_up(
            dimensions.maximum_context,
            block_size
        );
    block_count = cache_detail::checked_block_count(
        block_count,
        blocks_per_cache
    );
    state_ = std::make_shared<SharedState>(
        SharedState{
            std::make_shared<cache_detail::PageStorage>(
                dimensions,
                backend,
                block_size,
                block_count,
                cache_detail::PageAllocation::Pooled
            ),
            blocks_per_cache,
        }
    );
}

PagedKvCachePool::~PagedKvCachePool() = default;

const TransformerDimensions&
PagedKvCachePool::dimensions() const noexcept {
    return state_->storage->dimensions();
}

ExecutionBackend PagedKvCachePool::backend() const noexcept {
    return state_->storage->backend();
}

std::unique_ptr<DecoderKeyValueCache>
PagedKvCachePool::create() const {
    return cache_detail::make_leased_page_table_cache(
        state_->storage
    );
}

std::size_t PagedKvCachePool::block_size() const noexcept {
    return state_->storage->block_size();
}

std::size_t PagedKvCachePool::block_count() const noexcept {
    return state_->storage->block_count();
}

std::size_t
PagedKvCachePool::blocks_per_full_cache() const noexcept {
    return state_->blocks_per_cache;
}

std::size_t
PagedKvCachePool::free_block_count() const noexcept {
    return state_->storage->free_block_count();
}

std::size_t
PagedKvCachePool::leased_block_count() const noexcept {
    return block_count() - free_block_count();
}

}  // namespace riftco_transformer::stages::serving
