#pragma once

#include "riftco_transformer/core/backend.hpp"
#include "riftco_transformer/model/decoder_kv_cache.hpp"
#include "riftco_transformer/model/decoder_only_transformer.hpp"

#include <cstddef>
#include <memory>

namespace riftco_transformer::stages::serving {

// Creates one request-local decoder cache. A factory may either allocate
// independent storage per request or lease pages from shared storage.
class KeyValueCacheFactory {
public:
    KeyValueCacheFactory() = default;
    KeyValueCacheFactory(const KeyValueCacheFactory&) = delete;
    KeyValueCacheFactory& operator=(const KeyValueCacheFactory&) = delete;
    KeyValueCacheFactory(KeyValueCacheFactory&&) = delete;
    KeyValueCacheFactory& operator=(KeyValueCacheFactory&&) = delete;
    virtual ~KeyValueCacheFactory() = default;

    [[nodiscard]] virtual const TransformerDimensions&
    dimensions() const noexcept = 0;
    [[nodiscard]] virtual ExecutionBackend backend() const noexcept = 0;
    [[nodiscard]] virtual std::unique_ptr<DecoderKeyValueCache>
    create() const = 0;
};

// Gives every request one fixed-capacity page per transformer layer. This is
// the simple reference layout; it still uses the same paged-decode-attention
// backend request as the shared pool.
class ContiguousKvCacheFactory final : public KeyValueCacheFactory {
public:
    ContiguousKvCacheFactory(
        TransformerDimensions dimensions,
        ExecutionBackend backend
    );
    ~ContiguousKvCacheFactory() override;

    [[nodiscard]] const TransformerDimensions&
    dimensions() const noexcept override;
    [[nodiscard]] ExecutionBackend backend() const noexcept override;
    [[nodiscard]] std::unique_ptr<DecoderKeyValueCache>
    create() const override;

private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

// Owns backend-resident K/V pages shared by all caches created from this pool.
// Page IDs are leased lazily and returned when a request resets or is
// destroyed. One physical ID addresses the matching page in every layer.
class PagedKvCachePool final : public KeyValueCacheFactory {
public:
    PagedKvCachePool(
        TransformerDimensions dimensions,
        ExecutionBackend backend,
        std::size_t block_size,
        std::size_t block_count = 0
    );
    ~PagedKvCachePool() override;

    [[nodiscard]] const TransformerDimensions&
    dimensions() const noexcept override;
    [[nodiscard]] ExecutionBackend backend() const noexcept override;
    [[nodiscard]] std::unique_ptr<DecoderKeyValueCache>
    create() const override;

    [[nodiscard]] std::size_t block_size() const noexcept;
    [[nodiscard]] std::size_t block_count() const noexcept;
    [[nodiscard]] std::size_t blocks_per_full_cache() const noexcept;
    [[nodiscard]] std::size_t free_block_count() const noexcept;
    [[nodiscard]] std::size_t leased_block_count() const noexcept;

private:
    struct SharedState;
    std::shared_ptr<SharedState> state_;
};

}  // namespace riftco_transformer::stages::serving
