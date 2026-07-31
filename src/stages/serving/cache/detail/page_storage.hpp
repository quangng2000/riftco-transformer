#pragma once

#include "riftco_transformer/core/backend.hpp"
#include "riftco_transformer/core/tensor.hpp"
#include "riftco_transformer/model/decoder_only_transformer.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace riftco_transformer::stages::serving::cache_detail {

enum class PageAllocation {
    Fixed,
    Pooled,
};

class PageStorage final {
public:
    PageStorage(
        TransformerDimensions dimensions,
        ExecutionBackend backend,
        std::size_t block_size,
        std::size_t block_count,
        PageAllocation allocation
    );

    [[nodiscard]] const TransformerDimensions&
    dimensions() const noexcept;
    [[nodiscard]] ExecutionBackend backend() const noexcept;
    [[nodiscard]] std::size_t block_size() const noexcept;
    [[nodiscard]] std::size_t block_count() const noexcept;
    [[nodiscard]] std::size_t head_width() const noexcept;

    [[nodiscard]] Tensor& key_pages(std::size_t layer);
    [[nodiscard]] Tensor& value_pages(std::size_t layer);

    [[nodiscard]] std::uint32_t acquire_block();
    void release_block(std::uint32_t block) noexcept;
    [[nodiscard]] std::size_t free_block_count() const noexcept;

private:
    [[nodiscard]] bool is_pooled() const noexcept;

    TransformerDimensions dimensions_;
    ExecutionBackend backend_;
    std::size_t block_size_;
    std::size_t block_count_;
    PageAllocation allocation_;
    std::vector<Tensor> key_pages_;
    std::vector<Tensor> value_pages_;
    mutable std::mutex mutex_;
    std::vector<std::uint32_t> free_blocks_;
};

}  // namespace riftco_transformer::stages::serving::cache_detail
