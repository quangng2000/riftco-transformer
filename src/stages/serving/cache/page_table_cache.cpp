#include "stages/serving/cache/detail/page_table_cache.hpp"

#include "core/backend/attention/dispatch.hpp"
#include "core/backend/storage.hpp"
#include "stages/serving/cache/detail/page_storage.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace riftco_transformer::stages::serving::cache_detail {
namespace {

void require_token_tensor(
    const Tensor& tensor,
    const PageStorage& storage,
    const char* name
) {
    const Tensor::Shape expected{
        1,
        storage.dimensions().head_count,
        1,
        storage.head_width(),
    };
    if (tensor.shape() != expected) {
        throw std::invalid_argument(
            std::string("KV cache ") + name +
            " must have shape [1, heads, 1, head_width]"
        );
    }
    if (tensor.backend() != storage.backend()) {
        throw std::invalid_argument(
            std::string("KV cache ") + name +
            " backend does not match the cache"
        );
    }
}

void copy_token_to_page(
    const Tensor& source,
    Tensor& pages,
    const PageStorage& storage,
    std::uint32_t physical_block,
    std::size_t block_offset
) {
    const std::size_t head_count =
        storage.dimensions().head_count;
    const std::size_t head_width = storage.head_width();
    const auto input = source.data();
    auto output = pages.data();
    for (std::size_t head = 0; head < head_count; ++head) {
        const std::size_t source_offset = head * head_width;
        const std::size_t destination_offset =
            (
                (
                    static_cast<std::size_t>(physical_block) *
                        head_count +
                    head
                ) *
                    storage.block_size() +
                block_offset
            ) *
            head_width;
        std::copy_n(
            input.begin() +
                static_cast<std::ptrdiff_t>(source_offset),
            head_width,
            output.begin() +
                static_cast<std::ptrdiff_t>(destination_offset)
        );
    }
}

class PageTableKeyValueCache final : public DecoderKeyValueCache {
public:
    PageTableKeyValueCache(
        std::shared_ptr<PageStorage> storage,
        bool fixed_page
    )
        : storage_(std::move(storage)),
          fixed_page_(fixed_page),
          layer_written_(
              storage_->dimensions().block_count,
              false
          ) {
        if (fixed_page_) {
            if (storage_->block_count() != 1 ||
                storage_->block_size() != capacity()) {
                throw std::logic_error(
                    "contiguous KV cache storage is malformed"
                );
            }
            page_table_.push_back(0);
        }
    }

    ~PageTableKeyValueCache() override {
        reset();
    }

    [[nodiscard]] ExecutionBackend backend() const noexcept override {
        return storage_->backend();
    }

    [[nodiscard]] std::size_t layer_count() const noexcept override {
        return storage_->dimensions().block_count;
    }

    [[nodiscard]] std::size_t head_count() const noexcept override {
        return storage_->dimensions().head_count;
    }

    [[nodiscard]] std::size_t head_width() const noexcept override {
        return storage_->head_width();
    }

    [[nodiscard]] std::size_t capacity() const noexcept override {
        return storage_->dimensions().maximum_context;
    }

    [[nodiscard]] std::size_t size() const noexcept override {
        return size_;
    }

    void begin_token() override {
        if (token_active_) {
            throw std::logic_error(
                "KV cache already has an active token transaction"
            );
        }
        if (size_ == capacity()) {
            throw std::length_error("KV cache capacity is exhausted");
        }

        const std::size_t logical_block =
            size_ / storage_->block_size();
        if (!fixed_page_ && logical_block == page_table_.size()) {
            const std::uint32_t block = storage_->acquire_block();
            try {
                page_table_.push_back(block);
            } catch (...) {
                storage_->release_block(block);
                throw;
            }
            pending_page_ = true;
        } else if (logical_block >= page_table_.size()) {
            throw std::logic_error(
                "KV cache page table does not cover the next token"
            );
        }

        std::fill(
            layer_written_.begin(),
            layer_written_.end(),
            false
        );
        token_active_ = true;
    }

    [[nodiscard]] Tensor append_and_attend(
        std::size_t layer,
        const Tensor& query,
        const Tensor& key,
        const Tensor& value
    ) override {
        if (!token_active_) {
            throw std::logic_error(
                "KV cache append requires an active token transaction"
            );
        }
        if (layer >= layer_count()) {
            throw std::out_of_range(
                "KV cache layer is outside the model"
            );
        }
        if (layer_written_[layer]) {
            throw std::logic_error(
                "KV cache layer was appended more than once"
            );
        }
        require_token_tensor(query, *storage_, "query");
        require_token_tensor(key, *storage_, "key");
        require_token_tensor(value, *storage_, "value");

        const std::size_t logical_block =
            size_ / storage_->block_size();
        const std::size_t block_offset =
            size_ % storage_->block_size();
        const std::uint32_t physical_block =
            page_table_.at(logical_block);
        Tensor& key_pages = storage_->key_pages(layer);
        Tensor& value_pages = storage_->value_pages(layer);
        copy_token_to_page(
            key,
            key_pages,
            *storage_,
            physical_block,
            block_offset
        );
        copy_token_to_page(
            value,
            value_pages,
            *storage_,
            physical_block,
            block_offset
        );

        Tensor context(query.shape(), backend());
        backend_detail::dispatch_paged_decode_attention_forward(
            backend(),
            {
                backend_detail::tensor_storage(query),
                backend_detail::tensor_storage(key_pages),
                backend_detail::tensor_storage(value_pages),
                page_table_,
                backend_detail::tensor_storage(context),
                {
                    head_count(),
                    head_width(),
                    storage_->block_size(),
                    storage_->block_count(),
                    size_ + 1,
                },
            }
        );
        layer_written_[layer] = true;
        return context;
    }

    void commit_token() override {
        if (!token_active_) {
            throw std::logic_error(
                "KV cache commit requires an active token transaction"
            );
        }
        if (!std::all_of(
                layer_written_.begin(),
                layer_written_.end(),
                [](bool written) { return written; }
            )) {
            throw std::logic_error(
                "KV cache commit requires every layer exactly once"
            );
        }
        ++size_;
        token_active_ = false;
        pending_page_ = false;
    }

    void abort_token() noexcept override {
        if (!token_active_) {
            return;
        }
        if (pending_page_) {
            const std::uint32_t block = page_table_.back();
            page_table_.pop_back();
            storage_->release_block(block);
        }
        std::fill(
            layer_written_.begin(),
            layer_written_.end(),
            false
        );
        token_active_ = false;
        pending_page_ = false;
    }

    void reset() noexcept override {
        abort_token();
        if (!fixed_page_) {
            for (const std::uint32_t block : page_table_) {
                storage_->release_block(block);
            }
            page_table_.clear();
        }
        size_ = 0;
    }

private:
    std::shared_ptr<PageStorage> storage_;
    bool fixed_page_;
    std::vector<std::uint32_t> page_table_;
    std::vector<bool> layer_written_;
    std::size_t size_ = 0;
    bool token_active_ = false;
    bool pending_page_ = false;
};

}  // namespace

std::unique_ptr<DecoderKeyValueCache>
make_fixed_page_table_cache(std::shared_ptr<PageStorage> storage) {
    return std::make_unique<PageTableKeyValueCache>(
        std::move(storage),
        true
    );
}

std::unique_ptr<DecoderKeyValueCache>
make_leased_page_table_cache(std::shared_ptr<PageStorage> storage) {
    return std::make_unique<PageTableKeyValueCache>(
        std::move(storage),
        false
    );
}

}  // namespace riftco_transformer::stages::serving::cache_detail
