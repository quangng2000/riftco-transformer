#include "transformer_lab/data/token_batch.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace transformer_lab {
namespace {

std::size_t checked_cell_count(
    std::size_t batch_size,
    std::size_t context_size
) {
    if (batch_size == 0 || context_size == 0) {
        throw std::invalid_argument(
            "token batch dimensions must be greater than zero"
        );
    }
    if (batch_size >
        std::numeric_limits<std::size_t>::max() / context_size) {
        throw std::overflow_error("token batch shape exceeds addressable size");
    }
    return batch_size * context_size;
}

}  // namespace

TokenBatch::TokenBatch(
    std::size_t batch_size,
    std::size_t context_size,
    std::vector<TokenId> inputs,
    std::vector<TokenId> targets
)
    : batch_size_(batch_size),
      context_size_(context_size),
      inputs_(std::move(inputs)),
      targets_(std::move(targets)) {
    const auto expected_size = checked_cell_count(
        batch_size_,
        context_size_
    );
    if (inputs_.size() != expected_size ||
        targets_.size() != expected_size) {
        throw std::invalid_argument(
            "token batch data count does not match its shape"
        );
    }
}

TokenBatch::Shape TokenBatch::shape() const noexcept {
    return {batch_size_, context_size_};
}

std::size_t TokenBatch::batch_size() const noexcept {
    return batch_size_;
}

std::size_t TokenBatch::context_size() const noexcept {
    return context_size_;
}

std::span<const TokenId> TokenBatch::inputs() const noexcept {
    return inputs_;
}

std::span<const TokenId> TokenBatch::targets() const noexcept {
    return targets_;
}

TokenId TokenBatch::input_at(
    std::size_t row,
    std::size_t time
) const {
    return inputs_[offset(row, time)];
}

TokenId TokenBatch::target_at(
    std::size_t row,
    std::size_t time
) const {
    return targets_[offset(row, time)];
}

std::size_t TokenBatch::offset(
    std::size_t row,
    std::size_t time
) const {
    if (row >= batch_size_ || time >= context_size_) {
        throw std::out_of_range(
            "token batch index is outside its shape"
        );
    }
    return row * context_size_ + time;
}

TokenBatch make_next_token_batch(
    std::span<const TokenId> corpus_tokens,
    std::span<const std::size_t> window_starts,
    std::size_t context_size
) {
    const auto cell_count = checked_cell_count(
        window_starts.size(),
        context_size
    );

    for (const std::size_t start : window_starts) {
        if (start > corpus_tokens.size() ||
            corpus_tokens.size() - start <= context_size) {
            throw std::out_of_range(
                "next-token window extends beyond the corpus"
            );
        }
    }

    std::vector<TokenId> inputs;
    std::vector<TokenId> targets;
    inputs.reserve(cell_count);
    targets.reserve(cell_count);

    for (const std::size_t start : window_starts) {
        for (std::size_t time = 0; time < context_size; ++time) {
            inputs.push_back(corpus_tokens[start + time]);
            targets.push_back(corpus_tokens[start + time + 1]);
        }
    }

    return TokenBatch(
        window_starts.size(),
        context_size,
        std::move(inputs),
        std::move(targets)
    );
}

TokenBatch sample_next_token_batch(
    std::span<const TokenId> corpus_tokens,
    std::size_t batch_size,
    std::size_t context_size,
    std::mt19937& random
) {
    static_cast<void>(checked_cell_count(batch_size, context_size));
    if (corpus_tokens.size() <= context_size) {
        throw std::invalid_argument(
            "next-token sampling requires at least one complete window"
        );
    }

    const std::size_t largest_start =
        corpus_tokens.size() - context_size - 1;
    std::uniform_int_distribution<std::size_t> distribution(
        0,
        largest_start
    );
    std::vector<std::size_t> window_starts(batch_size);
    for (std::size_t& start : window_starts) {
        start = distribution(random);
    }
    return make_next_token_batch(
        corpus_tokens,
        window_starts,
        context_size
    );
}

}  // namespace transformer_lab
