#include "riftco_transformer/training/batch_source.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace riftco_transformer::training {
namespace {

void validate_batch_shape(
    std::size_t batch_size,
    std::size_t context_size
) {
    if (batch_size == 0 || context_size == 0) {
        throw std::invalid_argument(
            "batch source dimensions must be greater than zero"
        );
    }
    if (batch_size >
        std::numeric_limits<std::size_t>::max() / context_size) {
        throw std::overflow_error(
            "batch source shape exceeds addressable size"
        );
    }
}

}  // namespace

RandomWindowBatchSource::RandomWindowBatchSource(
    std::vector<TokenId> tokens,
    std::size_t batch_size,
    std::size_t context_size,
    std::uint32_t random_seed
)
    : tokens_(std::move(tokens)),
      batch_size_(batch_size),
      context_size_(context_size),
      random_(random_seed) {
    validate_batch_shape(batch_size_, context_size_);
    if (tokens_.size() <= context_size_) {
        throw std::invalid_argument(
            "random window source requires at least context_size + 1 tokens"
        );
    }
}

std::size_t RandomWindowBatchSource::batch_size() const noexcept {
    return batch_size_;
}

std::size_t RandomWindowBatchSource::context_size() const noexcept {
    return context_size_;
}

TokenBatch RandomWindowBatchSource::next_batch() {
    return sample_next_token_batch(
        tokens_,
        batch_size_,
        context_size_,
        random_
    );
}

std::span<const TokenId>
RandomWindowBatchSource::tokens() const noexcept {
    return tokens_;
}

std::size_t
RandomWindowBatchSource::valid_window_count() const noexcept {
    return tokens_.size() - context_size_;
}

SequenceWindowBatchSource::SequenceWindowBatchSource(
    std::vector<std::vector<TokenId>> sequences,
    std::size_t batch_size,
    std::size_t context_size,
    std::uint32_t random_seed
)
    : sequences_(std::move(sequences)),
      cumulative_window_counts_(),
      batch_size_(batch_size),
      context_size_(context_size),
      valid_window_count_(0),
      random_(random_seed) {
    validate_batch_shape(batch_size_, context_size_);
    if (sequences_.empty()) {
        throw std::invalid_argument(
            "sequence window source requires at least one sequence"
        );
    }

    cumulative_window_counts_.reserve(sequences_.size());
    for (const auto& sequence : sequences_) {
        const std::size_t window_count =
            sequence.size() > context_size_
                ? sequence.size() - context_size_
                : 0;
        if (valid_window_count_ >
            std::numeric_limits<std::size_t>::max() - window_count) {
            throw std::overflow_error(
                "sequence window count exceeds addressable size"
            );
        }
        valid_window_count_ += window_count;
        cumulative_window_counts_.push_back(valid_window_count_);
    }
    if (valid_window_count_ == 0) {
        throw std::invalid_argument(
            "sequence window source has no complete next-token window"
        );
    }
}

std::size_t SequenceWindowBatchSource::batch_size() const noexcept {
    return batch_size_;
}

std::size_t SequenceWindowBatchSource::context_size() const noexcept {
    return context_size_;
}

TokenBatch SequenceWindowBatchSource::next_batch() {
    std::uniform_int_distribution<std::size_t> distribution(
        0,
        valid_window_count_ - 1
    );
    std::vector<TokenId> inputs;
    std::vector<TokenId> targets;
    inputs.reserve(batch_size_ * context_size_);
    targets.reserve(batch_size_ * context_size_);

    for (std::size_t row = 0; row < batch_size_; ++row) {
        const std::size_t flat_window = distribution(random_);
        const auto sequence_cursor = std::upper_bound(
            cumulative_window_counts_.begin(),
            cumulative_window_counts_.end(),
            flat_window
        );
        if (sequence_cursor == cumulative_window_counts_.end()) {
            throw std::logic_error(
                "sampled sequence window is outside the source"
            );
        }
        const std::size_t sequence_index =
            static_cast<std::size_t>(
                sequence_cursor -
                cumulative_window_counts_.begin()
            );
        const std::size_t preceding_windows =
            sequence_index == 0
                ? 0
                : cumulative_window_counts_[sequence_index - 1];
        const std::size_t start = flat_window - preceding_windows;
        const auto& sequence = sequences_[sequence_index];
        for (std::size_t time = 0;
             time < context_size_;
             ++time) {
            inputs.push_back(sequence[start + time]);
            targets.push_back(sequence[start + time + 1]);
        }
    }

    return TokenBatch(
        batch_size_,
        context_size_,
        std::move(inputs),
        std::move(targets)
    );
}

std::span<const std::vector<TokenId>>
SequenceWindowBatchSource::sequences() const noexcept {
    return sequences_;
}

std::size_t
SequenceWindowBatchSource::valid_window_count() const noexcept {
    return valid_window_count_;
}

}  // namespace riftco_transformer::training
