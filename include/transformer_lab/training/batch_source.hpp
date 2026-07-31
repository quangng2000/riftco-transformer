#pragma once

#include "transformer_lab/data/token_batch.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace transformer_lab::training {

class BatchSource {
public:
    BatchSource() = default;
    BatchSource(const BatchSource&) = delete;
    BatchSource& operator=(const BatchSource&) = delete;
    BatchSource(BatchSource&&) = delete;
    BatchSource& operator=(BatchSource&&) = delete;
    virtual ~BatchSource() = default;

    [[nodiscard]] virtual std::size_t batch_size() const noexcept = 0;
    [[nodiscard]] virtual std::size_t context_size() const noexcept = 0;
    [[nodiscard]] virtual TokenBatch next_batch() = 0;
};

// Uniform sampling with replacement over all next-token windows in one
// owned token sequence.
class RandomWindowBatchSource final : public BatchSource {
public:
    RandomWindowBatchSource(
        std::vector<TokenId> tokens,
        std::size_t batch_size,
        std::size_t context_size,
        std::uint32_t random_seed
    );

    [[nodiscard]] std::size_t batch_size() const noexcept override;
    [[nodiscard]] std::size_t context_size() const noexcept override;
    [[nodiscard]] TokenBatch next_batch() override;

    [[nodiscard]] std::span<const TokenId> tokens() const noexcept;
    [[nodiscard]] std::size_t valid_window_count() const noexcept;

private:
    std::vector<TokenId> tokens_;
    std::size_t batch_size_;
    std::size_t context_size_;
    std::mt19937 random_;
};

// Uniform sampling with replacement over the union of valid windows in
// multiple owned sequences. No input/target row crosses a sequence boundary.
// Sequences too short to provide a complete window are retained but skipped;
// at least one sequence must provide a window.
class SequenceWindowBatchSource final : public BatchSource {
public:
    SequenceWindowBatchSource(
        std::vector<std::vector<TokenId>> sequences,
        std::size_t batch_size,
        std::size_t context_size,
        std::uint32_t random_seed
    );

    [[nodiscard]] std::size_t batch_size() const noexcept override;
    [[nodiscard]] std::size_t context_size() const noexcept override;
    [[nodiscard]] TokenBatch next_batch() override;

    [[nodiscard]] std::span<const std::vector<TokenId>>
    sequences() const noexcept;
    [[nodiscard]] std::size_t valid_window_count() const noexcept;

private:
    std::vector<std::vector<TokenId>> sequences_;
    std::vector<std::size_t> cumulative_window_counts_;
    std::size_t batch_size_;
    std::size_t context_size_;
    std::size_t valid_window_count_;
    std::mt19937 random_;
};

}  // namespace transformer_lab::training
