#pragma once

#include "transformer_lab/data/tokenizer.hpp"

#include <array>
#include <cstddef>
#include <random>
#include <span>
#include <vector>

namespace transformer_lab {

class TokenBatch {
public:
    using Shape = std::array<std::size_t, 2>;

    TokenBatch(
        std::size_t batch_size,
        std::size_t context_size,
        std::vector<TokenId> inputs,
        std::vector<TokenId> targets
    );

    [[nodiscard]] Shape shape() const noexcept;
    [[nodiscard]] std::size_t batch_size() const noexcept;
    [[nodiscard]] std::size_t context_size() const noexcept;
    [[nodiscard]] std::span<const TokenId> inputs() const noexcept;
    [[nodiscard]] std::span<const TokenId> targets() const noexcept;

    [[nodiscard]] TokenId input_at(
        std::size_t row,
        std::size_t time
    ) const;
    [[nodiscard]] TokenId target_at(
        std::size_t row,
        std::size_t time
    ) const;

private:
    std::size_t batch_size_;
    std::size_t context_size_;
    std::vector<TokenId> inputs_;
    std::vector<TokenId> targets_;

    [[nodiscard]] std::size_t offset(
        std::size_t row,
        std::size_t time
    ) const;
};

// Each start creates one row. Targets contain the byte token immediately after
// each corresponding input token.
[[nodiscard]] TokenBatch make_next_token_batch(
    std::span<const TokenId> corpus_tokens,
    std::span<const std::size_t> window_starts,
    std::size_t context_size
);

// Samples batch rows uniformly with replacement from every valid next-token
// window. The caller owns the random engine so a fixed seed reproduces the
// same batch sequence for a given standard-library implementation.
[[nodiscard]] TokenBatch sample_next_token_batch(
    std::span<const TokenId> corpus_tokens,
    std::size_t batch_size,
    std::size_t context_size,
    std::mt19937& random
);

}  // namespace transformer_lab
