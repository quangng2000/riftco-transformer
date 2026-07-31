#pragma once

#include "riftco_transformer/data/tokenizer.hpp"
#include "riftco_transformer/nn/module.hpp"

#include <cstddef>
#include <random>
#include <span>

namespace riftco_transformer {

class Embedding : public Module {
public:
    explicit Embedding(Tensor weight);
    Embedding(
        std::size_t vocabulary_size,
        std::size_t embedding_width,
        std::mt19937& random
    );

    [[nodiscard]] std::size_t vocabulary_size() const noexcept;
    [[nodiscard]] std::size_t embedding_width() const noexcept;

    [[nodiscard]] Variable forward(
        std::span<const TokenId> token_ids,
        Tensor::Shape token_shape
    ) const;
    // Transfers parameters in place. Call before building a forward graph.
    void to(ExecutionBackend backend);

    [[nodiscard]] const Parameter& weight() const noexcept;
    [[nodiscard]] ParameterList parameters();

private:
    Parameter weight_;
};

}  // namespace riftco_transformer
