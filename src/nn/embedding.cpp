#include "transformer_lab/nn/embedding.hpp"

#include "transformer_lab/nn/initialization.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace transformer_lab {
namespace {

Tensor checked_embedding_weight(Tensor weight) {
    if (weight.rank() != 2) {
        throw std::invalid_argument(
            "embedding weight must have shape [vocabulary, width]"
        );
    }
    return weight;
}

Tensor initialized_embedding_weight(
    std::size_t vocabulary_size,
    std::size_t embedding_width,
    std::mt19937& random
) {
    if (vocabulary_size == 0 || embedding_width == 0) {
        throw std::invalid_argument(
            "embedding dimensions must be greater than zero"
        );
    }
    const float limit =
        1.0F / std::sqrt(static_cast<float>(embedding_width));
    return uniform_tensor(
        {vocabulary_size, embedding_width},
        -limit,
        limit,
        random
    );
}

}  // namespace

Embedding::Embedding(Tensor weight)
    : weight_(checked_embedding_weight(std::move(weight))) {
    register_parameter("weight", weight_);
}

Embedding::Embedding(
    std::size_t vocabulary_size,
    std::size_t embedding_width,
    std::mt19937& random
)
    : weight_(initialized_embedding_weight(
          vocabulary_size,
          embedding_width,
          random
      )) {
    register_parameter("weight", weight_);
}

std::size_t Embedding::vocabulary_size() const noexcept {
    return weight_.value().shape()[0];
}

std::size_t Embedding::embedding_width() const noexcept {
    return weight_.value().shape()[1];
}

Variable Embedding::forward(
    std::span<const TokenId> token_ids,
    Tensor::Shape token_shape
) const {
    std::vector<std::size_t> row_indices;
    row_indices.reserve(token_ids.size());
    for (const TokenId token : token_ids) {
        row_indices.push_back(static_cast<std::size_t>(token));
    }
    return gather_rows(
        weight_.variable(),
        row_indices,
        std::move(token_shape)
    );
}

void Embedding::to(ExecutionBackend backend) {
    Module::to(backend);
}

const Parameter& Embedding::weight() const noexcept {
    return weight_;
}

ParameterList Embedding::parameters() {
    return Module::parameters();
}

}  // namespace transformer_lab
