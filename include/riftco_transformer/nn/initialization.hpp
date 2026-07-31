#pragma once

#include "riftco_transformer/core/tensor.hpp"

#include <cstddef>
#include <random>

namespace riftco_transformer {

[[nodiscard]] Tensor uniform_tensor(
    Tensor::Shape shape,
    float lower,
    float upper,
    std::mt19937& random
);
[[nodiscard]] Tensor uniform_tensor(
    Tensor::Shape shape,
    float lower,
    float upper,
    std::mt19937& random,
    ExecutionBackend backend
);

[[nodiscard]] Tensor xavier_uniform(
    Tensor::Shape shape,
    std::size_t fan_in,
    std::size_t fan_out,
    std::mt19937& random
);
[[nodiscard]] Tensor xavier_uniform(
    Tensor::Shape shape,
    std::size_t fan_in,
    std::size_t fan_out,
    std::mt19937& random,
    ExecutionBackend backend
);

}  // namespace riftco_transformer
