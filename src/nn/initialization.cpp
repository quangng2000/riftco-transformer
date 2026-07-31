#include "riftco_transformer/nn/initialization.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace riftco_transformer {

Tensor uniform_tensor(
    Tensor::Shape shape,
    float lower,
    float upper,
    std::mt19937& random
) {
    return uniform_tensor(
        std::move(shape),
        lower,
        upper,
        random,
        execution_backend()
    );
}

Tensor uniform_tensor(
    Tensor::Shape shape,
    float lower,
    float upper,
    std::mt19937& random,
    ExecutionBackend backend
) {
    if (!std::isfinite(lower) ||
        !std::isfinite(upper) ||
        lower >= upper) {
        throw std::invalid_argument(
            "uniform initialization requires finite ordered bounds"
        );
    }

    Tensor result(std::move(shape), backend);
    std::uniform_real_distribution<float> distribution(lower, upper);
    for (float& value : result.data()) {
        value = distribution(random);
    }
    return result;
}

Tensor xavier_uniform(
    Tensor::Shape shape,
    std::size_t fan_in,
    std::size_t fan_out,
    std::mt19937& random
) {
    return xavier_uniform(
        std::move(shape),
        fan_in,
        fan_out,
        random,
        execution_backend()
    );
}

Tensor xavier_uniform(
    Tensor::Shape shape,
    std::size_t fan_in,
    std::size_t fan_out,
    std::mt19937& random,
    ExecutionBackend backend
) {
    if (fan_in == 0 || fan_out == 0) {
        throw std::invalid_argument(
            "Xavier initialization fan sizes must be positive"
        );
    }
    const double fan_sum =
        static_cast<double>(fan_in) +
        static_cast<double>(fan_out);
    const float limit = static_cast<float>(
        std::sqrt(6.0 / fan_sum)
    );
    return uniform_tensor(
        std::move(shape),
        -limit,
        limit,
        random,
        backend
    );
}

}  // namespace riftco_transformer
