#pragma once

#include "riftco_transformer/core/backend.hpp"

#include <cstddef>

namespace riftco_transformer::training {

struct OptimizerStepMetrics {
    std::size_t step;
    double gradient_norm;
    double clip_scale;
};

// Dependency-inversion boundary for parameter update strategies.
class OptimizerStrategy {
public:
    OptimizerStrategy() = default;
    OptimizerStrategy(const OptimizerStrategy&) = delete;
    OptimizerStrategy& operator=(const OptimizerStrategy&) = delete;
    OptimizerStrategy(OptimizerStrategy&&) = delete;
    OptimizerStrategy& operator=(OptimizerStrategy&&) = delete;
    virtual ~OptimizerStrategy() = default;

    [[nodiscard]] virtual ExecutionBackend backend() const noexcept = 0;
    [[nodiscard]] virtual std::size_t step_count() const noexcept = 0;
    [[nodiscard]] virtual OptimizerStepMetrics step() = 0;
    virtual void zero_gradients() = 0;
};

}  // namespace riftco_transformer::training
