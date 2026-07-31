#include "transformer_lab/training/adam_optimizer_adapter.hpp"

namespace transformer_lab::training {

AdamOptimizerAdapter::AdamOptimizerAdapter(Adam& adam) noexcept
    : adam_(adam) {}

ExecutionBackend AdamOptimizerAdapter::backend() const noexcept {
    return adam_.backend();
}

std::size_t AdamOptimizerAdapter::step_count() const noexcept {
    return adam_.step_count();
}

OptimizerStepMetrics AdamOptimizerAdapter::step() {
    const AdamStepStats metrics = adam_.step();
    return {
        metrics.step,
        metrics.gradient_norm,
        metrics.clip_scale,
    };
}

void AdamOptimizerAdapter::zero_gradients() {
    adam_.zero_gradients();
}

Adam& AdamOptimizerAdapter::adam() noexcept {
    return adam_;
}

const Adam& AdamOptimizerAdapter::adam() const noexcept {
    return adam_;
}

}  // namespace transformer_lab::training
