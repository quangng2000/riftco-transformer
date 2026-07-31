#pragma once

#include "riftco_transformer/optim/adam.hpp"
#include "riftco_transformer/training/optimizer.hpp"

namespace riftco_transformer::training {

// Non-owning adapter that presents the native Adam optimizer through the
// stage-neutral OptimizerStrategy boundary. The Adam instance must outlive
// the adapter.
class AdamOptimizerAdapter final : public OptimizerStrategy {
public:
    explicit AdamOptimizerAdapter(Adam& adam) noexcept;

    [[nodiscard]] ExecutionBackend backend() const noexcept override;
    [[nodiscard]] std::size_t step_count() const noexcept override;
    [[nodiscard]] OptimizerStepMetrics step() override;
    void zero_gradients() override;

    [[nodiscard]] Adam& adam() noexcept;
    [[nodiscard]] const Adam& adam() const noexcept;

private:
    Adam& adam_;
};

}  // namespace riftco_transformer::training
