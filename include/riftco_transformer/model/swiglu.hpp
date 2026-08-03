#pragma once

#include "riftco_transformer/nn/linear.hpp"
#include "riftco_transformer/nn/module.hpp"

#include <cstddef>
#include <random>

namespace riftco_transformer {

// Llama/Mistral feed-forward network:
// down(silu(gate(input)) * up(input)). All projections are bias-free.
class SwiGLU : public Module {
public:
    SwiGLU(
        std::size_t model_width,
        std::size_t hidden_width,
        std::mt19937& random
    );

    [[nodiscard]] std::size_t model_width() const noexcept;
    [[nodiscard]] std::size_t hidden_width() const noexcept;
    [[nodiscard]] Variable forward(const Variable& input) const;
    void to(ExecutionBackend backend) override;

    [[nodiscard]] ParameterList parameters();

private:
    Linear gate_;
    Linear up_;
    Linear down_;
};

}  // namespace riftco_transformer
