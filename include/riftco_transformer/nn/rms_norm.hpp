#pragma once

#include "riftco_transformer/nn/module.hpp"

#include <cstddef>

namespace riftco_transformer {

// Root-mean-square normalization used by Llama- and Mistral-family decoders:
// y = x * scale / sqrt(mean(x^2) + epsilon). Unlike LayerNorm, RMSNorm does
// not subtract a mean and has no bias parameter.
[[nodiscard]] Variable rms_norm(
    const Variable& input,
    const Variable& scale,
    float epsilon = 1.0e-5F
);

class RMSNorm : public Module {
public:
    explicit RMSNorm(
        std::size_t width,
        float epsilon = 1.0e-5F
    );
    RMSNorm(Tensor scale, float epsilon = 1.0e-5F);

    [[nodiscard]] std::size_t width() const noexcept;
    [[nodiscard]] float epsilon() const noexcept;
    [[nodiscard]] Variable forward(const Variable& input) const;
    void to(ExecutionBackend backend) override;

    [[nodiscard]] const Parameter& scale() const noexcept;
    [[nodiscard]] ParameterList parameters();

private:
    Parameter scale_;
    float epsilon_;
};

}  // namespace riftco_transformer
