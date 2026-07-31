#pragma once

#include "riftco_transformer/nn/module.hpp"

#include <cstddef>

namespace riftco_transformer {

[[nodiscard]] Variable layer_norm(
    const Variable& input,
    const Variable& scale,
    const Variable& bias,
    float epsilon = 1.0e-5F
);

class LayerNorm : public Module {
public:
    explicit LayerNorm(
        std::size_t width,
        float epsilon = 1.0e-5F
    );
    LayerNorm(
        Tensor scale,
        Tensor bias,
        float epsilon = 1.0e-5F
    );

    [[nodiscard]] std::size_t width() const noexcept;
    [[nodiscard]] float epsilon() const noexcept;
    [[nodiscard]] Variable forward(const Variable& input) const;
    // Transfers parameters in place. Call before building a forward graph.
    void to(ExecutionBackend backend);

    [[nodiscard]] const Parameter& scale() const noexcept;
    [[nodiscard]] const Parameter& bias() const noexcept;
    [[nodiscard]] ParameterList parameters();

private:
    Parameter scale_;
    Parameter bias_;
    float epsilon_;
};

}  // namespace riftco_transformer
