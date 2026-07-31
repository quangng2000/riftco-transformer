#pragma once

#include "riftco_transformer/nn/parameter.hpp"

#include <cstddef>
#include <vector>

namespace riftco_transformer {

struct AdamOptions {
    float learning_rate = 1.0e-3F;
    float beta1 = 0.9F;
    float beta2 = 0.999F;
    float epsilon = 1.0e-8F;
    float maximum_gradient_norm = 1.0F;
};

struct AdamStepStats {
    std::size_t step;
    double gradient_norm;
    double clip_scale;
};

[[nodiscard]] double global_gradient_norm(
    const ParameterList& parameters
);

class Adam {
public:
    explicit Adam(
        ParameterList parameters,
        AdamOptions options = {}
    );

    Adam(const Adam&) = delete;
    Adam& operator=(const Adam&) = delete;
    Adam(Adam&&) noexcept = default;
    Adam& operator=(Adam&&) noexcept = default;

    [[nodiscard]] const AdamOptions& options() const noexcept;
    [[nodiscard]] ExecutionBackend backend() const noexcept;
    [[nodiscard]] std::size_t step_count() const noexcept;
    [[nodiscard]] std::size_t parameter_tensor_count() const noexcept;

    // Applies global gradient clipping and one transactional, bias-corrected
    // Adam update. Replacing the leaf values clears consumed gradients.
    [[nodiscard]] AdamStepStats step();

    void zero_gradients() const;

private:
    AdamOptions options_;
    ParameterList parameters_;
    ExecutionBackend backend_;
    std::vector<Tensor> first_moments_;
    std::vector<Tensor> second_moments_;
    std::size_t step_count_ = 0;
    double beta1_power_ = 1.0;
    double beta2_power_ = 1.0;
};

}  // namespace riftco_transformer
