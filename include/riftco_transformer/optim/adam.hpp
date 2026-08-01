#pragma once

#include "riftco_transformer/nn/parameter.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace riftco_transformer {

enum class AdamStateStorageKind : std::uint8_t {
    Contiguous,
    // Splits both moment tensors into bounded allocations. CUDA Tensor pages
    // use managed memory and may migrate; this is not a disk-spill contract.
    Paged,
};

struct AdamOptions {
    float learning_rate = 1.0e-3F;
    float beta1 = 0.9F;
    float beta2 = 0.999F;
    float epsilon = 1.0e-8F;
    float maximum_gradient_norm = 1.0F;
    AdamStateStorageKind state_storage =
        AdamStateStorageKind::Contiguous;
    std::size_t page_size = 4096;
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
    [[nodiscard]] AdamStateStorageKind
    state_storage_kind() const noexcept;
    [[nodiscard]] std::size_t state_page_size() const noexcept;
    [[nodiscard]] std::size_t state_page_count() const noexcept;
    [[nodiscard]] std::size_t state_payload_bytes() const noexcept;

    // Applies global gradient clipping and one transactional, bias-corrected
    // Adam update. Replacing the leaf values clears consumed gradients.
    [[nodiscard]] AdamStepStats step();

    void zero_gradients() const;

private:
    struct StatePage {
        std::size_t offset;
        Tensor first_moment;
        Tensor second_moment;
    };

    AdamOptions options_;
    ParameterList parameters_;
    ExecutionBackend backend_;
    std::vector<Tensor> first_moments_;
    std::vector<Tensor> second_moments_;
    std::vector<std::vector<StatePage>> state_pages_;
    std::size_t state_payload_bytes_ = 0;
    std::size_t step_count_ = 0;
    double beta1_power_ = 1.0;
    double beta2_power_ = 1.0;
};

}  // namespace riftco_transformer
