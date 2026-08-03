#pragma once

#include "riftco_transformer/nn/linear.hpp"
#include "riftco_transformer/nn/module.hpp"

#include <cstddef>
#include <random>

namespace riftco_transformer {

// Repeats each key/value head contiguously. Input and output layouts are
// [batch, head, time, head_width]. Backward sums repeated-head gradients into
// the source head.
[[nodiscard]] Variable repeat_key_value_heads(
    const Variable& input,
    std::size_t repetitions
);

class GroupedQueryAttention : public Module {
public:
    GroupedQueryAttention(
        std::size_t model_width,
        std::size_t query_head_count,
        std::size_t key_value_head_count,
        float rope_theta,
        std::mt19937& random
    );

    [[nodiscard]] std::size_t model_width() const noexcept;
    [[nodiscard]] std::size_t query_head_count() const noexcept;
    [[nodiscard]] std::size_t key_value_head_count() const noexcept;
    [[nodiscard]] std::size_t head_width() const noexcept;
    [[nodiscard]] float rope_theta() const noexcept;

    // [batch, time, model_width] -> [batch, time, model_width]. Position zero
    // is appropriate for full-sequence training; an offset supports future
    // incremental decoding without changing RoPE semantics.
    [[nodiscard]] Variable forward(
        const Variable& input,
        std::size_t position_offset = 0
    ) const;
    void to(ExecutionBackend backend) override;

    [[nodiscard]] ParameterList parameters();

private:
    std::size_t query_head_count_;
    std::size_t key_value_head_count_;
    float rope_theta_;
    Linear query_;
    Linear key_;
    Linear value_;
    Linear output_;
};

}  // namespace riftco_transformer
