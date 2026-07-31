#pragma once

#include "riftco_transformer/nn/linear.hpp"
#include "riftco_transformer/nn/module.hpp"

#include <cstddef>
#include <cstdint>
#include <random>

namespace riftco_transformer {

class DecoderOnlyTransformer;

// Selects the full-sequence causal-attention implementation used by module
// and model forward passes. Incremental decoding has its own paged-attention
// contract and is intentionally independent of this policy.
enum class FullSequenceAttentionKind : std::uint8_t {
    Materialized = 0,
    Flash = 1,
};

struct CausalAttentionResult {
    Variable context;
    Variable probabilities;
};

// [batch, time, model_width] -> [batch, head, time, head_width]
[[nodiscard]] Variable split_attention_heads(
    const Variable& input,
    std::size_t head_count
);

// [batch, head, time, head_width] -> [batch, time, model_width]
[[nodiscard]] Variable merge_attention_heads(const Variable& input);

// Queries, keys, and values all use [batch, head, time, head_width]. This
// probability-returning diagnostic helper always uses materialized attention;
// the runtime selector applies only to context-only module/model forwards.
[[nodiscard]] CausalAttentionResult
causal_scaled_dot_product_attention(
    const Variable& queries,
    const Variable& keys,
    const Variable& values
);

class CausalSelfAttention : public Module {
public:
    CausalSelfAttention(
        std::size_t model_width,
        std::size_t head_count,
        std::mt19937& random,
        FullSequenceAttentionKind attention_kind =
            FullSequenceAttentionKind::Materialized
    );

    [[nodiscard]] std::size_t model_width() const noexcept;
    [[nodiscard]] std::size_t head_count() const noexcept;
    [[nodiscard]] std::size_t head_width() const noexcept;
    [[nodiscard]] FullSequenceAttentionKind
    full_sequence_attention_kind() const noexcept;
    void set_full_sequence_attention_kind(
        FullSequenceAttentionKind attention_kind
    );

    // [batch, time, model_width] -> [batch, time, model_width]
    [[nodiscard]] Variable forward(const Variable& input) const;
    [[nodiscard]] Variable forward(
        const Variable& input,
        FullSequenceAttentionKind attention_kind
    ) const;
    // Transfers parameters in place. Call before building a forward graph.
    void to(ExecutionBackend backend);

    [[nodiscard]] ParameterList parameters();
    [[nodiscard]] ParameterList lora_parameters();

private:
    std::size_t head_count_;
    FullSequenceAttentionKind attention_kind_;
    Linear query_;
    Linear key_;
    Linear value_;
    Linear output_;

    friend class DecoderOnlyTransformer;
};

}  // namespace riftco_transformer
