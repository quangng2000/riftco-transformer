#pragma once

#include "riftco_transformer/model/causal_self_attention.hpp"
#include "riftco_transformer/model/feed_forward.hpp"
#include "riftco_transformer/nn/layer_norm.hpp"
#include "riftco_transformer/nn/module.hpp"

#include <cstddef>
#include <random>

namespace riftco_transformer {

class DecoderOnlyTransformer;

class TransformerBlock : public Module {
public:
    TransformerBlock(
        std::size_t model_width,
        std::size_t head_count,
        std::size_t feed_forward_width,
        std::mt19937& random,
        float layer_norm_epsilon = 1.0e-5F,
        FullSequenceAttentionKind attention_kind =
            FullSequenceAttentionKind::Materialized
    );

    TransformerBlock(const TransformerBlock&) = delete;
    TransformerBlock& operator=(const TransformerBlock&) = delete;
    TransformerBlock(TransformerBlock&&) = delete;
    TransformerBlock& operator=(TransformerBlock&&) = delete;

    [[nodiscard]] std::size_t model_width() const noexcept;
    [[nodiscard]] std::size_t head_count() const noexcept;
    [[nodiscard]] std::size_t feed_forward_width() const noexcept;
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
    std::size_t model_width_;
    std::size_t head_count_;
    std::size_t feed_forward_width_;
    LayerNorm attention_norm_;
    CausalSelfAttention attention_;
    LayerNorm feed_forward_norm_;
    FeedForward feed_forward_;

    friend class DecoderOnlyTransformer;
};

}  // namespace riftco_transformer
