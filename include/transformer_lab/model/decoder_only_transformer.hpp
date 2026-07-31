#pragma once

#include "transformer_lab/model/activation_checkpointing.hpp"
#include "transformer_lab/model/decoder_kv_cache.hpp"
#include "transformer_lab/model/lora.hpp"
#include "transformer_lab/model/transformer_block.hpp"
#include "transformer_lab/nn/embedding.hpp"
#include "transformer_lab/nn/layer_norm.hpp"
#include "transformer_lab/nn/linear.hpp"
#include "transformer_lab/nn/module.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <vector>

namespace transformer_lab {

struct TransformerDimensions {
    std::size_t vocabulary_size = 0;
    std::size_t maximum_context = 0;
    std::size_t model_width = 0;
    std::size_t head_count = 0;
    std::size_t block_count = 0;
    std::size_t feed_forward_width = 0;
};

class DecoderOnlyTransformer : public Module {
public:
    DecoderOnlyTransformer(
        TransformerDimensions dimensions,
        std::mt19937& random,
        float layer_norm_epsilon = 1.0e-5F,
        FullSequenceAttentionKind attention_kind =
            FullSequenceAttentionKind::Materialized,
        ActivationCheckpointingKind activation_checkpointing =
            ActivationCheckpointingKind::Disabled
    );

    DecoderOnlyTransformer(const DecoderOnlyTransformer&) = delete;
    DecoderOnlyTransformer& operator=(
        const DecoderOnlyTransformer&
    ) = delete;
    DecoderOnlyTransformer(DecoderOnlyTransformer&&) = delete;
    DecoderOnlyTransformer& operator=(
        DecoderOnlyTransformer&&
    ) = delete;

    [[nodiscard]] const TransformerDimensions& dimensions() const noexcept;
    [[nodiscard]] float layer_norm_epsilon() const noexcept;
    [[nodiscard]] ExecutionBackend backend() const noexcept;
    [[nodiscard]] FullSequenceAttentionKind
    full_sequence_attention_kind() const noexcept;
    void set_full_sequence_attention_kind(
        FullSequenceAttentionKind attention_kind
    );
    [[nodiscard]] ActivationCheckpointingKind
    activation_checkpointing_kind() const noexcept;
    void set_activation_checkpointing_kind(
        ActivationCheckpointingKind activation_checkpointing
    );

    // token_shape must be [batch, time]. Returns
    // [batch, time, vocabulary_size].
    [[nodiscard]] Variable forward(
        std::span<const TokenId> token_ids,
        Tensor::Shape token_shape
    ) const;
    // Serving-only, one-token decode using the cache's current size as the
    // learned position ID. Returns detached [1, 1, vocabulary_size] logits.
    [[nodiscard]] Tensor decode_token(
        TokenId token_id,
        DecoderKeyValueCache& cache
    ) const;
    // Transfers parameters in place. Call before building a forward graph.
    void to(ExecutionBackend backend);

    // Attaches adapters to the configured projection kinds in every block.
    // This is a one-time operation for the lifetime of the model.
    void attach_lora(const LoraConfig& config);
    [[nodiscard]] bool has_lora() const noexcept;
    [[nodiscard]] const LoraConfig& lora_config() const;
    [[nodiscard]] ParameterList lora_parameters();
    // One-way, model-wide transactional merge.
    void merge_lora();

    // Base parameters only, with order and names independent of LoRA state.
    [[nodiscard]] ParameterList parameters();

private:
    [[nodiscard]] std::vector<Linear*> selected_lora_projections(
        LoraTargetMask targets
    );

    TransformerDimensions dimensions_;
    float layer_norm_epsilon_;
    FullSequenceAttentionKind attention_kind_;
    ActivationCheckpointingKind activation_checkpointing_;
    Embedding token_embedding_;
    Embedding position_embedding_;
    std::vector<std::shared_ptr<TransformerBlock>> blocks_;
    ModuleList block_modules_;
    LayerNorm final_norm_;
    Linear language_model_head_;
    std::optional<LoraConfig> lora_config_;
    bool lora_was_merged_ = false;
    std::shared_ptr<std::uint64_t> graph_structure_version_;
};

}  // namespace transformer_lab
