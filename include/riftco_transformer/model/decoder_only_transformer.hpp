#pragma once

#include "riftco_transformer/model/activation_checkpointing.hpp"
#include "riftco_transformer/model/decoder_kv_cache.hpp"
#include "riftco_transformer/model/lora.hpp"
#include "riftco_transformer/model/transformer_block.hpp"
#include "riftco_transformer/nn/embedding.hpp"
#include "riftco_transformer/nn/layer_norm.hpp"
#include "riftco_transformer/nn/linear.hpp"
#include "riftco_transformer/nn/module.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <vector>

namespace riftco_transformer {

struct TransformerDimensions {
    std::size_t vocabulary_size = 0;
    std::size_t maximum_context = 0;
    std::size_t model_width = 0;
    std::size_t head_count = 0;
    std::size_t block_count = 0;
    std::size_t feed_forward_width = 0;
};

// Backend-neutral snapshot of one immutable packed Linear base weight. The
// decoder returns these entries in the same stable order used by its
// model-wide quantization pass: six projections per block, followed by the
// language-model head. No operation on this type materializes FP32 weights.
struct PackedLinearWeightState {
    QuantizedWeight::Shape shape;
    std::size_t block_size = 0;
    Nf4Payload payload;
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
    void to(ExecutionBackend backend) override;

    // Transactionally packs every Linear base weight (all six projection
    // kinds per block plus the language-model head) before LoRA attachment.
    // The LoRA target mask remains an independent adapter-only selection.
    void quantize_linear_weights_nf4(
        std::size_t block_size =
            QuantizedWeight::kDefaultNf4BlockSize
    );
    void quantize_linear_weights_nf4_double_quantized(
        std::size_t block_size =
            QuantizedWeight::kDefaultNf4BlockSize,
        std::size_t scale_block_size = 256
    );
    [[nodiscard]] bool
    has_quantized_linear_weights() const noexcept;
    [[nodiscard]] std::size_t
    quantized_linear_weight_count() const noexcept;
    [[nodiscard]] std::size_t
    double_quantized_linear_weight_count() const noexcept;
    [[nodiscard]] QuantizedMemoryUsage
    quantized_memory_usage() const noexcept;
    [[nodiscard]] std::vector<PackedLinearWeightState>
    packed_linear_weight_state() const;
    // Transactionally replaces an already-packed model's immutable base
    // weights. Shapes, count, and NF4 metadata are validated before commit;
    // active LoRA adapters and their optimizer-owned Parameters are untouched.
    void load_packed_linear_weight_state(
        std::span<const PackedLinearWeightState> state
    );

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
    void quantize_linear_weights_nf4_impl(
        std::size_t block_size,
        std::optional<std::size_t> scale_block_size
    );
    [[nodiscard]] std::vector<Linear*> all_linear_projections();
    [[nodiscard]] std::vector<const Linear*>
    all_linear_projections() const;
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

}  // namespace riftco_transformer
