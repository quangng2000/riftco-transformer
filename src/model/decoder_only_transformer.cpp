#include "riftco_transformer/model/decoder_only_transformer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace riftco_transformer {
namespace {

TransformerDimensions checked_dimensions(
    TransformerDimensions dimensions
) {
    if (dimensions.vocabulary_size == 0 ||
        dimensions.maximum_context == 0 ||
        dimensions.model_width == 0 ||
        dimensions.head_count == 0 ||
        dimensions.block_count == 0 ||
        dimensions.feed_forward_width == 0) {
        throw std::invalid_argument(
            "transformer dimensions must be greater than zero"
        );
    }
    if (dimensions.model_width % dimensions.head_count != 0) {
        throw std::invalid_argument(
            "transformer model width must be divisible by head count"
        );
    }

    const auto largest_token_id =
        static_cast<std::size_t>(
            std::numeric_limits<TokenId>::max()
        );
    if (dimensions.vocabulary_size - 1 > largest_token_id ||
        dimensions.maximum_context - 1 > largest_token_id) {
        throw std::invalid_argument(
            "vocabulary and context must fit the token ID type"
        );
    }
    return dimensions;
}

float checked_epsilon(float epsilon) {
    if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
        throw std::invalid_argument(
            "transformer layer normalization epsilon "
            "must be finite and positive"
        );
    }
    return epsilon;
}

FullSequenceAttentionKind checked_attention_kind(
    FullSequenceAttentionKind attention_kind
) {
    switch (attention_kind) {
        case FullSequenceAttentionKind::Materialized:
        case FullSequenceAttentionKind::Flash:
            return attention_kind;
    }
    throw std::invalid_argument(
        "full-sequence attention kind is not recognized"
    );
}

ActivationCheckpointingKind checked_activation_checkpointing(
    ActivationCheckpointingKind activation_checkpointing
) {
    switch (activation_checkpointing) {
        case ActivationCheckpointingKind::Disabled:
        case ActivationCheckpointingKind::TransformerBlock:
            return activation_checkpointing;
    }
    throw std::invalid_argument(
        "activation checkpointing kind is not recognized"
    );
}

std::vector<std::shared_ptr<TransformerBlock>> make_blocks(
    const TransformerDimensions& dimensions,
    std::mt19937& random,
    float layer_norm_epsilon,
    FullSequenceAttentionKind attention_kind
) {
    std::vector<std::shared_ptr<TransformerBlock>> blocks;
    blocks.reserve(dimensions.block_count);
    for (std::size_t index = 0;
         index < dimensions.block_count;
         ++index) {
        blocks.push_back(std::make_shared<TransformerBlock>(
            dimensions.model_width,
            dimensions.head_count,
            dimensions.feed_forward_width,
            random,
            layer_norm_epsilon,
            attention_kind
        ));
    }
    return blocks;
}

void validate_decode_cache(
    const TransformerDimensions& dimensions,
    ExecutionBackend model_backend,
    const DecoderKeyValueCache& cache
) {
    const std::size_t expected_head_width =
        dimensions.model_width / dimensions.head_count;
    if (cache.backend() != model_backend) {
        throw std::invalid_argument(
            "decoder cache backend must match the model backend"
        );
    }
    if (cache.layer_count() != dimensions.block_count) {
        throw std::invalid_argument(
            "decoder cache layer count must match the model"
        );
    }
    if (cache.head_count() != dimensions.head_count ||
        cache.head_width() != expected_head_width) {
        throw std::invalid_argument(
            "decoder cache head dimensions must match the model"
        );
    }
    if (cache.capacity() == 0 ||
        cache.capacity() > dimensions.maximum_context) {
        throw std::invalid_argument(
            "decoder cache capacity must be in the model context range"
        );
    }
    if (cache.size() > cache.capacity()) {
        throw std::logic_error(
            "decoder cache size exceeds its capacity"
        );
    }
    if (cache.size() == cache.capacity()) {
        throw std::length_error("decoder cache capacity is exhausted");
    }
}

class CacheTokenTransaction final {
public:
    explicit CacheTokenTransaction(DecoderKeyValueCache& cache)
        : cache_(cache) {
        cache_.begin_token();
    }

    CacheTokenTransaction(const CacheTokenTransaction&) = delete;
    CacheTokenTransaction& operator=(
        const CacheTokenTransaction&
    ) = delete;

    ~CacheTokenTransaction() noexcept {
        if (active_) {
            cache_.abort_token();
        }
    }

    void commit() {
        cache_.commit_token();
        active_ = false;
    }

private:
    DecoderKeyValueCache& cache_;
    bool active_ = true;
};

}  // namespace

DecoderOnlyTransformer::DecoderOnlyTransformer(
    TransformerDimensions dimensions,
    std::mt19937& random,
    float layer_norm_epsilon,
    FullSequenceAttentionKind attention_kind,
    ActivationCheckpointingKind activation_checkpointing
)
    : dimensions_(checked_dimensions(dimensions)),
      layer_norm_epsilon_(checked_epsilon(layer_norm_epsilon)),
      attention_kind_(checked_attention_kind(attention_kind)),
      activation_checkpointing_(
          checked_activation_checkpointing(activation_checkpointing)
      ),
      token_embedding_(
          dimensions_.vocabulary_size,
          dimensions_.model_width,
          random
      ),
      position_embedding_(
          dimensions_.maximum_context,
          dimensions_.model_width,
          random
      ),
      blocks_(make_blocks(
          dimensions_,
          random,
          layer_norm_epsilon_,
          attention_kind_
      )),
      final_norm_(
          dimensions_.model_width,
          layer_norm_epsilon_
      ),
      language_model_head_(
          dimensions_.model_width,
          dimensions_.vocabulary_size,
          random
      ),
      graph_structure_version_(
          std::make_shared<std::uint64_t>(0)
      ) {
    register_module("token_embedding", token_embedding_);
    register_module("position_embedding", position_embedding_);
    for (const auto& block : blocks_) {
        block_modules_.append(block);
    }
    register_module("blocks", block_modules_);
    register_module("final_norm", final_norm_);
    register_module("language_model_head", language_model_head_);
}

const TransformerDimensions&
DecoderOnlyTransformer::dimensions() const noexcept {
    return dimensions_;
}

float DecoderOnlyTransformer::layer_norm_epsilon() const noexcept {
    return layer_norm_epsilon_;
}

ExecutionBackend DecoderOnlyTransformer::backend() const noexcept {
    return token_embedding_.weight().value().backend();
}

FullSequenceAttentionKind
DecoderOnlyTransformer::full_sequence_attention_kind() const noexcept {
    return attention_kind_;
}

void DecoderOnlyTransformer::set_full_sequence_attention_kind(
    FullSequenceAttentionKind attention_kind
) {
    const auto checked = checked_attention_kind(attention_kind);
    for (auto& block : blocks_) {
        block->set_full_sequence_attention_kind(checked);
    }
    attention_kind_ = checked;
}

ActivationCheckpointingKind
DecoderOnlyTransformer::activation_checkpointing_kind() const noexcept {
    return activation_checkpointing_;
}

void DecoderOnlyTransformer::set_activation_checkpointing_kind(
    ActivationCheckpointingKind activation_checkpointing
) {
    activation_checkpointing_ =
        checked_activation_checkpointing(activation_checkpointing);
}

Variable DecoderOnlyTransformer::forward(
    std::span<const TokenId> token_ids,
    Tensor::Shape token_shape
) const {
    if (token_shape.size() != 2) {
        throw std::invalid_argument(
            "transformer token shape must be [batch, time]"
        );
    }
    const auto batch = token_shape[0];
    const auto time = token_shape[1];
    if (batch == 0 || time == 0) {
        throw std::invalid_argument(
            "transformer batch and time must be greater than zero"
        );
    }
    if (batch > std::numeric_limits<std::size_t>::max() / time ||
        batch * time != token_ids.size()) {
        throw std::invalid_argument(
            "transformer token count must match [batch, time]"
        );
    }
    if (time > dimensions_.maximum_context) {
        throw std::invalid_argument(
            "transformer sequence length exceeds maximum context"
        );
    }

    std::vector<TokenId> position_ids(token_ids.size());
    for (std::size_t batch_index = 0;
         batch_index < batch;
         ++batch_index) {
        for (std::size_t time_index = 0;
             time_index < time;
             ++time_index) {
            position_ids[batch_index * time + time_index] =
                static_cast<TokenId>(time_index);
        }
    }

    Variable hidden =
        token_embedding_.forward(token_ids, token_shape) +
        position_embedding_.forward(position_ids, token_shape);
    for (const auto& block : blocks_) {
        if (activation_checkpointing_ ==
            ActivationCheckpointingKind::Disabled) {
            hidden = block->forward(hidden, attention_kind_);
            continue;
        }

        std::vector<Variable> dependencies;
        const auto append_dependencies =
            [&](ParameterList parameters) {
                dependencies.reserve(
                    dependencies.size() + parameters.size()
                );
                for (const auto& named_parameter : parameters) {
                    if (named_parameter.parameter == nullptr) {
                        throw std::logic_error(
                            "checkpoint block has a null parameter"
                        );
                    }
                    dependencies.push_back(
                        named_parameter.parameter->variable()
                    );
                }
            };
        append_dependencies(block->parameters());
        append_dependencies(block->lora_parameters());

        const auto structure_version = graph_structure_version_;
        const std::uint64_t expected_structure_version =
            *structure_version;
        const auto captured_attention_kind = attention_kind_;
        hidden = checkpoint(
            hidden,
            dependencies,
            [
                block,
                structure_version,
                expected_structure_version,
                captured_attention_kind
            ](const Variable& checkpoint_input) {
                if (*structure_version !=
                    expected_structure_version) {
                    throw std::logic_error(
                        "cannot replay a checkpoint after the model "
                        "structure changed"
                    );
                }
                return block->forward(
                    checkpoint_input,
                    captured_attention_kind
                );
            }
        );
    }
    return language_model_head_.forward(
        final_norm_.forward(hidden)
    );
}

Tensor DecoderOnlyTransformer::decode_token(
    TokenId token_id,
    DecoderKeyValueCache& cache
) const {
    if (static_cast<std::size_t>(token_id) >=
        dimensions_.vocabulary_size) {
        throw std::out_of_range(
            "decoder token is outside the model vocabulary"
        );
    }
    validate_decode_cache(dimensions_, backend(), cache);

    const std::size_t position = cache.size();
    const TokenId position_id =
        static_cast<TokenId>(position);
    CacheTokenTransaction transaction(cache);

    const Tensor::Shape token_shape{1, 1};
    Variable hidden =
        token_embedding_.forward(
            std::span<const TokenId>(&token_id, 1),
            token_shape
        ) +
        position_embedding_.forward(
            std::span<const TokenId>(&position_id, 1),
            token_shape
        );

    const Tensor::Shape expected_attention_shape{
        1,
        dimensions_.head_count,
        1,
        dimensions_.model_width / dimensions_.head_count,
    };
    for (std::size_t layer = 0;
         layer < blocks_.size();
         ++layer) {
        const TransformerBlock& block = *blocks_[layer];
        const Variable normalized =
            block.attention_norm_.forward(hidden);
        const Variable query = split_attention_heads(
            block.attention_.query_.forward(normalized),
            dimensions_.head_count
        );
        const Variable key = split_attention_heads(
            block.attention_.key_.forward(normalized),
            dimensions_.head_count
        );
        const Variable value = split_attention_heads(
            block.attention_.value_.forward(normalized),
            dimensions_.head_count
        );

        Tensor context = cache.append_and_attend(
            layer,
            query.value(),
            key.value(),
            value.value()
        );
        if (context.shape() != expected_attention_shape) {
            throw std::runtime_error(
                "decoder cache returned an unexpected context shape"
            );
        }
        if (context.backend() != backend()) {
            throw std::runtime_error(
                "decoder cache returned context on the wrong backend"
            );
        }

        const Variable attention_state =
            hidden +
            block.attention_.output_.forward(
                merge_attention_heads(
                    Variable(std::move(context), false)
                )
            );
        hidden =
            attention_state +
            block.feed_forward_.forward(
                block.feed_forward_norm_.forward(
                    attention_state
                )
            );
    }

    const Variable logits = language_model_head_.forward(
        final_norm_.forward(hidden)
    );
    Tensor detached_logits = logits.value();
    transaction.commit();
    return detached_logits;
}

void DecoderOnlyTransformer::to(ExecutionBackend backend) {
    Module::to(backend);
}

void DecoderOnlyTransformer::quantize_linear_weights_nf4(
    std::size_t block_size
) {
    quantize_linear_weights_nf4_impl(block_size, std::nullopt);
}

void DecoderOnlyTransformer::quantize_linear_weights_nf4_double_quantized(
    std::size_t block_size,
    std::size_t scale_block_size
) {
    quantize_linear_weights_nf4_impl(block_size, scale_block_size);
}

void DecoderOnlyTransformer::quantize_linear_weights_nf4_impl(
    std::size_t block_size,
    std::optional<std::size_t> scale_block_size
) {
    if (has_lora() || lora_was_merged_) {
        throw std::invalid_argument(
            "transformer Linear weights must be quantized before LoRA"
        );
    }
    if (has_quantized_linear_weights()) {
        throw std::invalid_argument(
            "transformer Linear weights have already been quantized"
        );
    }
    if (*graph_structure_version_ ==
        std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error(
            "transformer graph structure version overflow"
        );
    }

    std::vector<Linear*> projections = all_linear_projections();
    std::vector<QuantizedWeight> candidates;
    candidates.reserve(projections.size());
    for (Linear* projection : projections) {
        candidates.push_back(
            projection->prepare_weight_quantization_nf4(
                block_size,
                scale_block_size
            )
        );
    }
    for (std::size_t index = 0; index < projections.size(); ++index) {
        projections[index]->commit_prepared_weight_quantization(
            std::move(candidates[index])
        );
    }
    ++*graph_structure_version_;
}

bool DecoderOnlyTransformer::has_quantized_linear_weights()
    const noexcept {
    return quantized_linear_weight_count() != 0;
}

std::size_t DecoderOnlyTransformer::quantized_linear_weight_count()
    const noexcept {
    std::size_t result = 0;
    const auto count = [&](const Linear& projection) {
        result += projection.has_quantized_weight() ? 1U : 0U;
    };
    for (const auto& block : blocks_) {
        count(block->attention_.query_);
        count(block->attention_.key_);
        count(block->attention_.value_);
        count(block->attention_.output_);
        count(block->feed_forward_.expand_);
        count(block->feed_forward_.project_);
    }
    count(language_model_head_);
    return result;
}

std::size_t
DecoderOnlyTransformer::double_quantized_linear_weight_count()
    const noexcept {
    std::size_t result = 0;
    const auto count = [&](const Linear& projection) {
        if (projection.has_quantized_weight() &&
            projection.quantized_weight()
                .uses_double_quantized_scales()) {
            ++result;
        }
    };
    for (const auto& block : blocks_) {
        count(block->attention_.query_);
        count(block->attention_.key_);
        count(block->attention_.value_);
        count(block->attention_.output_);
        count(block->feed_forward_.expand_);
        count(block->feed_forward_.project_);
    }
    count(language_model_head_);
    return result;
}

QuantizedMemoryUsage
DecoderOnlyTransformer::quantized_memory_usage() const noexcept {
    QuantizedMemoryUsage result;
    const auto add_saturated = [](std::size_t& destination,
                                  std::size_t value) {
        destination = value >
                              std::numeric_limits<std::size_t>::max() -
                                  destination
                          ? std::numeric_limits<std::size_t>::max()
                          : destination + value;
    };
    const auto accumulate = [&](const Linear& projection) {
        const QuantizedMemoryUsage current =
            projection.quantized_memory_usage();
        add_saturated(
            result.packed_code_bytes,
            current.packed_code_bytes
        );
        add_saturated(result.scale_bytes, current.scale_bytes);
        add_saturated(
            result.logical_payload_bytes,
            current.logical_payload_bytes
        );
        add_saturated(
            result.resident_payload_bytes,
            current.resident_payload_bytes
        );
        add_saturated(
            result.fp32_equivalent_bytes,
            current.fp32_equivalent_bytes
        );
        add_saturated(
            result.fp32_scale_bytes,
            current.fp32_scale_bytes
        );
        add_saturated(
            result.scale_code_bytes,
            current.scale_code_bytes
        );
        add_saturated(
            result.second_level_scale_bytes,
            current.second_level_scale_bytes
        );
        add_saturated(
            result.scale_offset_bytes,
            current.scale_offset_bytes
        );
    };
    for (const auto& block : blocks_) {
        accumulate(block->attention_.query_);
        accumulate(block->attention_.key_);
        accumulate(block->attention_.value_);
        accumulate(block->attention_.output_);
        accumulate(block->feed_forward_.expand_);
        accumulate(block->feed_forward_.project_);
    }
    accumulate(language_model_head_);
    return result;
}

std::vector<PackedLinearWeightState>
DecoderOnlyTransformer::packed_linear_weight_state() const {
    const std::vector<const Linear*> projections = all_linear_projections();
    std::vector<PackedLinearWeightState> result;
    result.reserve(projections.size());
    for (const Linear* projection : projections) {
        if (projection == nullptr || !projection->has_quantized_weight()) {
            throw std::logic_error(
                "packed model state requires every Linear base weight to be NF4"
            );
        }
        const QuantizedWeight& weight = projection->quantized_weight();
        result.push_back(PackedLinearWeightState{
            weight.shape(),
            weight.block_size(),
            weight.copy_payload_to_host(),
        });
    }
    return result;
}

void DecoderOnlyTransformer::load_packed_linear_weight_state(
    std::span<const PackedLinearWeightState> state
) {
    std::vector<Linear*> projections = all_linear_projections();
    if (state.size() != projections.size()) {
        throw std::invalid_argument(
            "packed model state weight count does not match the decoder"
        );
    }

    std::vector<QuantizedWeight> candidates;
    candidates.reserve(projections.size());
    for (std::size_t index = 0; index < projections.size(); ++index) {
        Linear* projection = projections[index];
        if (projection == nullptr || !projection->has_quantized_weight()) {
            throw std::invalid_argument(
                "packed model state can only restore into a fully quantized decoder"
            );
        }
        const QuantizedWeight& current = projection->quantized_weight();
        if (state[index].shape != current.shape()) {
            throw std::invalid_argument(
                "packed model state weight shape does not match the decoder"
            );
        }
        candidates.push_back(QuantizedWeight::from_packed_nf4(
            state[index].shape,
            state[index].block_size,
            state[index].payload,
            current.backend()
        ));
    }

    for (std::size_t index = 0; index < projections.size(); ++index) {
        projections[index]->quantized_weight_ = std::move(candidates[index]);
    }
}

void DecoderOnlyTransformer::attach_lora(
    const LoraConfig& config
) {
    if (has_lora() || lora_was_merged_) {
        throw std::logic_error(
            "LoRA can only be attached once to a transformer model"
        );
    }
    if (*graph_structure_version_ ==
        std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error(
            "transformer graph structure version overflow"
        );
    }
    if (config.rank == 0) {
        throw std::invalid_argument(
            "LoRA rank must be greater than zero"
        );
    }
    if (!std::isfinite(config.alpha) ||
        config.alpha <= 0.0F) {
        throw std::invalid_argument(
            "LoRA alpha must be finite and positive"
        );
    }
    const float scale =
        config.alpha / static_cast<float>(config.rank);
    if (!std::isfinite(scale) || scale <= 0.0F) {
        throw std::invalid_argument(
            "LoRA alpha divided by rank must be finite and positive"
        );
    }
    if (config.targets == 0 ||
        (config.targets & ~kLoraAllTargets) != 0) {
        throw std::invalid_argument(
            "LoRA targets must be a nonempty known target mask"
        );
    }

    std::vector<Linear*> selected =
        selected_lora_projections(config.targets);
    if (selected.empty()) {
        throw std::invalid_argument(
            "LoRA target mask selected no projections"
        );
    }
    for (const Linear* projection : selected) {
        if (config.rank >
            std::min(
                projection->input_width(),
                projection->output_width()
            )) {
            throw std::invalid_argument(
                "LoRA rank exceeds a selected projection dimension"
            );
        }
        if (projection->lora_ != nullptr ||
            projection->lora_was_merged_) {
            throw std::logic_error(
                "a selected projection already used LoRA"
            );
        }
    }

    const ParameterList base_parameters = parameters();
    if (base_parameters.empty() ||
        base_parameters.front().parameter == nullptr) {
        throw std::logic_error(
            "transformer base parameters are incomplete"
        );
    }
    const ExecutionBackend backend =
        base_parameters.front().parameter->value().backend();
    for (const auto& named_parameter : base_parameters) {
        if (named_parameter.parameter == nullptr) {
            throw std::logic_error(
                "transformer base parameters are incomplete"
            );
        }
        if (named_parameter.parameter->value().backend() != backend ||
            named_parameter.parameter->gradient().backend() != backend) {
            throw std::logic_error(
                "transformer base parameters must share one backend"
            );
        }
    }
    for (const Linear* projection : all_linear_projections()) {
        if (projection->has_quantized_weight() &&
            projection->quantized_weight().backend() != backend) {
            throw std::logic_error(
                "transformer quantized weights must share the model backend"
            );
        }
    }

    std::mt19937 random(config.random_seed);
    std::size_t attached_count = 0;
    try {
        for (Linear* projection : selected) {
            projection->attach_lora(
                config.rank,
                config.alpha,
                random
            );
            ++attached_count;
        }
    } catch (...) {
        while (attached_count > 0) {
            --attached_count;
            selected[attached_count]->discard_unmerged_lora();
        }
        throw;
    }
    lora_config_ = config;
    ++*graph_structure_version_;
}

bool DecoderOnlyTransformer::has_lora() const noexcept {
    return lora_config_.has_value();
}

const LoraConfig& DecoderOnlyTransformer::lora_config() const {
    if (!has_lora()) {
        throw std::logic_error(
            "transformer model has no active LoRA configuration"
        );
    }
    return *lora_config_;
}

ParameterList DecoderOnlyTransformer::lora_parameters() {
    if (!has_lora()) {
        return {};
    }

    ParameterList result;
    for (std::size_t index = 0; index < blocks_.size(); ++index) {
        append_parameter_group(
            result,
            "blocks." + std::to_string(index),
            blocks_[index]->lora_parameters()
        );
    }
    append_parameter_group(
        result,
        "language_model_head",
        language_model_head_.lora_parameters()
    );
    return result;
}

void DecoderOnlyTransformer::merge_lora() {
    if (!has_lora()) {
        if (lora_was_merged_) {
            throw std::logic_error(
                "LoRA has already been merged into this transformer model"
            );
        }
        throw std::logic_error(
            "cannot merge LoRA before attaching adapters"
        );
    }
    if (*graph_structure_version_ ==
        std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error(
            "transformer graph structure version overflow"
        );
    }

    struct PreparedMaterialization {
        Linear* projection;
        Linear::PreparedMaterializedWeight weight;
        bool includes_lora_delta;
    };
    std::vector<PreparedMaterialization> prepared;
    const std::vector<Linear*> projections = all_linear_projections();
    prepared.reserve(projections.size());

    // QLoRA export deliberately materializes every packed base projection.
    // Preparing every FP32 value, gradient, and canonical Parameter proxy
    // before committing preserves validation/allocation atomicity, at the
    // cost of a documented one-time export memory spike.
    for (Linear* projection : projections) {
        const bool includes_lora_delta = projection->has_lora();
        if (!projection->has_quantized_weight() &&
            !includes_lora_delta) {
            continue;
        }
        prepared.push_back({
            projection,
            projection->prepare_materialized_weight(
                includes_lora_delta
            ),
            includes_lora_delta,
        });
    }
    for (const auto& materialization : prepared) {
        materialization.projection
            ->validate_prepared_materialized_weight(
                materialization.weight
            );
    }
    for (auto& materialization : prepared) {
        materialization.projection->commit_prepared_materialized_weight(
            std::move(materialization.weight),
            materialization.includes_lora_delta
        );
    }

    lora_config_.reset();
    lora_was_merged_ = true;
    ++*graph_structure_version_;
}

ParameterList DecoderOnlyTransformer::parameters() {
    return Module::parameters();
}

std::vector<Linear*>
DecoderOnlyTransformer::all_linear_projections() {
    std::vector<Linear*> result;
    result.reserve(blocks_.size() * 6U + 1U);
    for (auto& block : blocks_) {
        result.push_back(&block->attention_.query_);
        result.push_back(&block->attention_.key_);
        result.push_back(&block->attention_.value_);
        result.push_back(&block->attention_.output_);
        result.push_back(&block->feed_forward_.expand_);
        result.push_back(&block->feed_forward_.project_);
    }
    result.push_back(&language_model_head_);
    return result;
}

std::vector<const Linear*>
DecoderOnlyTransformer::all_linear_projections() const {
    std::vector<const Linear*> result;
    result.reserve(blocks_.size() * 6U + 1U);
    for (const auto& block : blocks_) {
        result.push_back(&block->attention_.query_);
        result.push_back(&block->attention_.key_);
        result.push_back(&block->attention_.value_);
        result.push_back(&block->attention_.output_);
        result.push_back(&block->feed_forward_.expand_);
        result.push_back(&block->feed_forward_.project_);
    }
    result.push_back(&language_model_head_);
    return result;
}

std::vector<Linear*>
DecoderOnlyTransformer::selected_lora_projections(
    LoraTargetMask targets
) {
    std::vector<Linear*> result;
    result.reserve(blocks_.size() * 6U + 1U);
    for (auto& block : blocks_) {
        if ((targets & kLoraAttentionQuery) != 0) {
            result.push_back(&block->attention_.query_);
        }
        if ((targets & kLoraAttentionKey) != 0) {
            result.push_back(&block->attention_.key_);
        }
        if ((targets & kLoraAttentionValue) != 0) {
            result.push_back(&block->attention_.value_);
        }
        if ((targets & kLoraAttentionOutput) != 0) {
            result.push_back(&block->attention_.output_);
        }
        if ((targets & kLoraFeedForwardExpand) != 0) {
            result.push_back(&block->feed_forward_.expand_);
        }
        if ((targets & kLoraFeedForwardProject) != 0) {
            result.push_back(&block->feed_forward_.project_);
        }
    }
    if ((targets & kLoraLanguageModelHead) != 0) {
        result.push_back(&language_model_head_);
    }
    return result;
}

}  // namespace riftco_transformer
