#pragma once

#include "riftco_transformer/core/quantized_weight.hpp"
#include "riftco_transformer/nn/low_rank_adapter.hpp"
#include "riftco_transformer/nn/module.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <random>

namespace riftco_transformer {

class DecoderOnlyTransformer;

class Linear : public Module {
public:
  // Constructs a bias-free projection.
  explicit Linear(Tensor weight);
  Linear(Tensor weight, Tensor bias);
  Linear(std::size_t input_width, std::size_t output_width,
         std::mt19937 &random);
  Linear(std::size_t input_width, std::size_t output_width,
         std::mt19937 &random, bool use_bias);

  [[nodiscard]] std::size_t input_width() const noexcept;
  [[nodiscard]] std::size_t output_width() const noexcept;
  [[nodiscard]] bool has_bias() const noexcept;
  [[nodiscard]] Variable forward(const Variable &input) const;
  // Transfers parameters in place. Call before building a forward graph.
  void to(ExecutionBackend backend) override;

  // One-way dense-to-packed conversion. It is rejected while an external
  // ParameterHandle retains the dense weight, and while LoRA is active or
  // has previously been merged. Model-level QLoRA converts before attach.
  void quantize_weight_nf4(
      std::size_t block_size = QuantizedWeight::kDefaultNf4BlockSize);
  void quantize_weight_nf4_double_quantized(
      std::size_t block_size = QuantizedWeight::kDefaultNf4BlockSize,
      std::size_t scale_block_size = 256);
  [[nodiscard]] bool has_quantized_weight() const noexcept;
  [[nodiscard]] const QuantizedWeight &quantized_weight() const;
  [[nodiscard]] QuantizedMemoryUsage quantized_memory_usage() const noexcept;

  // Attaches one adapter exactly once for this Linear object's lifetime.
  void attach_lora(std::size_t rank, float alpha, std::mt19937 &random);
  [[nodiscard]] bool has_lora() const noexcept;
  [[nodiscard]] ParameterList lora_parameters();
  // One-way merge. The inactive adapter storage remains alive so previously
  // obtained native Parameter pointers do not dangle.
  void merge_lora();

  // Throws when the base weight is packed; packed weights are immutable
  // buffers rather than trainable Parameters.
  [[nodiscard]] const Parameter &weight() const;
  // Compatibility accessor. A bias-free Linear returns an inert zero-valued
  // Parameter that is excluded from forward(), parameters(), and optimizer
  // state. Callers must use has_bias() before treating this as trainable.
  [[nodiscard]] const Parameter &bias() const noexcept;
  // Base parameters only, independent of LoRA state.
  [[nodiscard]] ParameterList parameters();

private:
    struct PreparedMaterializedWeight {
        std::optional<Tensor> dense_value;
        std::optional<Parameter> replacement_parameter;
    };

    [[nodiscard]] ParameterList
    extra_parameters_for_transfer() override;
    [[nodiscard]] PreparedMaterializedWeight
    prepare_lora_merge() const;
    void commit_prepared_lora_merge(
        PreparedMaterializedWeight merged_weight
    );
    void discard_unmerged_lora() noexcept;

    [[nodiscard]] QuantizedWeight prepare_weight_quantization_nf4(
        std::size_t block_size,
        std::optional<std::size_t> scale_block_size = std::nullopt
    );
    void commit_prepared_weight_quantization(
        QuantizedWeight quantized_weight
    );
    [[nodiscard]] PreparedMaterializedWeight
    prepare_materialized_weight(
        bool include_lora_delta
    ) const;
    void validate_prepared_materialized_weight(
        const PreparedMaterializedWeight& materialized_weight
    ) const;
    void commit_prepared_materialized_weight(
        PreparedMaterializedWeight materialized_weight,
        bool mark_lora_merged
    );

    std::optional<Parameter> weight_;
    std::optional<QuantizedWeight> quantized_weight_;
    Parameter bias_;
    std::unique_ptr<LowRankAdapter> lora_;
    bool has_bias_ = true;
    bool lora_was_merged_ = false;

    friend class DecoderOnlyTransformer;
};

}  // namespace riftco_transformer
