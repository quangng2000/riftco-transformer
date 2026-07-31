#pragma once

#include "transformer_lab/nn/low_rank_adapter.hpp"
#include "transformer_lab/nn/module.hpp"

#include <cstddef>
#include <memory>
#include <random>

namespace transformer_lab {

class DecoderOnlyTransformer;

class Linear : public Module {
public:
    Linear(Tensor weight, Tensor bias);
    Linear(
        std::size_t input_width,
        std::size_t output_width,
        std::mt19937& random
    );

    [[nodiscard]] std::size_t input_width() const noexcept;
    [[nodiscard]] std::size_t output_width() const noexcept;
    [[nodiscard]] Variable forward(const Variable& input) const;
    // Transfers parameters in place. Call before building a forward graph.
    void to(ExecutionBackend backend);

    // Attaches one adapter exactly once for this Linear object's lifetime.
    void attach_lora(
        std::size_t rank,
        float alpha,
        std::mt19937& random
    );
    [[nodiscard]] bool has_lora() const noexcept;
    [[nodiscard]] ParameterList lora_parameters();
    // One-way merge. The inactive adapter storage remains alive so previously
    // obtained native Parameter pointers do not dangle.
    void merge_lora();

    [[nodiscard]] const Parameter& weight() const noexcept;
    [[nodiscard]] const Parameter& bias() const noexcept;
    // Base parameters only, independent of LoRA state.
    [[nodiscard]] ParameterList parameters();

private:
    [[nodiscard]] ParameterList
    extra_parameters_for_transfer() override;
    [[nodiscard]] Tensor prepare_lora_merge() const;
    void commit_prepared_lora_merge(Tensor merged_weight) noexcept;
    void discard_unmerged_lora() noexcept;

    Parameter weight_;
    Parameter bias_;
    std::unique_ptr<LowRankAdapter> lora_;
    bool lora_was_merged_ = false;

    friend class DecoderOnlyTransformer;
};

}  // namespace transformer_lab
