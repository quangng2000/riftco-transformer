#include "riftco_transformer/nn/linear.hpp"

#include "riftco_transformer/core/tensor_ops.hpp"
#include "riftco_transformer/nn/initialization.hpp"
#include "riftco_transformer/nn/quantized_linear.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <utility>

namespace riftco_transformer {
namespace {

Tensor checked_linear_weight(Tensor weight) {
    if (weight.rank() != 2) {
        throw std::invalid_argument(
            "linear weight must have shape [output, input]"
        );
    }
    return weight;
}

Tensor checked_linear_bias(
    Tensor bias,
    const Tensor& weight
) {
    if (bias.shape() != Tensor::Shape({weight.shape()[0]})) {
        throw std::invalid_argument(
            "linear bias must have shape [output]"
        );
    }
    if (bias.backend() != weight.backend()) {
        throw std::invalid_argument(
            "linear weight and bias must use the same backend"
        );
    }
    return bias;
}

Tensor initialized_linear_weight(
    std::size_t input_width,
    std::size_t output_width,
    std::mt19937& random
) {
    if (input_width == 0 || output_width == 0) {
        throw std::invalid_argument(
            "linear dimensions must be greater than zero"
        );
    }
    return xavier_uniform(
        {output_width, input_width},
        input_width,
        output_width,
        random
    );
}

}  // namespace

Linear::Linear(Tensor weight)
    : weight_(std::in_place, checked_linear_weight(std::move(weight))),
      bias_(Tensor::zeros({weight_->value().shape()[0]},
                          weight_->value().backend())),
      has_bias_(false) {
  register_parameter("weight", *weight_);
}

Linear::Linear(Tensor weight, Tensor bias)
    : weight_(
          std::in_place,
          checked_linear_weight(std::move(weight))
      ),
      bias_(checked_linear_bias(
          std::move(bias),
          weight_->value()
      )) {
    register_parameter("weight", *weight_);
    register_parameter("bias", bias_);
}

Linear::Linear(std::size_t input_width, std::size_t output_width,
               std::mt19937 &random)
    : Linear(input_width, output_width, random, true) {}

Linear::Linear(std::size_t input_width, std::size_t output_width,
               std::mt19937 &random, bool use_bias)
    : weight_(std::in_place,
              initialized_linear_weight(input_width, output_width, random)),
      bias_(Tensor::zeros({output_width}, weight_->value().backend())),
      has_bias_(use_bias) {
  register_parameter("weight", *weight_);
  if (has_bias()) {
    register_parameter("bias", bias_);
  }
}

std::size_t Linear::input_width() const noexcept {
    return has_quantized_weight()
               ? quantized_weight_->shape()[1]
               : weight_->value().shape()[1];
}

std::size_t Linear::output_width() const noexcept {
    return has_quantized_weight()
               ? quantized_weight_->shape()[0]
               : weight_->value().shape()[0];
}

bool Linear::has_bias() const noexcept { return has_bias_; }

Variable Linear::forward(const Variable& input) const {
    if (input.value().rank() == 0 ||
        input.value().shape().back() != input_width()) {
        throw std::invalid_argument(
            "linear input final dimension must equal input width"
        );
    }

    Tensor::Shape output_shape = input.value().shape();
    output_shape.back() = output_width();
    Variable projected = [&] {
        if (has_quantized_weight()) {
            return quantized_linear(input, *quantized_weight_);
        }
        const auto rows = input.value().numel() / input_width();
        const Variable flattened = reshape(
            input,
            {rows, input_width()}
        );
        return reshape(
            matmul(
                flattened,
                transpose_2d(weight_->variable())
            ),
            output_shape
        );
    }();
    const Variable base_output =
        has_bias() ? projected +
                         broadcast_to(bias_.variable(), std::move(output_shape))
                   : projected;
    if (!has_lora()) {
        return base_output;
    }
    return base_output + lora_->forward(input);
}

void Linear::to(ExecutionBackend backend) {
    std::optional<QuantizedWeight> transferred_weight;
    if (has_quantized_weight() &&
        quantized_weight_->backend() != backend) {
        transferred_weight.emplace(
            quantized_weight_->to(backend)
        );
    }
    Module::to(backend);
    if (transferred_weight.has_value()) {
        quantized_weight_ = std::move(*transferred_weight);
    }
}

void Linear::quantize_weight_nf4(std::size_t block_size) {
    QuantizedWeight candidate =
        prepare_weight_quantization_nf4(block_size);
    commit_prepared_weight_quantization(std::move(candidate));
}

void Linear::quantize_weight_nf4_double_quantized(
    std::size_t block_size,
    std::size_t scale_block_size
) {
    QuantizedWeight candidate = prepare_weight_quantization_nf4(
        block_size,
        scale_block_size
    );
    commit_prepared_weight_quantization(std::move(candidate));
}

bool Linear::has_quantized_weight() const noexcept {
    return quantized_weight_.has_value();
}

const QuantizedWeight& Linear::quantized_weight() const {
    if (!has_quantized_weight()) {
        throw std::logic_error(
            "linear module does not have a quantized weight"
        );
    }
    return *quantized_weight_;
}

QuantizedMemoryUsage Linear::quantized_memory_usage() const noexcept {
    return has_quantized_weight()
               ? quantized_weight_->memory_usage()
               : QuantizedMemoryUsage{};
}

void Linear::attach_lora(
    std::size_t rank,
    float alpha,
    std::mt19937& random
) {
    if (lora_ != nullptr || lora_was_merged_) {
        throw std::logic_error(
            "LoRA can only be attached once to a Linear module"
        );
    }

    auto candidate = std::make_unique<LowRankAdapter>(
        input_width(),
        output_width(),
        rank,
        alpha,
        random,
        has_quantized_weight()
            ? quantized_weight_->backend()
            : weight_->value().backend()
    );
    lora_ = std::move(candidate);
}

bool Linear::has_lora() const noexcept {
    return lora_ != nullptr && !lora_was_merged_;
}

ParameterList Linear::lora_parameters() {
    if (!has_lora()) {
        return {};
    }
    return lora_->parameters();
}

void Linear::merge_lora() {
    PreparedMaterializedWeight merged_weight = prepare_lora_merge();
    validate_prepared_materialized_weight(merged_weight);
    commit_prepared_lora_merge(std::move(merged_weight));
}

const Parameter& Linear::weight() const {
    if (!weight_.has_value()) {
        throw std::logic_error(
            "quantized Linear has no dense weight Parameter"
        );
    }
    return *weight_;
}

const Parameter& Linear::bias() const noexcept {
    return bias_;
}

ParameterList Linear::parameters() {
    return Module::parameters();
}

ParameterList Linear::extra_parameters_for_transfer() {
  ParameterList result;
  if (!has_bias()) {
    result.push_back({"bias", bias_.handle()});
  }
  if (lora_ != nullptr) {
    ParameterList adapter_parameters = lora_->parameters();
    result.insert(result.end(),
                  std::make_move_iterator(adapter_parameters.begin()),
                  std::make_move_iterator(adapter_parameters.end()));
  }
  return result;
}

Linear::PreparedMaterializedWeight
Linear::prepare_lora_merge() const {
    if (!has_lora()) {
        if (lora_was_merged_) {
            throw std::logic_error(
                "LoRA has already been merged into this Linear module"
            );
        }
        throw std::logic_error(
            "cannot merge LoRA before attaching an adapter"
        );
    }

    return prepare_materialized_weight(true);
}

void Linear::commit_prepared_lora_merge(
    PreparedMaterializedWeight merged_weight
) {
    commit_prepared_materialized_weight(
        std::move(merged_weight),
        true
    );
}

void Linear::discard_unmerged_lora() noexcept {
    if (!lora_was_merged_) {
        lora_.reset();
    }
}

QuantizedWeight Linear::prepare_weight_quantization_nf4(
    std::size_t block_size,
    std::optional<std::size_t> scale_block_size
) {
    if (has_quantized_weight()) {
        throw std::logic_error(
            "Linear weight has already been quantized"
        );
    }
    if (lora_ != nullptr || lora_was_merged_) {
        throw std::logic_error(
            "Linear weight must be quantized before attaching LoRA"
        );
    }
    if (!weight_.has_value()) {
        throw std::logic_error(
            "Linear has neither a dense nor quantized weight"
        );
    }
    preflight_parameter_deactivation("weight", *weight_);
    return scale_block_size.has_value()
               ? QuantizedWeight::quantize_nf4_double_quantized(
                     weight_->value(),
                     block_size,
                     *scale_block_size
                 )
               : QuantizedWeight::quantize_nf4(
                     weight_->value(),
                     block_size
                 );
}

void Linear::commit_prepared_weight_quantization(
    QuantizedWeight quantized_weight
) {
    if (has_quantized_weight() || !weight_.has_value()) {
        throw std::logic_error(
            "Linear dense weight is unavailable for quantization"
        );
    }
    if (quantized_weight.shape() != weight_->value().shape() ||
        quantized_weight.backend() != weight_->value().backend()) {
        throw std::invalid_argument(
            "prepared quantized weight does not match the dense weight"
        );
    }

    // Moving QuantizedWeight into optional storage is non-throwing. Drop the
    // registration handle first, install the packed state, then release the
    // owning dense wrapper so no FP32 base allocation remains.
    preflight_parameter_deactivation("weight", *weight_);
    deactivate_parameter("weight", *weight_);
    quantized_weight_.emplace(std::move(quantized_weight));
    weight_.reset();
}

Linear::PreparedMaterializedWeight
Linear::prepare_materialized_weight(
    bool include_lora_delta
) const {
    Tensor candidate = has_quantized_weight()
                           ? quantized_weight_->dequantize()
                           : Tensor(weight_->value());
    if (include_lora_delta) {
        if (!has_lora()) {
            throw std::logic_error(
                "cannot materialize a LoRA delta without an active adapter"
            );
        }
        candidate = tensor_ops::add(
            candidate,
            lora_->weight_delta()
        );
    }
    if (!std::all_of(
            candidate.data().begin(),
            candidate.data().end(),
            [](float value) {
                return std::isfinite(value);
            }
        )) {
        throw std::domain_error(
            "materialized Linear weight must contain only finite values"
        );
    }
    PreparedMaterializedWeight prepared;
    if (!has_quantized_weight()) {
        prepared.dense_value.emplace(std::move(candidate));
        return prepared;
    }

    // Parameter construction allocates the restored FP32 gradient and the
    // first handle allocates its canonical proxy. Finish both allocations in
    // the prepare phase so a model-wide commit cannot fail midway for OOM.
    prepared.replacement_parameter.emplace(std::move(candidate));
    {
        const ParameterHandle prepared_handle =
            prepared.replacement_parameter->handle();
        static_cast<void>(prepared_handle);
    }
    return prepared;
}

void Linear::validate_prepared_materialized_weight(
    const PreparedMaterializedWeight& materialized_weight
) const {
    const bool has_dense_candidate =
        materialized_weight.dense_value.has_value();
    const bool has_replacement =
        materialized_weight.replacement_parameter.has_value();
    if (has_dense_candidate == has_replacement ||
        has_replacement != has_quantized_weight()) {
        throw std::logic_error(
            "prepared Linear materialization does not match its base state"
        );
    }
    const Tensor& value = has_replacement
                              ? materialized_weight
                                    .replacement_parameter->value()
                              : *materialized_weight.dense_value;
    if (value.shape() !=
        Tensor::Shape({output_width(), input_width()})) {
        throw std::invalid_argument(
            "materialized Linear weight has the wrong shape"
        );
    }
    const ExecutionBackend expected_backend = has_quantized_weight()
                                                  ? quantized_weight_->backend()
                                                  : weight_->value().backend();
    if (value.backend() != expected_backend) {
        throw std::invalid_argument(
            "materialized Linear weight has the wrong backend"
        );
    }
}

void Linear::commit_prepared_materialized_weight(
    PreparedMaterializedWeight materialized_weight,
    bool mark_lora_merged
) {
    if (!has_quantized_weight()) {
        weight_->set_value(
            std::move(*materialized_weight.dense_value)
        );
    } else {
        weight_.emplace(std::move(
            *materialized_weight.replacement_parameter
        ));
        reactivate_parameter("weight", *weight_);
        quantized_weight_.reset();
    }
    if (mark_lora_merged) {
        lora_was_merged_ = true;
    }
}

}  // namespace riftco_transformer
