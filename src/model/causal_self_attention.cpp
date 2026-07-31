#include "riftco_transformer/model/causal_self_attention.hpp"

#include "core/backend/adapter.hpp"

#include <array>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace riftco_transformer {
namespace {

std::size_t checked_head_count(std::size_t model_width,
                               std::size_t head_count) {
  if (model_width == 0) {
    throw std::invalid_argument(
        "attention model width must be greater than zero");
  }
  if (head_count == 0) {
    throw std::invalid_argument(
        "attention head count must be greater than zero");
  }
  if (model_width % head_count != 0) {
    throw std::invalid_argument(
        "attention model width must be divisible by head count");
  }
  return head_count;
}

FullSequenceAttentionKind checked_attention_kind(
    FullSequenceAttentionKind attention_kind) {
  switch (attention_kind) {
  case FullSequenceAttentionKind::Materialized:
  case FullSequenceAttentionKind::Flash:
    return attention_kind;
  }
  throw std::invalid_argument(
      "full-sequence attention kind is not recognized");
}

} // namespace

namespace model_detail {

class CausalAttentionGraphFactory {
public:
  [[nodiscard]] static Variable context_only(const Variable &queries,
                                             const Variable &keys,
                                             const Variable &values,
                                             FullSequenceAttentionKind kind) {
    if (kind == FullSequenceAttentionKind::Flash) {
      auto state = flash_forward(queries, keys, values);
      return make_flash_context_variable(
          queries, keys, values, std::move(state.context),
          std::move(state.row_maxima), std::move(state.row_exp_sums),
          state.dimensions);
    }
    if (kind != FullSequenceAttentionKind::Materialized) {
      throw std::invalid_argument(
          "full-sequence attention kind is not recognized");
    }
    auto state = materialize_forward(queries, keys, values);
    return make_context_variable(
        queries, keys, values, std::move(state.context),
        std::move(state.probabilities), state.dimensions);
  }

  [[nodiscard]] static CausalAttentionResult
  with_probabilities(const Variable &queries, const Variable &keys,
                     const Variable &values) {
    auto state = materialize_forward(queries, keys, values);
    Variable context =
        make_context_variable(queries, keys, values, std::move(state.context),
                              state.probabilities, state.dimensions);
    Variable probabilities = make_probability_variable(
        queries, keys, state.probabilities, state.dimensions);
    return {
        std::move(context),
        std::move(probabilities),
    };
  }

private:
  struct ForwardState {
    Tensor context;
    std::shared_ptr<Tensor> probabilities;
    backend_detail::MaterializedCausalAttentionDimensions dimensions;
  };

  struct FlashForwardState {
    Tensor context;
    std::shared_ptr<Tensor> row_maxima;
    std::shared_ptr<Tensor> row_exp_sums;
    backend_detail::FlashCausalAttentionDimensions dimensions;
  };

  struct CheckedInputDimensions {
    std::size_t batch;
    std::size_t heads;
    std::size_t time;
    std::size_t head_width;
  };

  [[nodiscard]] static CheckedInputDimensions
  checked_input_dimensions(const Variable &queries, const Variable &keys,
                           const Variable &values) {
    if (queries.value().rank() != 4 || keys.value().rank() != 4 ||
        values.value().rank() != 4) {
      throw std::invalid_argument(
          "causal attention requires rank-four Q, K, and V tensors");
    }
    if (queries.value().shape() != keys.value().shape() ||
        queries.value().shape() != values.value().shape()) {
      throw std::invalid_argument(
          "causal attention requires identical Q, K, and V shapes");
    }
    if (queries.value().backend() != keys.value().backend() ||
        queries.value().backend() != values.value().backend()) {
      throw std::invalid_argument(
          "causal attention tensors must use the same backend");
    }
    return {
        queries.value().shape()[0],
        queries.value().shape()[1],
        queries.value().shape()[2],
        queries.value().shape()[3],
    };
  }

  [[nodiscard]] static ForwardState
  materialize_forward(const Variable &queries, const Variable &keys,
                      const Variable &values) {
    const auto checked =
        checked_input_dimensions(queries, keys, values);
    const auto batch = checked.batch;
    const auto heads = checked.heads;
    const auto time = checked.time;
    const auto head_width = checked.head_width;
    const backend_detail::MaterializedCausalAttentionDimensions dimensions{
        batch,
        heads,
        time,
        head_width,
    };
    Tensor probabilities({batch, heads, time, time}, queries.value().backend());
    Tensor context(queries.value().shape(), queries.value().backend());
    backend_detail::dispatch_materialized_causal_attention_forward(
        queries.value().backend(),
        {
            backend_detail::tensor_storage(queries.value()),
            backend_detail::tensor_storage(keys.value()),
            backend_detail::tensor_storage(values.value()),
            backend_detail::tensor_storage(probabilities),
            backend_detail::tensor_storage(context),
            dimensions,
        });

    return {
        std::move(context),
        std::make_shared<Tensor>(std::move(probabilities)),
        dimensions,
    };
  }

  [[nodiscard]] static FlashForwardState
  flash_forward(const Variable &queries, const Variable &keys,
                const Variable &values) {
    const auto checked =
        checked_input_dimensions(queries, keys, values);
    const backend_detail::FlashCausalAttentionDimensions dimensions{
        checked.batch,
        checked.heads,
        checked.time,
        checked.head_width,
    };
    const Tensor::Shape row_shape{
        checked.batch,
        checked.heads,
        checked.time,
    };
    Tensor row_maxima(row_shape, queries.value().backend());
    Tensor row_exp_sums(row_shape, queries.value().backend());
    Tensor context(queries.value().shape(), queries.value().backend());
    backend_detail::dispatch_flash_causal_attention_forward(
        queries.value().backend(),
        {
            backend_detail::tensor_storage(queries.value()),
            backend_detail::tensor_storage(keys.value()),
            backend_detail::tensor_storage(values.value()),
            backend_detail::tensor_storage(row_maxima),
            backend_detail::tensor_storage(row_exp_sums),
            backend_detail::tensor_storage(context),
            dimensions,
        });

    return {
        std::move(context),
        std::make_shared<Tensor>(std::move(row_maxima)),
        std::make_shared<Tensor>(std::move(row_exp_sums)),
        dimensions,
    };
  }

  [[nodiscard]] static Variable
  make_context_variable(const Variable &queries, const Variable &keys,
                        const Variable &values, Tensor context,
                        std::shared_ptr<Tensor> saved_probabilities,
                        backend_detail::MaterializedCausalAttentionDimensions dimensions) {
    const std::array inputs{queries, keys, values};
    return custom_gradient(
        std::move(context), inputs,
        [queries, keys, values,
         saved_probabilities = std::move(saved_probabilities),
         dimensions](const Tensor &upstream) {
          Tensor query_gradient(queries.value().shape(),
                                queries.value().backend());
          Tensor key_gradient(keys.value().shape(), keys.value().backend());
          Tensor value_gradient(values.value().shape(),
                                values.value().backend());
          backend_detail::dispatch_materialized_causal_attention_context_backward(
              queries.value().backend(),
              {
                  backend_detail::tensor_storage(queries.value()),
                  backend_detail::tensor_storage(keys.value()),
                  backend_detail::tensor_storage(values.value()),
                  backend_detail::tensor_storage(*saved_probabilities),
                  backend_detail::tensor_storage(upstream),
                  backend_detail::tensor_storage(query_gradient),
                  backend_detail::tensor_storage(key_gradient),
                  backend_detail::tensor_storage(value_gradient),
                  dimensions,
              });
          return std::vector<Tensor>{
              std::move(query_gradient),
              std::move(key_gradient),
              std::move(value_gradient),
          };
        });
  }

  [[nodiscard]] static Variable make_flash_context_variable(
      const Variable &queries, const Variable &keys, const Variable &values,
      Tensor context, std::shared_ptr<Tensor> saved_row_maxima,
      std::shared_ptr<Tensor> saved_row_exp_sums,
      backend_detail::FlashCausalAttentionDimensions dimensions) {
    const std::array inputs{queries, keys, values};
    return custom_gradient(
        std::move(context), inputs,
        [queries, keys, values,
         saved_row_maxima = std::move(saved_row_maxima),
         saved_row_exp_sums = std::move(saved_row_exp_sums),
         dimensions](const Tensor &upstream) {
          Tensor query_gradient(queries.value().shape(),
                                queries.value().backend());
          Tensor key_gradient(keys.value().shape(), keys.value().backend());
          Tensor value_gradient(values.value().shape(),
                                values.value().backend());
          backend_detail::dispatch_flash_causal_attention_backward(
              queries.value().backend(),
              {
                  backend_detail::tensor_storage(queries.value()),
                  backend_detail::tensor_storage(keys.value()),
                  backend_detail::tensor_storage(values.value()),
                  backend_detail::tensor_storage(*saved_row_maxima),
                  backend_detail::tensor_storage(*saved_row_exp_sums),
                  backend_detail::tensor_storage(upstream),
                  backend_detail::tensor_storage(query_gradient),
                  backend_detail::tensor_storage(key_gradient),
                  backend_detail::tensor_storage(value_gradient),
                  dimensions,
              });
          return std::vector<Tensor>{
              std::move(query_gradient),
              std::move(key_gradient),
              std::move(value_gradient),
          };
        });
  }

  [[nodiscard]] static Variable make_probability_variable(
      const Variable &queries, const Variable &keys,
      const std::shared_ptr<Tensor> &saved_probabilities,
      backend_detail::MaterializedCausalAttentionDimensions dimensions) {
    const std::array inputs{queries, keys};
    return custom_gradient(
        Tensor(*saved_probabilities), inputs,
        [queries, keys, saved_probabilities,
         dimensions](const Tensor &upstream) {
          Tensor query_gradient(queries.value().shape(),
                                queries.value().backend());
          Tensor key_gradient(keys.value().shape(), keys.value().backend());
          backend_detail::dispatch_materialized_causal_attention_probabilities_backward(
              queries.value().backend(),
              {
                  backend_detail::tensor_storage(queries.value()),
                  backend_detail::tensor_storage(keys.value()),
                  backend_detail::tensor_storage(*saved_probabilities),
                  backend_detail::tensor_storage(upstream),
                  backend_detail::tensor_storage(query_gradient),
                  backend_detail::tensor_storage(key_gradient),
                  dimensions,
              });
          return std::vector<Tensor>{
              std::move(query_gradient),
              std::move(key_gradient),
          };
        });
  }
};

} // namespace model_detail

Variable split_attention_heads(const Variable &input, std::size_t head_count) {
  if (input.value().rank() != 3) {
    throw std::invalid_argument(
        "split_attention_heads requires [batch, time, model_width]");
  }
  if (head_count == 0) {
    throw std::invalid_argument(
        "attention head count must be greater than zero");
  }

  const auto batch = input.value().shape()[0];
  const auto time = input.value().shape()[1];
  const auto model_width = input.value().shape()[2];
  if (model_width % head_count != 0) {
    throw std::invalid_argument(
        "model width must be divisible by attention head count");
  }
  const auto head_width = model_width / head_count;

  return permute(reshape(input, {batch, time, head_count, head_width}),
                 {0, 2, 1, 3});
}

Variable merge_attention_heads(const Variable &input) {
  if (input.value().rank() != 4) {
    throw std::invalid_argument(
        "merge_attention_heads requires [batch, head, time, head_width]");
  }

  const auto batch = input.value().shape()[0];
  const auto head_count = input.value().shape()[1];
  const auto time = input.value().shape()[2];
  const auto head_width = input.value().shape()[3];
  if (head_count > std::numeric_limits<std::size_t>::max() / head_width) {
    throw std::overflow_error("merged attention model width overflows");
  }

  return reshape(permute(input, {0, 2, 1, 3}),
                 {batch, time, head_count * head_width});
}

CausalAttentionResult causal_scaled_dot_product_attention(
    const Variable &queries, const Variable &keys, const Variable &values) {
  return model_detail::CausalAttentionGraphFactory::with_probabilities(
      queries, keys, values);
}

CausalSelfAttention::CausalSelfAttention(std::size_t model_width,
                                         std::size_t head_count,
                                         std::mt19937 &random,
                                         FullSequenceAttentionKind attention_kind)
    : head_count_(checked_head_count(model_width, head_count)),
      attention_kind_(checked_attention_kind(attention_kind)),
      query_(model_width, model_width, random),
      key_(model_width, model_width, random),
      value_(model_width, model_width, random),
      output_(model_width, model_width, random) {
  register_module("query", query_);
  register_module("key", key_);
  register_module("value", value_);
  register_module("output", output_);
}

std::size_t CausalSelfAttention::model_width() const noexcept {
  return query_.input_width();
}

std::size_t CausalSelfAttention::head_count() const noexcept {
  return head_count_;
}

std::size_t CausalSelfAttention::head_width() const noexcept {
  return model_width() / head_count();
}

FullSequenceAttentionKind
CausalSelfAttention::full_sequence_attention_kind() const noexcept {
  return attention_kind_;
}

void CausalSelfAttention::set_full_sequence_attention_kind(
    FullSequenceAttentionKind attention_kind) {
  attention_kind_ = checked_attention_kind(attention_kind);
}

Variable CausalSelfAttention::forward(const Variable &input) const {
  return forward(input, attention_kind_);
}

Variable CausalSelfAttention::forward(
    const Variable &input,
    FullSequenceAttentionKind attention_kind) const {
  if (input.value().rank() != 3 || input.value().shape()[2] != model_width()) {
    throw std::invalid_argument("causal self-attention input must have shape "
                                "[batch, time, model_width]");
  }
  const auto checked_kind = checked_attention_kind(attention_kind);

  const Variable queries =
      split_attention_heads(query_.forward(input), head_count());
  const Variable keys =
      split_attention_heads(key_.forward(input), head_count());
  const Variable values =
      split_attention_heads(value_.forward(input), head_count());
  const Variable context =
      model_detail::CausalAttentionGraphFactory::context_only(queries, keys,
                                                              values,
                                                              checked_kind);
  return output_.forward(merge_attention_heads(context));
}

void CausalSelfAttention::to(ExecutionBackend backend) {
  Module::to(backend);
}

ParameterList CausalSelfAttention::parameters() {
  return Module::parameters();
}

ParameterList CausalSelfAttention::lora_parameters() {
  ParameterList result;
  append_parameter_group(result, "query", query_.lora_parameters());
  append_parameter_group(result, "key", key_.lora_parameters());
  append_parameter_group(result, "value", value_.lora_parameters());
  append_parameter_group(result, "output", output_.lora_parameters());
  return result;
}

} // namespace riftco_transformer
