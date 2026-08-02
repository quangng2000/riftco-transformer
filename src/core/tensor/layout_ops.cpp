#include "riftco_transformer/core/tensor_ops.hpp"

#include "core/backend/adapter.hpp"
#include "core/backend/nn/dispatch.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace riftco_transformer::tensor_ops {
namespace {

Tensor concatenate_last_axis_impl(std::span<const Tensor *const> inputs) {
  if (inputs.empty()) {
    throw std::invalid_argument(
        "concatenate_last_axis requires at least one tensor");
  }
  if (inputs.front() == nullptr) {
    throw std::invalid_argument(
        "concatenate_last_axis received an empty tensor reference");
  }
  const Tensor &first = *inputs.front();
  if (first.rank() == 0) {
    throw std::invalid_argument(
        "concatenate_last_axis does not accept scalar tensors");
  }

  Tensor::Shape output_shape = first.shape();
  std::size_t output_width = 0;
  for (const Tensor *input : inputs) {
    if (input == nullptr) {
      throw std::invalid_argument(
          "concatenate_last_axis received an empty tensor reference");
    }
    if (input->rank() != first.rank()) {
      throw std::invalid_argument(
          "concatenated tensors must have the same rank");
    }
    if (input->backend() != first.backend()) {
      throw std::invalid_argument(
          "concatenated tensors must use the same backend");
    }
    for (std::size_t axis = 0; axis + 1 < first.rank(); ++axis) {
      if (input->shape()[axis] != first.shape()[axis]) {
        throw std::invalid_argument(
            "concatenated tensor prefix dimensions must match");
      }
    }
    const std::size_t input_width = input->shape().back();
    if (output_width > std::numeric_limits<std::size_t>::max() - input_width) {
      throw std::overflow_error(
          "concatenated final dimension exceeds addressable size");
    }
    output_width += input_width;
  }

  output_shape.back() = output_width;
  Tensor result(std::move(output_shape), first.backend());
  const std::size_t row_count = first.numel() / first.shape().back();
  for (std::size_t row = 0; row < row_count; ++row) {
    std::size_t output_column = 0;
    for (const Tensor *input : inputs) {
      const std::size_t input_width = input->shape().back();
      const auto source = input->data().subspan(row * input_width, input_width);
      auto destination = result.data().subspan(
          row * output_width + output_column, input_width);
      std::copy(source.begin(), source.end(), destination.begin());
      output_column += input_width;
    }
  }
  return result;
}

} // namespace

Tensor permute(const Tensor& value, Tensor::Shape axes) {
    if (axes.size() != value.rank()) {
        throw std::invalid_argument(
            "permutation must provide one axis per tensor dimension"
        );
    }

    std::vector<bool> seen(value.rank(), false);
    Tensor::Shape output_shape;
    output_shape.reserve(value.rank());
    for (const std::size_t axis : axes) {
        if (axis >= value.rank()) {
            throw std::out_of_range("permutation axis is outside tensor rank");
        }
        if (seen[axis]) {
            throw std::invalid_argument("permutation axes must be unique");
        }
        seen[axis] = true;
        output_shape.push_back(value.shape()[axis]);
    }

    Tensor result(std::move(output_shape), value.backend());
    backend_detail::dispatch_permute(
        value.backend(),
        {
            backend_detail::tensor_storage(value),
            backend_detail::tensor_storage(result),
            value.shape(),
            axes,
        }
    );
    return result;
}

Tensor transpose_2d(const Tensor& value) {
    if (value.rank() != 2) {
        throw std::invalid_argument("transpose_2d requires a rank-2 tensor");
    }
    return permute(value, {1, 0});
}

Tensor concatenate_last_axis(std::span<const Tensor> inputs) {
  std::vector<const Tensor *> input_references;
  input_references.reserve(inputs.size());
  for (const Tensor &input : inputs) {
    input_references.push_back(&input);
  }
  return concatenate_last_axis_impl(input_references);
}

Tensor concatenate_last_axis(const Tensor &left, const Tensor &right) {
  const std::array<const Tensor *, 2> inputs{&left, &right};
  return concatenate_last_axis_impl(inputs);
}

Tensor broadcast_to(const Tensor& value, Tensor::Shape output_shape) {
    if (value.rank() > output_shape.size()) {
        throw std::invalid_argument(
            "broadcast output rank cannot be smaller than input rank"
        );
    }

    const auto rank_offset = output_shape.size() - value.rank();
    for (std::size_t input_dimension = 0; input_dimension < value.rank();
         ++input_dimension) {
        const auto output_dimension = rank_offset + input_dimension;
        if (value.shape()[input_dimension] != 1 &&
            value.shape()[input_dimension] != output_shape[output_dimension]) {
            throw std::invalid_argument(
                "tensor shape is not compatible with broadcast output"
            );
        }
    }

    Tensor result(std::move(output_shape), value.backend());
    backend_detail::dispatch_broadcast(
        value.backend(),
        {
            backend_detail::tensor_storage(value),
            backend_detail::tensor_storage(result),
            value.shape(),
            result.shape(),
        }
    );
    return result;
}

Tensor sum_to_shape(const Tensor& value, Tensor::Shape output_shape) {
    if (output_shape.size() > value.rank()) {
        throw std::invalid_argument(
            "sum-to-shape output rank cannot exceed input rank"
        );
    }

    const auto rank_offset = value.rank() - output_shape.size();
    for (std::size_t output_dimension = 0;
         output_dimension < output_shape.size();
         ++output_dimension) {
        const auto input_dimension = rank_offset + output_dimension;
        if (output_shape[output_dimension] != 1 &&
            output_shape[output_dimension] != value.shape()[input_dimension]) {
            throw std::invalid_argument(
                "sum-to-shape output is not broadcast-compatible"
            );
        }
    }

    Tensor result(std::move(output_shape), value.backend());
    backend_detail::dispatch_sum_to_shape(
        value.backend(),
        {
            backend_detail::tensor_storage(value),
            backend_detail::tensor_storage(result),
            value.shape(),
            result.shape(),
        }
    );
    return result;
}

}  // namespace riftco_transformer::tensor_ops
