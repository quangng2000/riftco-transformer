#include "riftco_transformer/core/tensor.hpp"

#include "core/backend/adapter.hpp"
#include "core/backend/nn/dispatch.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace riftco_transformer {

const Tensor::Shape& Tensor::shape() const noexcept {
    return shape_;
}

const Tensor::Shape& Tensor::strides() const noexcept {
    return strides_;
}

std::size_t Tensor::rank() const noexcept {
    return shape_.size();
}

float& Tensor::flat(std::size_t index) {
    if (index >= numel()) {
        throw std::out_of_range("tensor flat index is outside storage");
    }
    return data()[index];
}

const float& Tensor::flat(std::size_t index) const {
    if (index >= numel()) {
        throw std::out_of_range("tensor flat index is outside storage");
    }
    return data()[index];
}

float& Tensor::at(std::initializer_list<std::size_t> indices) {
    if (storage_ == nullptr) {
        throw std::logic_error("cannot index a moved-from tensor");
    }
    return data()[offset({indices.begin(), indices.size()})];
}

const float& Tensor::at(std::initializer_list<std::size_t> indices) const {
    if (storage_ == nullptr) {
        throw std::logic_error("cannot index a moved-from tensor");
    }
    return data()[offset({indices.begin(), indices.size()})];
}

Tensor Tensor::reshape(Shape new_shape) const {
    if (storage_ == nullptr) {
        throw std::logic_error("cannot reshape a moved-from tensor");
    }
    if (checked_numel(new_shape) != numel()) {
        throw std::invalid_argument(
            "tensor value count does not match its shape"
        );
    }
    Tensor result(std::move(new_shape), backend());
    backend_detail::dispatch_copy(
        backend(),
        {
            backend_detail::tensor_storage(*this),
            backend_detail::tensor_storage(result),
            numel(),
        }
    );
    return result;
}

std::size_t Tensor::checked_numel(const Shape& shape) {
    std::size_t count = 1;
    for (const auto dimension : shape) {
        if (dimension == 0) {
            throw std::invalid_argument(
                "tensor dimensions must be greater than zero"
            );
        }
        if (count > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::overflow_error("tensor shape exceeds addressable size");
        }
        count *= dimension;
    }
    return count;
}

Tensor::Shape Tensor::make_strides(const Shape& shape) {
    static_cast<void>(checked_numel(shape));

    Shape strides(shape.size(), 1);
    for (std::size_t index = shape.size(); index > 1; --index) {
        const auto current = index - 2;
        strides[current] = strides[current + 1] * shape[current + 1];
    }
    return strides;
}

std::size_t Tensor::offset(std::span<const std::size_t> indices) const {
    if (indices.size() != rank()) {
        throw std::invalid_argument(
            "tensor index count does not match tensor rank"
        );
    }

    std::size_t flat_index = 0;
    for (std::size_t dimension = 0; dimension < rank(); ++dimension) {
        if (indices[dimension] >= shape_[dimension]) {
            throw std::out_of_range("tensor index is outside its dimension");
        }
        flat_index += indices[dimension] * strides_[dimension];
    }
    return flat_index;
}

}  // namespace riftco_transformer
