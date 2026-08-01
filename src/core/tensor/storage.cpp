#include "riftco_transformer/core/tensor.hpp"

#include "core/backend/adapter.hpp"
#include "core/backend/nn/dispatch.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

namespace riftco_transformer {

Tensor::Tensor(Shape shape)
    : Tensor(std::move(shape), 0.0F, execution_backend()) {}

Tensor::Tensor(Shape shape, ExecutionBackend backend)
    : Tensor(std::move(shape), 0.0F, backend) {}

Tensor::Tensor(Shape shape, float fill_value)
    : Tensor(std::move(shape), fill_value, execution_backend()) {}

Tensor::Tensor(Shape shape, float fill_value, ExecutionBackend backend)
    : shape_(std::move(shape)),
      strides_(make_strides(shape_)),
      storage_(
          backend_detail::make_tensor_storage(
              backend,
              checked_numel(shape_),
              fill_value
          )
      ) {}

Tensor::Tensor(Shape shape, std::vector<float> values)
    : Tensor(std::move(shape), std::move(values), execution_backend()) {}

Tensor::Tensor(Shape shape, std::vector<float> values, ExecutionBackend backend)
    : shape_(std::move(shape)),
      strides_(make_strides(shape_)),
      storage_(nullptr) {
    const auto expected_values = checked_numel(shape_);
    if (values.size() != expected_values) {
        throw std::invalid_argument(
            "tensor value count does not match its shape"
        );
    }
    storage_ = backend_detail::make_tensor_storage(backend, std::move(values));
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_),
      strides_(other.strides_),
      storage_(nullptr) {
    if (other.storage_ == nullptr) {
        return;
    }
    storage_ = backend_detail::make_tensor_storage(
        other.backend(),
        other.numel(),
        0.0F
    );
    backend_detail::dispatch_copy(
        other.backend(),
        {
            *other.storage_,
            *storage_,
            other.numel(),
        }
    );
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) {
        return *this;
    }
    Tensor replacement(other);
    shape_.swap(replacement.shape_);
    strides_.swap(replacement.strides_);
    storage_.swap(replacement.storage_);
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)),
      storage_(std::move(other.storage_)) {
    other.shape_.clear();
    other.strides_.clear();
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    shape_ = std::move(other.shape_);
    strides_ = std::move(other.strides_);
    storage_ = std::move(other.storage_);
    other.shape_.clear();
    other.strides_.clear();
    return *this;
}

Tensor::~Tensor() = default;

Tensor Tensor::zeros(Shape shape) {
    return Tensor(std::move(shape), 0.0F, execution_backend());
}

Tensor Tensor::zeros(Shape shape, ExecutionBackend backend) {
    return Tensor(std::move(shape), 0.0F, backend);
}

Tensor Tensor::full(Shape shape, float value) {
    return Tensor(std::move(shape), value, execution_backend());
}

Tensor Tensor::full(Shape shape, float value, ExecutionBackend backend) {
    return Tensor(std::move(shape), value, backend);
}

std::size_t Tensor::numel() const noexcept {
    return storage_ == nullptr ? 0 : storage_->size();
}

ExecutionBackend Tensor::backend() const noexcept {
    return storage_ == nullptr ? ExecutionBackend::Cpu : storage_->backend();
}

std::span<float> Tensor::data() noexcept {
    return storage_ == nullptr ? std::span<float>{} : storage_->data();
}

std::span<const float> Tensor::data() const noexcept {
    return storage_ == nullptr ? std::span<const float>{} : storage_->data();
}

Tensor Tensor::to(ExecutionBackend target_backend) const {
    if (storage_ == nullptr) {
        throw std::logic_error("cannot transfer a moved-from tensor");
    }
    if (target_backend == backend()) {
        return Tensor(*this);
    }
    return Tensor(
        shape_,
        std::vector<float>(data().begin(), data().end()),
        target_backend
    );
}

namespace backend_detail {

TensorStorage& tensor_storage(Tensor& tensor) noexcept {
    return *tensor.storage_;
}

const TensorStorage& tensor_storage(const Tensor& tensor) noexcept {
    return *tensor.storage_;
}

}  // namespace backend_detail

}  // namespace riftco_transformer
