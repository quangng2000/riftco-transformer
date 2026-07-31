#pragma once

#include "transformer_lab/core/backend.hpp"

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <span>
#include <vector>

namespace transformer_lab {

class Tensor;

namespace backend_detail {
class TensorStorage;
[[nodiscard]] TensorStorage& tensor_storage(Tensor&) noexcept;
[[nodiscard]] const TensorStorage& tensor_storage(
    const Tensor&
) noexcept;
}  // namespace backend_detail

class Tensor {
public:
    using Shape = std::vector<std::size_t>;

    // An empty shape represents a scalar and therefore contains one value.
    // Constructors without an explicit backend capture the calling thread's
    // current construction default.
    explicit Tensor(Shape shape);
    Tensor(Shape shape, ExecutionBackend backend);
    Tensor(Shape shape, float fill_value);
    Tensor(
        Shape shape,
        float fill_value,
        ExecutionBackend backend
    );
    Tensor(Shape shape, std::vector<float> values);
    Tensor(
        Shape shape,
        std::vector<float> values,
        ExecutionBackend backend
    );

    // Copying owns an independent backend allocation; moving transfers it.
    // A moved-from tensor is a safe singular value with numel()==0 and empty
    // data(); it may be destroyed or assigned before reuse.
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor();

    [[nodiscard]] static Tensor zeros(Shape shape);
    [[nodiscard]] static Tensor zeros(
        Shape shape,
        ExecutionBackend backend
    );
    [[nodiscard]] static Tensor full(Shape shape, float value);
    [[nodiscard]] static Tensor full(
        Shape shape,
        float value,
        ExecutionBackend backend
    );

    [[nodiscard]] const Shape& shape() const noexcept;
    [[nodiscard]] const Shape& strides() const noexcept;
    [[nodiscard]] std::size_t rank() const noexcept;
    [[nodiscard]] std::size_t numel() const noexcept;
    [[nodiscard]] ExecutionBackend backend() const noexcept;

    [[nodiscard]] std::span<float> data() noexcept;
    [[nodiscard]] std::span<const float> data() const noexcept;

    float& flat(std::size_t index);
    const float& flat(std::size_t index) const;

    float& at(std::initializer_list<std::size_t> indices);
    const float& at(std::initializer_list<std::size_t> indices) const;

    [[nodiscard]] Tensor reshape(Shape new_shape) const;
    // Returns an independent deep copy on the requested backend.
    [[nodiscard]] Tensor to(ExecutionBackend backend) const;

private:
    Shape shape_;
    Shape strides_;
    std::unique_ptr<backend_detail::TensorStorage> storage_;

    [[nodiscard]] static std::size_t checked_numel(const Shape& shape);
    [[nodiscard]] static Shape make_strides(const Shape& shape);
    [[nodiscard]] std::size_t offset(
        std::span<const std::size_t> indices
    ) const;

    friend backend_detail::TensorStorage&
    backend_detail::tensor_storage(Tensor&) noexcept;
    friend const backend_detail::TensorStorage&
    backend_detail::tensor_storage(const Tensor&) noexcept;
};

}  // namespace transformer_lab
