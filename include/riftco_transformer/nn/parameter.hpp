#pragma once

#include "riftco_transformer/core/autograd.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace riftco_transformer {

class Parameter;
class ParameterState;

// An owning, pointer-compatible reference to a parameter's canonical proxy.
// Copies retain the underlying value/gradient state independently of the
// wrapper or module that originally registered the parameter. All live handles
// for one state expose the same Parameter address, preserving identity-based
// duplicate detection.
class ParameterHandle {
public:
    ParameterHandle() noexcept = default;
    ParameterHandle(std::nullptr_t) noexcept;
    ParameterHandle(Parameter* parameter);

    ParameterHandle(const ParameterHandle&) noexcept = default;
    ParameterHandle& operator=(const ParameterHandle&) noexcept = default;
    ParameterHandle(ParameterHandle&& other) noexcept;
    ParameterHandle& operator=(ParameterHandle&& other) noexcept;

    [[nodiscard]] Parameter* get() const noexcept;
    [[nodiscard]] Parameter* operator->() const noexcept;
    [[nodiscard]] Parameter& operator*() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

    // Retains source compatibility with code that stores or deduplicates the
    // public NamedParameter field as a raw Parameter pointer.
    [[nodiscard]] operator Parameter*() const noexcept;

private:
    std::shared_ptr<ParameterState> owner_;
    Parameter* parameter_ = nullptr;
};

struct NamedParameter;
using ParameterList = std::vector<NamedParameter>;

// Transactionally transfers parameter values in place. Parameters that change
// backend receive fresh zero gradients there; parameters already on the
// destination are unchanged. Call this before constructing a forward graph or
// backend-specific optimizer state.
void move_parameters_to(
    const ParameterList& parameters,
    ExecutionBackend backend
);

class Parameter {
public:
    explicit Parameter(Tensor initial_value);

    Parameter(const Parameter&) = delete;
    Parameter& operator=(const Parameter&) = delete;
    // Moving a wrapper never relocates its canonical registered identity.
    Parameter(Parameter&& other) noexcept;
    Parameter& operator=(Parameter&& other);

    [[nodiscard]] const Variable& variable() const noexcept;
    [[nodiscard]] const Tensor& value() const noexcept;
    [[nodiscard]] const Tensor& gradient() const noexcept;
    [[nodiscard]] bool has_pending_gradient() const noexcept;
    [[nodiscard]] ParameterHandle handle();

    void zero_gradient() const;
    void set_value(Tensor new_value);
    void to(ExecutionBackend backend);

private:
    struct CanonicalProxyTag {};

    Parameter(
        ParameterState* state,
        CanonicalProxyTag
    ) noexcept;

    [[nodiscard]] std::shared_ptr<ParameterState>
    shared_state() const noexcept;

    void replace_state(
        Tensor new_value,
        Tensor new_gradient
    ) noexcept;
    void mark_registered() noexcept;

    std::shared_ptr<ParameterState> state_owner_;
    ParameterState* state_;
    bool registered_wrapper_ = false;

    friend class ParameterHandle;
    friend class ParameterState;
    friend class Module;
    friend class Adam;

    friend void move_parameters_to(
        const ParameterList&,
        ExecutionBackend
    );
};

struct NamedParameter {
    std::string name;

private:
    ParameterHandle owner_;

public:
    // Compatibility view. The private handle owns this canonical proxy, so
    // the pointer remains valid for the lifetime of any NamedParameter copy.
    // The pointer itself is read-only so it cannot diverge from owner_.
    Parameter* const parameter;

    NamedParameter() noexcept;
    NamedParameter(std::string name, std::nullptr_t) noexcept;
    NamedParameter(std::string name, Parameter* parameter);
    NamedParameter(std::string name, ParameterHandle parameter) noexcept;

    NamedParameter(const NamedParameter& other);
    NamedParameter& operator=(const NamedParameter& other);
    NamedParameter(NamedParameter&& other) noexcept;
    NamedParameter& operator=(NamedParameter&& other) noexcept;

    [[nodiscard]] const ParameterHandle& handle() const noexcept;
};

void append_parameter_group(
    ParameterList& destination,
    std::string_view prefix,
    ParameterList group
);

[[nodiscard]] std::size_t parameter_count(
    const ParameterList& parameters
);

}  // namespace riftco_transformer
