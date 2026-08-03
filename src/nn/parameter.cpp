#include "riftco_transformer/nn/parameter.hpp"

#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace riftco_transformer {

class ParameterState final
    : public std::enable_shared_from_this<ParameterState> {
public:
    explicit ParameterState(Tensor initial_value)
        : variable(std::move(initial_value), true) {}

    [[nodiscard]] Parameter* canonical_parameter() {
        std::lock_guard lock(canonical_mutex);
        if (canonical == nullptr) {
            canonical = std::unique_ptr<Parameter>(
                new Parameter(
                    this,
                    Parameter::CanonicalProxyTag{}
                )
            );
        }
        return canonical.get();
    }

    Variable variable;

private:
    std::mutex canonical_mutex;
    std::unique_ptr<Parameter> canonical;
};

ParameterHandle::ParameterHandle(std::nullptr_t) noexcept {}

ParameterHandle::ParameterHandle(Parameter* parameter) {
    if (parameter == nullptr) {
        return;
    }
    owner_ = parameter->shared_state();
    parameter_ = owner_->canonical_parameter();
}

ParameterHandle::ParameterHandle(ParameterHandle&& other) noexcept
    : owner_(std::move(other.owner_)),
      parameter_(other.parameter_) {
    other.parameter_ = nullptr;
}

ParameterHandle& ParameterHandle::operator=(
    ParameterHandle&& other
) noexcept {
    if (this == &other) {
        return *this;
    }
    owner_ = std::move(other.owner_);
    parameter_ = other.parameter_;
    other.parameter_ = nullptr;
    return *this;
}

Parameter* ParameterHandle::get() const noexcept {
    return parameter_;
}

Parameter* ParameterHandle::operator->() const noexcept {
    return get();
}

Parameter& ParameterHandle::operator*() const noexcept {
    return *get();
}

ParameterHandle::operator bool() const noexcept {
    return get() != nullptr;
}

ParameterHandle::operator Parameter*() const noexcept {
    return get();
}

NamedParameter::NamedParameter() noexcept
    : parameter(nullptr) {}

NamedParameter::NamedParameter(
    std::string parameter_name,
    std::nullptr_t
) noexcept
    : name(std::move(parameter_name)),
      parameter(nullptr) {}

NamedParameter::NamedParameter(
    std::string parameter_name,
    Parameter* parameter_value
)
    : name(std::move(parameter_name)),
      owner_(parameter_value),
      parameter(owner_.get()) {}

NamedParameter::NamedParameter(
    std::string parameter_name,
    ParameterHandle parameter_value
) noexcept
    : name(std::move(parameter_name)),
      owner_(std::move(parameter_value)),
      parameter(owner_.get()) {}

NamedParameter::NamedParameter(const NamedParameter& other)
    : name(other.name),
      owner_(other.owner_),
      parameter(owner_.get()) {}

NamedParameter& NamedParameter::operator=(
    const NamedParameter& other
) {
    if (this == &other) {
        return *this;
    }
    NamedParameter replacement(other);
    std::destroy_at(this);
    std::construct_at(this, std::move(replacement));
    return *this;
}

NamedParameter::NamedParameter(NamedParameter&& other) noexcept
    : name(std::move(other.name)),
      owner_(other.owner_),
      parameter(owner_.get()) {}

NamedParameter& NamedParameter::operator=(
    NamedParameter&& other
) noexcept {
    if (this == &other) {
        return *this;
    }
    NamedParameter replacement(std::move(other));
    std::destroy_at(this);
    std::construct_at(this, std::move(replacement));
    return *this;
}

const ParameterHandle& NamedParameter::handle() const noexcept {
    return owner_;
}

Parameter::Parameter(Tensor initial_value)
    : state_owner_(
          std::make_shared<ParameterState>(std::move(initial_value))
      ),
      state_(state_owner_.get()) {}

Parameter::Parameter(
    ParameterState* state,
    CanonicalProxyTag
) noexcept
    : state_(state) {}

Parameter::Parameter(Parameter&& other) noexcept
    : state_owner_(other.shared_state()),
      state_(state_owner_.get()) {}

Parameter& Parameter::operator=(Parameter&& other) {
    if (this == &other) {
        return *this;
    }
    if (registered_wrapper_) {
        throw std::logic_error(
            "cannot rebind a registered parameter wrapper"
        );
    }
    if (state_owner_ == nullptr) {
        throw std::logic_error(
            "cannot rebind a canonical parameter handle"
        );
    }
    state_owner_ = other.shared_state();
    state_ = state_owner_.get();
    return *this;
}

std::shared_ptr<ParameterState>
Parameter::shared_state() const noexcept {
    if (state_owner_ != nullptr) {
        return state_owner_;
    }
    return state_->weak_from_this().lock();
}

bool Parameter::has_pending_gradient() const noexcept {
    return variable().has_pending_gradient();
}

const Variable& Parameter::variable() const noexcept {
    return state_->variable;
}

const Tensor& Parameter::value() const noexcept {
    return variable().value();
}

const Tensor& Parameter::gradient() const noexcept {
    return variable().gradient();
}

ParameterHandle Parameter::handle() {
    return ParameterHandle(this);
}

void Parameter::zero_gradient() const {
    variable().zero_gradient();
}

void Parameter::set_value(Tensor new_value) {
    state_->variable.replace_leaf_value(std::move(new_value));
}

void Parameter::to(ExecutionBackend backend) {
    if (value().backend() == backend) {
        return;
    }
    Tensor next_value = value().to(backend);
    Tensor next_gradient = Tensor::zeros(
        next_value.shape(),
        backend
    );
    replace_state(
        std::move(next_value),
        std::move(next_gradient)
    );
}

void Parameter::replace_state(
    Tensor new_value,
    Tensor new_gradient
) noexcept {
    state_->variable.replace_leaf_state(
        std::move(new_value),
        std::move(new_gradient)
    );
}

void Parameter::mark_registered() noexcept {
    registered_wrapper_ = true;
}

void append_parameter_group(
    ParameterList& destination,
    std::string_view prefix,
    ParameterList group
) {
    for (auto& named_parameter : group) {
        const std::string qualified_name =
            prefix.empty()
                ? std::move(named_parameter.name)
                : std::string(prefix) + "." + named_parameter.name;
        named_parameter.name = qualified_name;
        destination.push_back(std::move(named_parameter));
    }
}

std::size_t parameter_count(const ParameterList& parameters) {
    std::size_t total = 0;
    for (const auto& named_parameter : parameters) {
        if (named_parameter.parameter == nullptr) {
            throw std::invalid_argument(
                "parameter list contains a null parameter"
            );
        }
        const auto count = named_parameter.parameter->value().numel();
        if (total >
            std::numeric_limits<std::size_t>::max() - count) {
            throw std::overflow_error("parameter count overflow");
        }
        total += count;
    }
    return total;
}

void move_parameters_to(
    const ParameterList& parameters,
    ExecutionBackend backend
) {
    std::unordered_set<Parameter*> seen_parameters;
    for (const auto& named_parameter : parameters) {
        if (named_parameter.parameter == nullptr) {
            throw std::invalid_argument(
                "parameter list contains a null parameter"
            );
        }
        if (!seen_parameters.insert(named_parameter.parameter).second) {
            throw std::invalid_argument(
                "parameter list contains a duplicate parameter"
            );
        }
    }

    struct TransferCandidate {
        Parameter* parameter;
        Tensor value;
        Tensor gradient;
    };
    std::vector<TransferCandidate> candidates;
    candidates.reserve(parameters.size());
    for (const auto& named_parameter : parameters) {
        if (named_parameter.parameter->value().backend() == backend) {
            continue;
        }
        Tensor next_value =
            named_parameter.parameter->value().to(backend);
        Tensor next_gradient = Tensor::zeros(
            next_value.shape(),
            backend
        );
        candidates.push_back({
            named_parameter.parameter,
            std::move(next_value),
            std::move(next_gradient),
        });
    }

    for (auto& candidate : candidates) {
        candidate.parameter->replace_state(
            std::move(candidate.value),
            std::move(candidate.gradient)
        );
    }
}

}  // namespace riftco_transformer
