#include "riftco_transformer/optim/adam.hpp"

#include "core/backend/adapter.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace riftco_transformer {
namespace {

static_assert(std::is_nothrow_move_constructible_v<Tensor>);
static_assert(std::is_nothrow_move_assignable_v<Tensor>);

AdamOptions checked_options(AdamOptions options) {
    if (!std::isfinite(options.learning_rate) ||
        options.learning_rate <= 0.0F) {
        throw std::invalid_argument(
            "Adam learning rate must be finite and positive"
        );
    }
    if (!std::isfinite(options.beta1) ||
        options.beta1 <= 0.0F ||
        options.beta1 >= 1.0F ||
        !std::isfinite(options.beta2) ||
        options.beta2 <= 0.0F ||
        options.beta2 >= 1.0F) {
        throw std::invalid_argument(
            "Adam beta values must be finite and in (0, 1)"
        );
    }
    if (!std::isfinite(options.epsilon) ||
        options.epsilon <= 0.0F) {
        throw std::invalid_argument(
            "Adam epsilon must be finite and positive"
        );
    }
    if (!std::isfinite(options.maximum_gradient_norm) ||
        options.maximum_gradient_norm <= 0.0F) {
        throw std::invalid_argument(
            "Adam maximum gradient norm must be finite and positive"
        );
    }
    return options;
}

ParameterList checked_parameters(ParameterList parameters) {
    if (parameters.empty()) {
        throw std::invalid_argument(
            "Adam requires at least one parameter"
        );
    }

    std::unordered_set<const Parameter*> seen_parameters;
    std::unordered_set<std::string> seen_names;
    for (const auto& named_parameter : parameters) {
        if (named_parameter.name.empty()) {
            throw std::invalid_argument(
                "Adam parameter names must not be empty"
            );
        }
        if (named_parameter.parameter == nullptr) {
            throw std::invalid_argument(
                "Adam parameter list contains a null pointer"
            );
        }
        if (!seen_parameters.insert(named_parameter.parameter).second) {
            throw std::invalid_argument(
                "Adam parameter list contains a duplicate parameter"
            );
        }
        if (!seen_names.insert(named_parameter.name).second) {
            throw std::invalid_argument(
                "Adam parameter names must be unique"
            );
        }
    }
    return parameters;
}

void update_scaled_sum_of_squares(
    double value,
    double& scale,
    double& sum_of_squares
) {
    if (!std::isfinite(value)) {
        throw std::domain_error(
            "Adam gradients must contain only finite values"
        );
    }
    const double magnitude = std::fabs(value);
    if (magnitude == 0.0) {
        return;
    }
    if (scale < magnitude) {
        const double ratio = scale / magnitude;
        sum_of_squares =
            1.0 + sum_of_squares * ratio * ratio;
        scale = magnitude;
    } else {
        const double ratio = magnitude / scale;
        sum_of_squares += ratio * ratio;
    }
}

void require_finite_parameter_values(
    const NamedParameter& named_parameter
) {
    for (const float value :
         named_parameter.parameter->value().data()) {
        if (!std::isfinite(value)) {
            throw std::domain_error(
                "Adam parameter '" + named_parameter.name +
                "' contains a non-finite value"
            );
        }
    }
}

void require_parameter_backend(
    const NamedParameter& named_parameter,
    ExecutionBackend backend
) {
    const Parameter& parameter = *named_parameter.parameter;
    if (parameter.value().backend() != backend ||
        parameter.gradient().backend() != backend) {
        throw std::invalid_argument(
            "Adam parameter '" + named_parameter.name +
            "' does not match the optimizer backend"
        );
    }
    if (parameter.gradient().shape() != parameter.value().shape()) {
        throw std::logic_error(
            "Adam parameter '" + named_parameter.name +
            "' gradient shape does not match its value"
        );
    }
}

}  // namespace

double global_gradient_norm(const ParameterList& parameters) {
    std::unordered_set<const Parameter*> seen_parameters;
    double scale = 0.0;
    double sum_of_squares = 1.0;
    for (const auto& named_parameter : parameters) {
        if (named_parameter.parameter == nullptr) {
            throw std::invalid_argument(
                "gradient norm parameter list contains a null pointer"
            );
        }
        if (!seen_parameters.insert(named_parameter.parameter).second) {
            throw std::invalid_argument(
                "gradient norm parameter list contains a duplicate"
            );
        }
        if (named_parameter.parameter->gradient().shape() !=
            named_parameter.parameter->value().shape()) {
            throw std::logic_error(
                "parameter gradient shape does not match its value"
            );
        }
        if (named_parameter.parameter->gradient().backend() !=
            named_parameter.parameter->value().backend()) {
            throw std::logic_error(
                "parameter gradient backend does not match its value"
            );
        }
        for (const float gradient :
             named_parameter.parameter->gradient().data()) {
            update_scaled_sum_of_squares(
                static_cast<double>(gradient),
                scale,
                sum_of_squares
            );
        }
    }
    return scale == 0.0
               ? 0.0
               : scale * std::sqrt(sum_of_squares);
}

Adam::Adam(
    ParameterList parameters,
    AdamOptions options
)
    : options_(checked_options(options)),
      parameters_(checked_parameters(std::move(parameters))),
      backend_(
          parameters_.front().parameter->value().backend()
      ) {
    first_moments_.reserve(parameters_.size());
    second_moments_.reserve(parameters_.size());
    for (const auto& named_parameter : parameters_) {
        require_parameter_backend(named_parameter, backend_);
        require_finite_parameter_values(named_parameter);
        first_moments_.push_back(
            Tensor::zeros(
                named_parameter.parameter->value().shape(),
                backend_
            )
        );
        second_moments_.push_back(
            Tensor::zeros(
                named_parameter.parameter->value().shape(),
                backend_
            )
        );
    }
    static_cast<void>(global_gradient_norm(parameters_));
}

const AdamOptions& Adam::options() const noexcept {
    return options_;
}

ExecutionBackend Adam::backend() const noexcept {
    return backend_;
}

std::size_t Adam::step_count() const noexcept {
    return step_count_;
}

std::size_t Adam::parameter_tensor_count() const noexcept {
    return parameters_.size();
}

AdamStepStats Adam::step() {
    if (step_count_ == std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("Adam step counter overflow");
    }

    const double gradient_norm =
        global_gradient_norm(parameters_);
    const double clip_scale =
        gradient_norm > options_.maximum_gradient_norm
            ? static_cast<double>(
                  options_.maximum_gradient_norm
              ) / gradient_norm
            : 1.0;
    const double next_beta1_power =
        beta1_power_ * static_cast<double>(options_.beta1);
    const double next_beta2_power =
        beta2_power_ * static_cast<double>(options_.beta2);
    const double first_correction = 1.0 - next_beta1_power;
    const double second_correction = 1.0 - next_beta2_power;

    std::vector<Tensor> next_values;
    std::vector<Tensor> next_first_moments;
    std::vector<Tensor> next_second_moments;
    next_values.reserve(parameters_.size());
    next_first_moments.reserve(parameters_.size());
    next_second_moments.reserve(parameters_.size());

    for (std::size_t parameter_index = 0;
         parameter_index < parameters_.size();
         ++parameter_index) {
        const auto& named_parameter = parameters_[parameter_index];
        require_parameter_backend(named_parameter, backend_);
        require_finite_parameter_values(named_parameter);
        const Tensor& value = named_parameter.parameter->value();
        next_values.push_back(
            Tensor::zeros(value.shape(), backend_)
        );
        next_first_moments.push_back(
            Tensor::zeros(value.shape(), backend_)
        );
        next_second_moments.push_back(
            Tensor::zeros(value.shape(), backend_)
        );
    }

    std::vector<backend_detail::AdamTensorUpdate> updates;
    updates.reserve(parameters_.size());
    for (std::size_t parameter_index = 0;
         parameter_index < parameters_.size();
         ++parameter_index) {
        const Parameter& parameter =
            *parameters_[parameter_index].parameter;
        updates.push_back({
            backend_detail::tensor_storage(parameter.value()),
            backend_detail::tensor_storage(parameter.gradient()),
            backend_detail::tensor_storage(
                first_moments_[parameter_index]
            ),
            backend_detail::tensor_storage(
                second_moments_[parameter_index]
            ),
            backend_detail::tensor_storage(
                next_values[parameter_index]
            ),
            backend_detail::tensor_storage(
                next_first_moments[parameter_index]
            ),
            backend_detail::tensor_storage(
                next_second_moments[parameter_index]
            ),
        });
    }
    backend_detail::dispatch_adam_update(
        backend_,
        {
            std::span<const backend_detail::AdamTensorUpdate>(
                updates
            ),
            options_.learning_rate,
            options_.beta1,
            options_.beta2,
            options_.epsilon,
            clip_scale,
            first_correction,
            second_correction,
        }
    );

    for (std::size_t parameter_index = 0;
         parameter_index < parameters_.size();
         ++parameter_index) {
        parameters_[parameter_index].parameter->set_value(
            std::move(next_values[parameter_index])
        );
        first_moments_[parameter_index] =
            std::move(next_first_moments[parameter_index]);
        second_moments_[parameter_index] =
            std::move(next_second_moments[parameter_index]);
    }

    ++step_count_;
    beta1_power_ = next_beta1_power;
    beta2_power_ = next_beta2_power;
    return {
        step_count_,
        gradient_norm,
        clip_scale,
    };
}

void Adam::zero_gradients() const {
    for (const auto& named_parameter : parameters_) {
        named_parameter.parameter->zero_gradient();
    }
}

}  // namespace riftco_transformer
