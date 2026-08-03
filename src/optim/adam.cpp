#include "riftco_transformer/optim/adam.hpp"

#include "core/backend/adapter.hpp"
#include "core/backend/optim/adam/dispatch.hpp"

#include <algorithm>
#include <array>
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
    switch (options.state_storage) {
        case AdamStateStorageKind::Contiguous:
        case AdamStateStorageKind::Paged:
            break;
        default:
            throw std::invalid_argument(
                "Adam state storage kind is not recognized"
            );
    }
    if (options.page_size == 0) {
        throw std::invalid_argument(
            "Adam state page size must be greater than zero"
        );
    }
    return options;
}

void add_state_payload_bytes(
    std::size_t element_count,
    std::size_t& total
) {
    constexpr std::size_t bytes_per_element = 2 * sizeof(float);
    if (element_count >
        (std::numeric_limits<std::size_t>::max() - total) /
            bytes_per_element) {
        throw std::overflow_error(
            "Adam state payload size overflow"
        );
    }
    total += element_count * bytes_per_element;
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

std::size_t total_parameter_values(const ParameterList& parameters) {
    std::size_t total = 0;
    for (const auto& named_parameter : parameters) {
        const std::size_t count =
            named_parameter.parameter->value().numel();
        if (count > std::numeric_limits<std::size_t>::max() - total) {
            throw std::overflow_error("Adam state value count overflow");
        }
        total += count;
    }
    return total;
}

void require_finite_state_values(
    const std::vector<float>& values,
    const char* description
) {
    if (!std::all_of(
            values.begin(),
            values.end(),
            [](float value) { return std::isfinite(value); }
        )) {
        throw std::invalid_argument(
            std::string("Adam ") + description +
            " must contain only finite values"
        );
    }
}

void require_clean_step_boundary(const ParameterList& parameters) {
    for (const auto& named_parameter : parameters) {
        if (named_parameter.parameter->has_pending_gradient()) {
            throw std::logic_error(
                "Adam state can only be captured at a clean "
                "post-step boundary"
            );
        }
    }
}

void require_valid_beta_power(
    double power,
    std::size_t step_count,
    float beta,
    const char* description
) {
    if (!std::isfinite(power) || power < 0.0 || power > 1.0 ||
        (step_count == 0 && power != 1.0) ||
        (step_count != 0 && power == 1.0)) {
        throw std::invalid_argument(
            std::string("Adam ") + description +
            " is inconsistent with the step count"
        );
    }
    const double expected = std::pow(
        static_cast<double>(beta),
        static_cast<double>(step_count)
    );
    if (expected <= std::numeric_limits<double>::min()) {
        // Adam advances this value by repeated multiplication. Once the
        // mathematical value is subnormal, pow() and the recurrence can
        // round differently or settle at different subnormal values.
        if (power > std::numeric_limits<double>::min()) {
            throw std::invalid_argument(
                std::string("Adam ") + description +
                " does not match beta raised to the step count"
            );
        }
        return;
    }
    constexpr double relative_tolerance = 1.0e-12;
    const double absolute_tolerance = std::max(
        expected * relative_tolerance,
        8.0 * std::numeric_limits<double>::denorm_min()
    );
    if (std::fabs(power - expected) > absolute_tolerance) {
        throw std::invalid_argument(
            std::string("Adam ") + description +
            " does not match beta raised to the step count"
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
    if (options_.state_storage == AdamStateStorageKind::Contiguous) {
        first_moments_.reserve(parameters_.size());
        second_moments_.reserve(parameters_.size());
    } else {
        state_pages_.resize(parameters_.size());
    }
    std::size_t parameter_index = 0;
    for (const auto& named_parameter : parameters_) {
        require_parameter_backend(named_parameter, backend_);
        require_finite_parameter_values(named_parameter);
        const Tensor& value = named_parameter.parameter->value();
        add_state_payload_bytes(value.numel(), state_payload_bytes_);
        if (options_.state_storage == AdamStateStorageKind::Contiguous) {
            first_moments_.push_back(
                Tensor::zeros(value.shape(), backend_)
            );
            second_moments_.push_back(
                Tensor::zeros(value.shape(), backend_)
            );
        } else {
            auto& pages = state_pages_[parameter_index];
            const std::size_t page_count =
                1 + (value.numel() - 1) / options_.page_size;
            pages.reserve(page_count);
            for (std::size_t offset = 0;
                 offset < value.numel();) {
                const std::size_t length = std::min(
                    options_.page_size,
                    value.numel() - offset
                );
                pages.push_back({
                    offset,
                    Tensor::zeros({length}, backend_),
                    Tensor::zeros({length}, backend_),
                });
                offset += length;
            }
        }
        ++parameter_index;
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

AdamStateStorageKind Adam::state_storage_kind() const noexcept {
    return options_.state_storage;
}

std::size_t Adam::state_page_size() const noexcept {
    return options_.page_size;
}

std::size_t Adam::state_page_count() const noexcept {
    std::size_t result = 0;
    for (const auto& pages : state_pages_) {
        result += pages.size();
    }
    return result;
}

std::size_t Adam::state_payload_bytes() const noexcept {
    return state_payload_bytes_;
}

std::size_t Adam::state_value_count() const noexcept {
    return state_payload_bytes_ / (2 * sizeof(float));
}

double Adam::beta1_power() const noexcept {
    return beta1_power_;
}

double Adam::beta2_power() const noexcept {
    return beta2_power_;
}

AdamState Adam::state() const {
    require_clean_step_boundary(parameters_);
    const std::size_t total = total_parameter_values(parameters_);
    AdamState result;
    result.step_count = step_count_;
    result.beta1_power = beta1_power_;
    result.beta2_power = beta2_power_;
    result.parameter_values.reserve(total);
    result.first_moments.reserve(total);
    result.second_moments.reserve(total);

    for (std::size_t parameter_index = 0;
         parameter_index < parameters_.size();
         ++parameter_index) {
        const auto& parameter = *parameters_[parameter_index].parameter;
        require_parameter_backend(parameters_[parameter_index], backend_);
        require_finite_parameter_values(parameters_[parameter_index]);
        const auto values = parameter.value().data();
        result.parameter_values.insert(
            result.parameter_values.end(),
            values.begin(),
            values.end()
        );

        if (options_.state_storage == AdamStateStorageKind::Contiguous) {
            const auto first = first_moments_[parameter_index].data();
            const auto second = second_moments_[parameter_index].data();
            result.first_moments.insert(
                result.first_moments.end(), first.begin(), first.end()
            );
            result.second_moments.insert(
                result.second_moments.end(), second.begin(), second.end()
            );
            continue;
        }

        std::size_t expected_offset = 0;
        for (const auto& page : state_pages_[parameter_index]) {
            if (page.offset != expected_offset ||
                page.first_moment.numel() !=
                    page.second_moment.numel()) {
                throw std::logic_error(
                    "Adam paged state is not contiguous"
                );
            }
            const auto first = page.first_moment.data();
            const auto second = page.second_moment.data();
            result.first_moments.insert(
                result.first_moments.end(), first.begin(), first.end()
            );
            result.second_moments.insert(
                result.second_moments.end(), second.begin(), second.end()
            );
            expected_offset += first.size();
        }
        if (expected_offset != parameter.value().numel()) {
            throw std::logic_error(
                "Adam paged state does not match its parameter"
            );
        }
    }
    require_finite_state_values(result.first_moments, "first moments");
    require_finite_state_values(result.second_moments, "second moments");
    return result;
}

void Adam::load_state(AdamState state) {
    require_clean_step_boundary(parameters_);
    const std::size_t total = total_parameter_values(parameters_);
    if (state.parameter_values.size() != total ||
        state.first_moments.size() != total ||
        state.second_moments.size() != total) {
        throw std::invalid_argument(
            "Adam state value counts do not match the parameter list"
        );
    }
    require_valid_beta_power(
        state.beta1_power,
        state.step_count,
        options_.beta1,
        "beta1 power"
    );
    require_valid_beta_power(
        state.beta2_power,
        state.step_count,
        options_.beta2,
        "beta2 power"
    );
    require_finite_state_values(
        state.parameter_values, "parameter values"
    );
    require_finite_state_values(state.first_moments, "first moments");
    require_finite_state_values(state.second_moments, "second moments");

    std::vector<Tensor> next_values;
    std::vector<Tensor> next_gradients;
    std::vector<Tensor> next_first_moments;
    std::vector<Tensor> next_second_moments;
    std::vector<std::vector<StatePage>> next_state_pages;
    next_values.reserve(parameters_.size());
    next_gradients.reserve(parameters_.size());
    if (options_.state_storage == AdamStateStorageKind::Contiguous) {
        next_first_moments.reserve(parameters_.size());
        next_second_moments.reserve(parameters_.size());
    } else {
        next_state_pages.resize(parameters_.size());
    }

    std::size_t offset = 0;
    for (std::size_t parameter_index = 0;
         parameter_index < parameters_.size();
         ++parameter_index) {
        const auto& named_parameter = parameters_[parameter_index];
        require_parameter_backend(named_parameter, backend_);
        const auto& old_value = named_parameter.parameter->value();
        const std::size_t count = old_value.numel();
        next_values.emplace_back(
            old_value.shape(),
            std::vector<float>(
                state.parameter_values.begin() +
                    static_cast<std::ptrdiff_t>(offset),
                state.parameter_values.begin() +
                    static_cast<std::ptrdiff_t>(offset + count)
            ),
            backend_
        );
        next_gradients.push_back(
            Tensor::zeros(old_value.shape(), backend_)
        );

        if (options_.state_storage == AdamStateStorageKind::Contiguous) {
            next_first_moments.emplace_back(
                old_value.shape(),
                std::vector<float>(
                    state.first_moments.begin() +
                        static_cast<std::ptrdiff_t>(offset),
                    state.first_moments.begin() +
                        static_cast<std::ptrdiff_t>(offset + count)
                ),
                backend_
            );
            next_second_moments.emplace_back(
                old_value.shape(),
                std::vector<float>(
                    state.second_moments.begin() +
                        static_cast<std::ptrdiff_t>(offset),
                    state.second_moments.begin() +
                        static_cast<std::ptrdiff_t>(offset + count)
                ),
                backend_
            );
        } else {
            auto& pages = next_state_pages[parameter_index];
            const std::size_t page_count =
                1 + (count - 1) / options_.page_size;
            pages.reserve(page_count);
            for (std::size_t page_offset = 0;
                 page_offset < count;) {
                const std::size_t length = std::min(
                    options_.page_size, count - page_offset
                );
                const std::size_t flat_offset = offset + page_offset;
                pages.push_back({
                    page_offset,
                    Tensor(
                        {length},
                        std::vector<float>(
                            state.first_moments.begin() +
                                static_cast<std::ptrdiff_t>(flat_offset),
                            state.first_moments.begin() +
                                static_cast<std::ptrdiff_t>(
                                    flat_offset + length
                                )
                        ),
                        backend_
                    ),
                    Tensor(
                        {length},
                        std::vector<float>(
                            state.second_moments.begin() +
                                static_cast<std::ptrdiff_t>(flat_offset),
                            state.second_moments.begin() +
                                static_cast<std::ptrdiff_t>(
                                    flat_offset + length
                                )
                        ),
                        backend_
                    ),
                });
                page_offset += length;
            }
        }
        offset += count;
    }

    // Validation and every allocation are complete. Parameter replacement
    // and vector moves below are non-allocating commits.
    for (std::size_t parameter_index = 0;
         parameter_index < parameters_.size();
         ++parameter_index) {
        parameters_[parameter_index].parameter->replace_state(
            std::move(next_values[parameter_index]),
            std::move(next_gradients[parameter_index])
        );
    }
    if (options_.state_storage == AdamStateStorageKind::Contiguous) {
        first_moments_ = std::move(next_first_moments);
        second_moments_ = std::move(next_second_moments);
    } else {
        state_pages_ = std::move(next_state_pages);
    }
    step_count_ = state.step_count;
    beta1_power_ = state.beta1_power;
    beta2_power_ = state.beta2_power;
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
    if (options_.state_storage == AdamStateStorageKind::Contiguous) {
        next_first_moments.reserve(parameters_.size());
        next_second_moments.reserve(parameters_.size());
    }

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
        if (options_.state_storage == AdamStateStorageKind::Contiguous) {
            next_first_moments.push_back(
                Tensor::zeros(value.shape(), backend_)
            );
            next_second_moments.push_back(
                Tensor::zeros(value.shape(), backend_)
            );
        }
    }

    const auto dispatch_updates =
        [&](std::span<const backend_detail::AdamTensorUpdate> updates) {
            backend_detail::dispatch_adam_update(
                backend_,
                {
                    updates,
                    options_.learning_rate,
                    options_.beta1,
                    options_.beta2,
                    options_.epsilon,
                    clip_scale,
                    first_correction,
                    second_correction,
                }
            );
        };

    std::vector<std::vector<StatePage>> next_state_pages;
    if (options_.state_storage == AdamStateStorageKind::Contiguous) {
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
        dispatch_updates(updates);
    } else {
        next_state_pages.resize(parameters_.size());
        for (std::size_t parameter_index = 0;
             parameter_index < parameters_.size();
             ++parameter_index) {
            const Parameter& parameter =
                *parameters_[parameter_index].parameter;
            const auto value = parameter.value().data();
            const auto gradient = parameter.gradient().data();
            auto next_value = next_values[parameter_index].data();
            auto& candidate_pages = next_state_pages[parameter_index];
            candidate_pages.reserve(state_pages_[parameter_index].size());

            for (const auto& page : state_pages_[parameter_index]) {
                const std::size_t length = page.first_moment.numel();
                if (page.second_moment.numel() != length ||
                    page.offset > value.size() ||
                    length > value.size() - page.offset) {
                    throw std::logic_error(
                        "Adam paged state does not match its parameter"
                    );
                }
                const auto value_page =
                    value.subspan(page.offset, length);
                const auto gradient_page =
                    gradient.subspan(page.offset, length);
                Tensor page_value(
                    {length},
                    std::vector<float>(
                        value_page.begin(),
                        value_page.end()
                    ),
                    backend_
                );
                Tensor page_gradient(
                    {length},
                    std::vector<float>(
                        gradient_page.begin(),
                        gradient_page.end()
                    ),
                    backend_
                );
                Tensor next_page_value =
                    Tensor::zeros({length}, backend_);
                Tensor next_first =
                    Tensor::zeros({length}, backend_);
                Tensor next_second =
                    Tensor::zeros({length}, backend_);

                const std::array<backend_detail::AdamTensorUpdate, 1>
                    update{{{
                        backend_detail::tensor_storage(page_value),
                        backend_detail::tensor_storage(page_gradient),
                        backend_detail::tensor_storage(page.first_moment),
                        backend_detail::tensor_storage(page.second_moment),
                        backend_detail::tensor_storage(next_page_value),
                        backend_detail::tensor_storage(next_first),
                        backend_detail::tensor_storage(next_second),
                    }}};
                dispatch_updates(update);
                std::copy(
                    next_page_value.data().begin(),
                    next_page_value.data().end(),
                    next_value.subspan(page.offset, length).begin()
                );
                candidate_pages.push_back({
                    page.offset,
                    std::move(next_first),
                    std::move(next_second),
                });
            }
        }
    }

    for (std::size_t parameter_index = 0;
         parameter_index < parameters_.size();
         ++parameter_index) {
        parameters_[parameter_index].parameter->set_value(
            std::move(next_values[parameter_index])
        );
        if (options_.state_storage == AdamStateStorageKind::Contiguous) {
            first_moments_[parameter_index] =
                std::move(next_first_moments[parameter_index]);
            second_moments_[parameter_index] =
                std::move(next_second_moments[parameter_index]);
        }
    }
    if (options_.state_storage == AdamStateStorageKind::Paged) {
        state_pages_ = std::move(next_state_pages);
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
