#include "riftco_transformer/model/feed_forward.hpp"

#include "riftco_transformer/nn/activations.hpp"

#include <stdexcept>

namespace riftco_transformer {
namespace {

FeedForwardActivation checked_activation(FeedForwardActivation activation) {
  switch (activation) {
  case FeedForwardActivation::Gelu:
  case FeedForwardActivation::Relu:
    return activation;
  }
  throw std::invalid_argument("feed-forward activation is not recognized");
}

} // namespace

FeedForward::FeedForward(std::size_t model_width, std::size_t hidden_width,
                         std::mt19937 &random)
    : FeedForward(model_width, hidden_width, random,
                  FeedForwardActivation::Gelu) {}

FeedForward::FeedForward(std::size_t model_width, std::size_t hidden_width,
                         std::mt19937 &random, FeedForwardActivation activation)
    : activation_(checked_activation(activation)),
      expand_(model_width, hidden_width, random),
      project_(hidden_width, model_width, random) {
  register_module("expand", expand_);
  register_module("project", project_);
}

std::size_t FeedForward::model_width() const noexcept {
    return expand_.input_width();
}

std::size_t FeedForward::hidden_width() const noexcept {
    return expand_.output_width();
}

FeedForwardActivation FeedForward::activation() const noexcept {
  return activation_;
}

Variable FeedForward::forward(const Variable& input) const {
  const Variable expanded = expand_.forward(input);
  switch (activation()) {
  case FeedForwardActivation::Gelu:
    return project_.forward(gelu(expanded));
  case FeedForwardActivation::Relu:
    return project_.forward(relu(expanded));
  }
  throw std::logic_error("feed-forward activation state is invalid");
}

void FeedForward::to(ExecutionBackend backend) {
    Module::to(backend);
}

ParameterList FeedForward::parameters() {
    return Module::parameters();
}

ParameterList FeedForward::lora_parameters() {
    ParameterList result;
    append_parameter_group(
        result,
        "expand",
        expand_.lora_parameters()
    );
    append_parameter_group(
        result,
        "project",
        project_.lora_parameters()
    );
    return result;
}

}  // namespace riftco_transformer
