#include "riftco_transformer/experiments/conditional_reverse/learned_training.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace riftco_transformer::experiments::conditional_reverse {
namespace {

[[nodiscard]] std::size_t checked_add(std::size_t left, std::size_t right,
                                      const char *description) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::overflow_error(description);
  }
  return left + right;
}

[[nodiscard]] std::size_t checked_multiply(std::size_t left, std::size_t right,
                                           const char *description) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error(description);
  }
  return left * right;
}

[[nodiscard]] bool same_protocol(const LearnedProtocolConfig &left,
                                 const LearnedProtocolConfig &right) {
  return left.sequence_length == right.sequence_length &&
         left.alphabet == right.alphabet &&
         left.reverse_when_first_is == right.reverse_when_first_is &&
         left.delimiter == right.delimiter;
}

[[nodiscard]] LearnedTrainingConfig
checked_training_config(LearnedTrainingConfig config) {
  config.validate();
  return config;
}

void require_dataset_compatibility(const LearnedHybrid &model,
                                   const LearnedDataset &dataset) {
  if (dataset.empty()) {
    throw std::invalid_argument(
        "learned conditional-reverse training dataset must be nonempty");
  }
  if (!same_protocol(model.config().protocol, dataset.config())) {
    throw std::invalid_argument(
        "learned conditional-reverse model and dataset protocols differ");
  }
}

[[nodiscard]] float checked_scalar_loss(const Variable &loss) {
  if (loss.value().rank() != 0 || loss.value().numel() != 1) {
    throw std::logic_error(
        "learned conditional-reverse training loss must be scalar");
  }
  const Tensor host = loss.value().backend() == ExecutionBackend::Cpu
                          ? Tensor(loss.value())
                          : loss.value().to(ExecutionBackend::Cpu);
  const float value = host.flat(0);
  if (!std::isfinite(value)) {
    throw std::domain_error(
        "learned conditional-reverse training loss must be finite");
  }
  return value;
}

[[nodiscard]] LearnedBatch indexed_batch(const LearnedDataset &dataset,
                                         std::span<const std::size_t> indices) {
  std::vector<LearnedExample> examples;
  examples.reserve(indices.size());
  for (const std::size_t index : indices) {
    if (index >= dataset.size()) {
      throw std::out_of_range(
          "learned conditional-reverse training index is out of range");
    }
    examples.push_back(dataset[index]);
  }
  return make_learned_batch(examples, dataset.config());
}

struct RepresentationAccumulator {
  std::string name;
  std::vector<std::size_t> trailing_leading_shape;
  std::size_t example_count = 0;
  std::size_t columns = 0;
  std::vector<float> values;
};

void append_representations(
    std::vector<RepresentationAccumulator> &accumulators,
    const analysis::RepresentationTrace &trace, std::size_t batch_size) {
  const auto entries = trace.entries();
  if (accumulators.empty()) {
    accumulators.reserve(entries.size());
    for (const analysis::NamedRepresentation &entry : entries) {
      if (entry.leading_shape.empty() ||
          entry.leading_shape.front() != batch_size) {
        throw std::logic_error(
            "learned representation capture must lead with batch size");
      }
      RepresentationAccumulator accumulator;
      accumulator.name = entry.name;
      accumulator.trailing_leading_shape.assign(entry.leading_shape.begin() + 1,
                                                entry.leading_shape.end());
      accumulator.example_count = batch_size;
      accumulator.columns = entry.observations.columns;
      accumulator.values = entry.observations.values;
      accumulators.push_back(std::move(accumulator));
    }
    return;
  }
  if (entries.size() != accumulators.size()) {
    throw std::logic_error(
        "learned representation capture names changed between batches");
  }
  for (std::size_t index = 0; index < entries.size(); ++index) {
    const analysis::NamedRepresentation &entry = entries[index];
    RepresentationAccumulator &accumulator = accumulators[index];
    if (entry.leading_shape.empty()) {
      throw std::logic_error(
          "learned representation capture must lead with batch size");
    }
    const std::vector<std::size_t> trailing(entry.leading_shape.begin() + 1,
                                            entry.leading_shape.end());
    if (entry.name != accumulator.name ||
        entry.leading_shape.front() != batch_size ||
        trailing != accumulator.trailing_leading_shape ||
        entry.observations.columns != accumulator.columns) {
      throw std::logic_error(
          "learned representation capture shape changed between batches");
    }
    accumulator.example_count =
        checked_add(accumulator.example_count, batch_size,
                    "learned representation capture batch count overflows");
    accumulator.values.insert(accumulator.values.end(),
                              entry.observations.values.begin(),
                              entry.observations.values.end());
  }
}

[[nodiscard]] analysis::RepresentationTrace
finish_representations(std::vector<RepresentationAccumulator> accumulators) {
  analysis::RepresentationTrace result;
  for (RepresentationAccumulator &accumulator : accumulators) {
    std::vector<std::size_t> leading_shape{accumulator.example_count};
    leading_shape.insert(leading_shape.end(),
                         accumulator.trailing_leading_shape.begin(),
                         accumulator.trailing_leading_shape.end());
    if (accumulator.columns == 0 ||
        accumulator.values.size() % accumulator.columns != 0) {
      throw std::logic_error(
          "learned representation capture has inconsistent storage");
    }
    result.capture({
        .name = std::move(accumulator.name),
        .leading_shape = std::move(leading_shape),
        .observations =
            {
                .rows = accumulator.values.size() / accumulator.columns,
                .columns = accumulator.columns,
                .values = std::move(accumulator.values),
            },
    });
  }
  return result;
}

void accumulate_metrics(LearnedEvaluationMetrics &total,
                        const LearnedEvaluationMetrics &batch) {
  total.loss += batch.loss * static_cast<float>(batch.example_count);
  total.example_count += batch.example_count;
  total.target_token_count += batch.target_token_count;
  total.correct_target_token_count += batch.correct_target_token_count;
  total.correct_sequence_count += batch.correct_sequence_count;
  total.reverse_example_count += batch.reverse_example_count;
  total.reverse_correct_target_token_count +=
      batch.reverse_correct_target_token_count;
  total.reverse_correct_sequence_count += batch.reverse_correct_sequence_count;
  total.copy_example_count += batch.copy_example_count;
  total.copy_correct_target_token_count +=
      batch.copy_correct_target_token_count;
  total.copy_correct_sequence_count += batch.copy_correct_sequence_count;
}

void finalize_metrics(LearnedEvaluationMetrics &metrics,
                      std::size_t sequence_length) {
  if (metrics.example_count == 0 || metrics.target_token_count == 0) {
    throw std::logic_error(
        "learned conditional-reverse cannot finalize empty metrics");
  }
  metrics.loss /= static_cast<float>(metrics.example_count);
  metrics.target_token_accuracy =
      static_cast<double>(metrics.correct_target_token_count) /
      static_cast<double>(metrics.target_token_count);
  metrics.exact_sequence_accuracy =
      static_cast<double>(metrics.correct_sequence_count) /
      static_cast<double>(metrics.example_count);
  if (metrics.reverse_example_count != 0) {
    metrics.reverse_target_token_accuracy =
        static_cast<double>(metrics.reverse_correct_target_token_count) /
        static_cast<double>(metrics.reverse_example_count * sequence_length);
    metrics.reverse_exact_sequence_accuracy =
        static_cast<double>(metrics.reverse_correct_sequence_count) /
        static_cast<double>(metrics.reverse_example_count);
  }
  if (metrics.copy_example_count != 0) {
    metrics.copy_target_token_accuracy =
        static_cast<double>(metrics.copy_correct_target_token_count) /
        static_cast<double>(metrics.copy_example_count * sequence_length);
    metrics.copy_exact_sequence_accuracy =
        static_cast<double>(metrics.copy_correct_sequence_count) /
        static_cast<double>(metrics.copy_example_count);
  }
}

} // namespace

void LearnedTrainingConfig::validate() const {
  if (epochs == 0 || batch_size == 0 || evaluation_batch_size == 0) {
    throw std::invalid_argument(
        "learned conditional-reverse training dimensions must be positive");
  }
  if (maximum_steps.has_value() && *maximum_steps == 0) {
    throw std::invalid_argument(
        "learned conditional-reverse maximum steps must be positive");
  }
  // Adam performs the detailed numerical validation. These checks keep bad
  // experiment configurations from constructing a partial trainer.
  if (!std::isfinite(adam.learning_rate) || adam.learning_rate <= 0.0F ||
      !std::isfinite(adam.beta1) || adam.beta1 <= 0.0F || adam.beta1 >= 1.0F ||
      !std::isfinite(adam.beta2) || adam.beta2 <= 0.0F || adam.beta2 >= 1.0F ||
      !std::isfinite(adam.epsilon) || adam.epsilon <= 0.0F ||
      !std::isfinite(adam.maximum_gradient_norm) ||
      adam.maximum_gradient_norm <= 0.0F) {
    throw std::invalid_argument(
        "learned conditional-reverse Adam options are invalid");
  }
}

LearnedEvaluationResult
evaluate_learned_dataset(const LearnedHybrid &model,
                         const LearnedDataset &dataset, std::size_t batch_size,
                         const LearnedForwardOptions &options) {
  require_dataset_compatibility(model, dataset);
  if (batch_size == 0) {
    throw std::invalid_argument(
        "learned conditional-reverse evaluation batch size must be positive");
  }

  LearnedEvaluationResult result;
  std::vector<RepresentationAccumulator> representation_accumulators;
  for (std::size_t start = 0; start < dataset.size();) {
    const std::size_t count = std::min(batch_size, dataset.size() - start);
    const std::span<const LearnedExample> examples =
        dataset.examples().subspan(start, count);
    LearnedEvaluationResult batch_result = model.evaluate(examples, options);
    accumulate_metrics(result.metrics, batch_result.metrics);
    result.predictions.insert(result.predictions.end(),
                              batch_result.predictions.begin(),
                              batch_result.predictions.end());
    result.per_example_token_accuracy.insert(
        result.per_example_token_accuracy.end(),
        batch_result.per_example_token_accuracy.begin(),
        batch_result.per_example_token_accuracy.end());
    result.per_example_exact_accuracy.insert(
        result.per_example_exact_accuracy.end(),
        batch_result.per_example_exact_accuracy.begin(),
        batch_result.per_example_exact_accuracy.end());
    if (options.capture_representations) {
      append_representations(representation_accumulators,
                             batch_result.representations, count);
    }
    start = checked_add(start, count,
                        "learned conditional-reverse evaluation overflows");
  }
  if (options.capture_representations) {
    result.representations =
        finish_representations(std::move(representation_accumulators));
  }
  finalize_metrics(result.metrics, model.config().protocol.sequence_length);
  return result;
}

LearnedHybridTrainer::LearnedHybridTrainer(LearnedHybrid &model,
                                           LearnedTrainingConfig config)
    : model_(model), config_(checked_training_config(std::move(config))),
      optimizer_(model_.parameters(), config_.adam) {
  if (optimizer_.backend() != model_.backend()) {
    throw std::invalid_argument(
        "learned conditional-reverse model and Adam backends differ");
  }
}

const LearnedTrainingConfig &LearnedHybridTrainer::config() const noexcept {
  return config_;
}

LearnedHybrid &LearnedHybridTrainer::model() noexcept { return model_; }

const LearnedHybrid &LearnedHybridTrainer::model() const noexcept {
  return model_;
}

Adam &LearnedHybridTrainer::optimizer() noexcept { return optimizer_; }

const Adam &LearnedHybridTrainer::optimizer() const noexcept {
  return optimizer_;
}

LearnedTrainingStepMetrics
LearnedHybridTrainer::train_step(const LearnedBatch &batch, std::size_t epoch) {
  if (model_.backend() != optimizer_.backend()) {
    throw std::invalid_argument(
        "learned conditional-reverse model moved after Adam construction");
  }
  const std::size_t expected =
      checked_multiply(batch.batch_size, batch.context_length,
                       "learned conditional-reverse training batch overflows");
  if (batch.batch_size == 0 ||
      batch.context_length != model_.config().protocol.context_length() ||
      batch.inputs.size() != expected || batch.targets.size() != expected) {
    throw std::invalid_argument(
        "learned conditional-reverse training batch has wrong dimensions");
  }
  optimizer_.zero_gradients();
  const LearnedForwardResult forward =
      model_.forward(batch.inputs, batch.batch_size);
  const Variable loss = learned_target_half_loss(
      forward.logits, batch.targets, model_.config().protocol.sequence_length);
  const float loss_value = checked_scalar_loss(loss);
  loss.backward();
  const AdamStepStats adam = optimizer_.step();
  return {
      .step = adam.step,
      .epoch = epoch,
      .target_loss = loss_value,
      .gradient_norm = adam.gradient_norm,
      .clip_scale = adam.clip_scale,
  };
}

LearnedTrainingHistory
LearnedHybridTrainer::fit(const LearnedDataset &training,
                          const LearnedDataset &validation) {
  require_dataset_compatibility(model_, training);
  require_dataset_compatibility(model_, validation);
  config_.validate();

  std::vector<std::size_t> indices(training.size());
  std::iota(indices.begin(), indices.end(), std::size_t{0});
  std::mt19937 random(config_.seed);
  LearnedTrainingHistory history;
  bool stop = false;
  for (std::size_t epoch = 0; epoch < config_.epochs && !stop; ++epoch) {
    if (config_.maximum_steps.has_value() &&
        optimizer_.step_count() >= *config_.maximum_steps) {
      break;
    }
    if (config_.shuffle) {
      std::shuffle(indices.begin(), indices.end(), random);
    }
    for (std::size_t start = 0; start < indices.size();) {
      if (config_.maximum_steps.has_value() &&
          optimizer_.step_count() >= *config_.maximum_steps) {
        stop = true;
        break;
      }
      const std::size_t count =
          std::min(config_.batch_size, indices.size() - start);
      const LearnedBatch batch = indexed_batch(
          training,
          std::span<const std::size_t>(indices).subspan(start, count));
      history.steps.push_back(train_step(batch, epoch + 1));
      start = checked_add(start, count,
                          "learned conditional-reverse training overflows");
    }
    history.epochs.push_back({
        .epoch = epoch + 1,
        .training = evaluate_learned_dataset(model_, training,
                                             config_.evaluation_batch_size)
                        .metrics,
        .validation = evaluate_learned_dataset(model_, validation,
                                               config_.evaluation_batch_size)
                          .metrics,
    });
  }
  return history;
}

} // namespace riftco_transformer::experiments::conditional_reverse
