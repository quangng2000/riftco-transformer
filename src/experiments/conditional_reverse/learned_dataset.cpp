#include "riftco_transformer/experiments/conditional_reverse/learned_dataset.hpp"

#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_set>
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

[[nodiscard]] std::vector<std::uint8_t>
reverse_lookup(const LearnedProtocolConfig &config) {
  std::vector<std::uint8_t> result(config.alphabet.size(), 0);
  for (const char trigger : config.reverse_when_first_is) {
    const std::size_t index = config.alphabet.find(trigger);
    result[index] = 1;
  }
  return result;
}

[[nodiscard]] LearnedExample
generate_example(const LearnedProtocolConfig &config,
                 std::span<const std::uint8_t> reverses,
                 std::span<const TokenId> source) {
  std::vector<TokenId> target(source.begin(), source.end());
  const bool reversed = reverses[source.front()];
  if (reversed) {
    std::reverse(target.begin(), target.end());
  }

  LearnedExample result;
  result.reversed = reversed;
  result.tokens.reserve(config.context_length() + 1);
  result.tokens.insert(result.tokens.end(), source.begin(), source.end());
  result.tokens.push_back(config.delimiter_token());
  result.tokens.insert(result.tokens.end(), target.begin(), target.end());
  result.inputs.assign(result.tokens.begin(), result.tokens.end() - 1);
  result.targets.assign(result.tokens.begin() + 1, result.tokens.end());
  return result;
}

[[nodiscard]] LearnedExample
generate_random_example(const LearnedProtocolConfig &config,
                        std::span<const std::uint8_t> reverses,
                        std::mt19937 &random) {
  std::uniform_int_distribution<std::size_t> symbol_distribution(
      0, config.alphabet.size() - 1);
  std::vector<TokenId> source(config.sequence_length);
  for (TokenId &token : source) {
    token = static_cast<TokenId>(symbol_distribution(random));
  }
  return generate_example(config, reverses, source);
}

[[nodiscard]] LearnedDataset
generate_split(const LearnedProtocolConfig &config,
               std::span<const std::uint8_t> reverses, std::size_t count,
               std::mt19937 &random) {
  std::vector<LearnedExample> examples;
  examples.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    examples.push_back(generate_random_example(config, reverses, random));
  }
  return LearnedDataset(config, std::move(examples));
}

[[nodiscard]] std::size_t
checked_source_space(const LearnedProtocolConfig &config) {
  std::size_t result = 1;
  for (std::size_t position = 0; position < config.sequence_length;
       ++position) {
    result =
        checked_multiply(result, config.alphabet.size(),
                         "learned conditional-reverse source space overflows");
  }
  return result;
}

[[nodiscard]] std::vector<TokenId>
decode_source_index(std::size_t index, const LearnedProtocolConfig &config) {
  std::vector<TokenId> source(config.sequence_length);
  for (std::size_t position = config.sequence_length; position > 0;
       --position) {
    source[position - 1] = static_cast<TokenId>(index % config.alphabet.size());
    index /= config.alphabet.size();
  }
  return source;
}

[[nodiscard]] std::vector<LearnedExample>
generate_disjoint_examples(const LearnedProtocolConfig &config,
                           std::span<const std::uint8_t> reverses,
                           std::size_t count, std::mt19937 &random) {
  const std::size_t source_space = checked_source_space(config);
  if (count > source_space) {
    throw std::invalid_argument(
        "learned conditional-reverse disjoint splits exceed the source "
        "space");
  }

  // Floyd's algorithm samples without replacement in O(count) storage even
  // when the finite source space is much larger than the requested splits.
  std::unordered_set<std::size_t> selected;
  selected.reserve(count);
  std::vector<std::size_t> source_indices;
  source_indices.reserve(count);
  for (std::size_t upper = source_space - count; upper < source_space;
       ++upper) {
    std::uniform_int_distribution<std::size_t> distribution(0, upper);
    const std::size_t candidate = distribution(random);
    const std::size_t chosen = selected.contains(candidate) ? upper : candidate;
    selected.insert(chosen);
    source_indices.push_back(chosen);
  }
  std::shuffle(source_indices.begin(), source_indices.end(), random);

  std::vector<LearnedExample> examples;
  examples.reserve(count);
  for (const std::size_t source_index : source_indices) {
    const std::vector<TokenId> source =
        decode_source_index(source_index, config);
    examples.push_back(generate_example(config, reverses, source));
  }
  return examples;
}

[[nodiscard]] LearnedDataset
take_examples(const LearnedProtocolConfig &config,
              std::vector<LearnedExample> &examples, std::size_t &offset,
              std::size_t count) {
  std::vector<LearnedExample> split;
  split.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    split.push_back(std::move(examples[offset++]));
  }
  return LearnedDataset(config, std::move(split));
}

void validate_example(const LearnedExample &example,
                      const LearnedProtocolConfig &config) {
  const std::size_t context_length = config.context_length();
  if (example.tokens.size() != context_length + 1 ||
      example.inputs.size() != context_length ||
      example.targets.size() != context_length) {
    throw std::invalid_argument(
        "learned conditional-reverse example has the wrong length");
  }
  if (example.tokens[config.sequence_length] != config.delimiter_token()) {
    throw std::invalid_argument(
        "learned conditional-reverse example is missing its delimiter");
  }
  if (!std::equal(example.inputs.begin(), example.inputs.end(),
                  example.tokens.begin()) ||
      !std::equal(example.targets.begin(), example.targets.end(),
                  example.tokens.begin() + 1)) {
    throw std::invalid_argument(
        "learned conditional-reverse teacher-forced views are inconsistent");
  }
  for (const TokenId token : example.tokens) {
    if (static_cast<std::size_t>(token) >= config.vocabulary_size()) {
      throw std::out_of_range(
          "learned conditional-reverse token is outside the vocabulary");
    }
  }
  for (std::size_t position = 0; position < config.sequence_length;
       ++position) {
    const TokenId source = example.tokens[position];
    const TokenId output =
        example.tokens[config.sequence_length + 1 + position];
    if (static_cast<std::size_t>(source) >= config.alphabet.size() ||
        static_cast<std::size_t>(output) >= config.alphabet.size()) {
      throw std::invalid_argument(
          "learned conditional-reverse delimiter may appear only between "
          "source and output");
    }
  }
  const char first_symbol = config.alphabet[example.tokens.front()];
  const bool expected_reversed =
      config.reverse_when_first_is.find(first_symbol) != std::string::npos;
  if (example.reversed != expected_reversed) {
    throw std::invalid_argument(
        "learned conditional-reverse branch flag disagrees with source");
  }
  for (std::size_t position = 0; position < config.sequence_length;
       ++position) {
    const std::size_t source_position =
        expected_reversed ? config.sequence_length - 1 - position : position;
    if (example.tokens[config.sequence_length + 1 + position] !=
        example.tokens[source_position]) {
      throw std::invalid_argument(
          "learned conditional-reverse output violates task semantics");
    }
  }
}

} // namespace

void LearnedProtocolConfig::validate() const {
  if (sequence_length == 0) {
    throw std::invalid_argument(
        "learned conditional-reverse sequence length must be positive");
  }
  if (alphabet.size() < 2) {
    throw std::invalid_argument(
        "learned conditional-reverse alphabet needs at least two symbols");
  }
  if (alphabet.size() >=
      static_cast<std::size_t>(std::numeric_limits<TokenId>::max())) {
    throw std::overflow_error(
        "learned conditional-reverse vocabulary exceeds TokenId");
  }
  std::unordered_set<char> symbols;
  symbols.reserve(alphabet.size());
  for (const char symbol : alphabet) {
    if (symbol == delimiter) {
      throw std::invalid_argument(
          "learned conditional-reverse delimiter must not be in alphabet");
    }
    if (!symbols.insert(symbol).second) {
      throw std::invalid_argument(
          "learned conditional-reverse alphabet symbols must be unique");
    }
  }
  std::unordered_set<char> triggers;
  triggers.reserve(reverse_when_first_is.size());
  for (const char trigger : reverse_when_first_is) {
    if (!symbols.contains(trigger)) {
      throw std::invalid_argument(
          "learned conditional-reverse trigger is absent from alphabet");
    }
    if (!triggers.insert(trigger).second) {
      throw std::invalid_argument(
          "learned conditional-reverse triggers must be unique");
    }
  }
  if (triggers.empty() || triggers.size() == symbols.size()) {
    throw std::invalid_argument(
        "learned conditional-reverse needs reverse and copy symbols");
  }
  static_cast<void>(
      checked_add(context_length(), 1,
                  "learned conditional-reverse protocol length overflows"));
}

std::size_t LearnedProtocolConfig::vocabulary_size() const noexcept {
  return alphabet.size() + 1;
}

TokenId LearnedProtocolConfig::delimiter_token() const {
  if (alphabet.size() >
      static_cast<std::size_t>(std::numeric_limits<TokenId>::max())) {
    throw std::overflow_error(
        "learned conditional-reverse delimiter exceeds TokenId");
  }
  return static_cast<TokenId>(alphabet.size());
}

std::size_t LearnedProtocolConfig::context_length() const {
  return checked_multiply(sequence_length, 2,
                          "learned conditional-reverse context overflows");
}

void LearnedSplitSizes::validate() const {
  if (train == 0 || probe == 0 || validation == 0 || test == 0) {
    throw std::invalid_argument(
        "learned conditional-reverse splits must all be nonempty");
  }
  const std::size_t first = checked_add(
      train, probe, "learned conditional-reverse split count overflows");
  const std::size_t second = checked_add(
      validation, test, "learned conditional-reverse split count overflows");
  static_cast<void>(checked_add(
      first, second, "learned conditional-reverse split count overflows"));
}

LearnedDataset::LearnedDataset(LearnedProtocolConfig config,
                               std::vector<LearnedExample> examples)
    : config_(std::move(config)), examples_(std::move(examples)) {
  config_.validate();
  for (const LearnedExample &example : examples_) {
    validate_example(example, config_);
  }
}

const LearnedProtocolConfig &LearnedDataset::config() const noexcept {
  return config_;
}

std::size_t LearnedDataset::size() const noexcept { return examples_.size(); }

bool LearnedDataset::empty() const noexcept { return examples_.empty(); }

const LearnedExample &LearnedDataset::operator[](std::size_t index) const {
  return examples_.at(index);
}

std::span<const LearnedExample> LearnedDataset::examples() const noexcept {
  return examples_;
}

LearnedDatasetSplits
generate_learned_datasets(const LearnedProtocolConfig &config,
                          const LearnedSplitSizes &sizes) {
  config.validate();
  sizes.validate();
  const std::size_t first =
      checked_add(sizes.train, sizes.probe,
                  "learned conditional-reverse split count overflows");
  const std::size_t second =
      checked_add(sizes.validation, sizes.test,
                  "learned conditional-reverse split count overflows");
  const std::size_t total = checked_add(
      first, second, "learned conditional-reverse split count overflows");
  const std::size_t protocol_token_count =
      checked_add(config.context_length(), 1,
                  "learned conditional-reverse protocol length overflows");
  static_cast<void>(checked_multiply(
      total, protocol_token_count,
      "learned conditional-reverse generated token count overflows"));
  static_cast<void>(checked_multiply(
      total, sizeof(LearnedExample),
      "learned conditional-reverse generated example storage overflows"));
  const std::vector<std::uint8_t> reverses = reverse_lookup(config);
  std::mt19937 random(config.seed);
  if (sizes.disjoint_sources) {
    std::vector<LearnedExample> examples =
        generate_disjoint_examples(config, reverses, total, random);
    std::size_t offset = 0;
    LearnedDataset train = take_examples(config, examples, offset, sizes.train);
    LearnedDataset probe = take_examples(config, examples, offset, sizes.probe);
    LearnedDataset validation =
        take_examples(config, examples, offset, sizes.validation);
    LearnedDataset test = take_examples(config, examples, offset, sizes.test);
    return {std::move(train), std::move(probe), std::move(validation),
            std::move(test)};
  }
  LearnedDataset train = generate_split(config, reverses, sizes.train, random);
  LearnedDataset probe = generate_split(config, reverses, sizes.probe, random);
  LearnedDataset validation =
      generate_split(config, reverses, sizes.validation, random);
  LearnedDataset test = generate_split(config, reverses, sizes.test, random);
  return {std::move(train), std::move(probe), std::move(validation),
          std::move(test)};
}

std::vector<LearnedExample>
generate_balanced_learned_examples(const LearnedProtocolConfig &config,
                                   std::size_t count, std::uint32_t seed) {
  config.validate();
  if (count == 0) {
    throw std::invalid_argument(
        "learned conditional-reverse balanced set must be nonempty");
  }
  const std::size_t protocol_token_count =
      checked_add(config.context_length(), 1,
                  "learned conditional-reverse protocol length overflows");
  static_cast<void>(checked_multiply(
      count, protocol_token_count,
      "learned conditional-reverse balanced token count overflows"));

  const std::vector<std::uint8_t> reverses = reverse_lookup(config);
  std::vector<TokenId> reverse_symbols;
  std::vector<TokenId> copy_symbols;
  for (std::size_t symbol = 0; symbol < reverses.size(); ++symbol) {
    (reverses[symbol] != 0 ? reverse_symbols : copy_symbols)
        .push_back(static_cast<TokenId>(symbol));
  }

  std::mt19937 random(seed);
  std::uniform_int_distribution<std::size_t> symbol_distribution(
      0, config.alphabet.size() - 1);
  std::uniform_int_distribution<std::size_t> reverse_distribution(
      0, reverse_symbols.size() - 1);
  std::uniform_int_distribution<std::size_t> copy_distribution(
      0, copy_symbols.size() - 1);
  std::vector<LearnedExample> examples;
  examples.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const bool reverse = index % 2 == 0;
    std::vector<TokenId> source(config.sequence_length);
    source.front() = reverse ? reverse_symbols[reverse_distribution(random)]
                             : copy_symbols[copy_distribution(random)];
    for (std::size_t position = 1; position < source.size(); ++position) {
      source[position] = static_cast<TokenId>(symbol_distribution(random));
    }
    examples.push_back(generate_example(config, reverses, source));
  }
  return examples;
}

LearnedBatch make_learned_batch(std::span<const LearnedExample> examples,
                                const LearnedProtocolConfig &config) {
  config.validate();
  if (examples.empty()) {
    throw std::invalid_argument(
        "learned conditional-reverse batch must be nonempty");
  }
  const std::size_t context_length = config.context_length();
  const std::size_t element_count = checked_multiply(
      examples.size(), context_length,
      "learned conditional-reverse batch element count overflows");
  LearnedBatch result;
  result.batch_size = examples.size();
  result.context_length = context_length;
  result.inputs.reserve(element_count);
  result.targets.reserve(element_count);
  result.reversed.reserve(examples.size());
  for (const LearnedExample &example : examples) {
    validate_example(example, config);
    result.inputs.insert(result.inputs.end(), example.inputs.begin(),
                         example.inputs.end());
    result.targets.insert(result.targets.end(), example.targets.begin(),
                          example.targets.end());
    result.reversed.push_back(example.reversed ? 1 : 0);
  }
  return result;
}

} // namespace riftco_transformer::experiments::conditional_reverse
