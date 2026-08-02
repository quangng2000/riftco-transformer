#include "riftco_transformer/experiments/conditional_reverse/task.hpp"

#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace riftco_transformer::experiments::conditional_reverse {
namespace {

std::unordered_map<char, SymbolId>
make_symbol_index(std::string_view alphabet) {
  std::unordered_map<char, SymbolId> result;
  result.reserve(alphabet.size());
  for (std::size_t index = 0; index < alphabet.size(); ++index) {
    if (!result.emplace(alphabet[index], index).second) {
      throw std::invalid_argument(
          "conditional-reverse alphabet symbols must be unique");
    }
  }
  return result;
}

} // namespace

void TaskConfig::validate() const {
  if (sequence_length == 0) {
    throw std::invalid_argument(
        "conditional-reverse sequence length must be positive");
  }
  if (alphabet.size() < 2) {
    throw std::invalid_argument(
        "conditional-reverse alphabet needs at least two symbols");
  }
  const auto symbol_index = make_symbol_index(alphabet);
  std::unordered_set<char> seen_reverse_symbols;
  seen_reverse_symbols.reserve(reverse_when_first_is.size());
  for (const char symbol : reverse_when_first_is) {
    if (!symbol_index.contains(symbol)) {
      throw std::invalid_argument(
          "conditional-reverse trigger is absent from the alphabet");
    }
    if (!seen_reverse_symbols.insert(symbol).second) {
      throw std::invalid_argument(
          "conditional-reverse trigger symbols must be unique");
    }
  }
  if (reverse_when_first_is.empty() ||
      reverse_when_first_is.size() == alphabet.size()) {
    throw std::invalid_argument(
        "conditional-reverse task needs both reverse and copy symbols");
  }
  if (sequence_length >
      std::numeric_limits<std::size_t>::max() / alphabet.size()) {
    throw std::overflow_error(
        "conditional-reverse encoded sequence dimension overflows");
  }
}

Task::Task(TaskConfig config) : config_(std::move(config)) {
  config_.validate();
  const auto symbol_index = make_symbol_index(config_.alphabet);
  reverse_symbols_.assign(config_.alphabet.size(), false);
  for (const char symbol : config_.reverse_when_first_is) {
    reverse_symbols_[symbol_index.at(symbol)] = true;
  }
  for (std::size_t symbol = 0; symbol < reverse_symbols_.size(); ++symbol) {
    if (reverse_symbols_[symbol]) {
      reversing_symbol_ids_.push_back(symbol);
    } else {
      copying_symbol_ids_.push_back(symbol);
    }
  }
}

const TaskConfig &Task::config() const noexcept { return config_; }

std::size_t Task::symbol_count() const noexcept {
  return config_.alphabet.size();
}

bool Task::reverses(SymbolId first_symbol) const {
  if (first_symbol >= symbol_count()) {
    throw std::out_of_range(
        "conditional-reverse symbol ID is outside the alphabet");
  }
  return reverse_symbols_[first_symbol];
}

std::vector<SymbolId> Task::encode(std::string_view text) const {
  const auto symbol_index = make_symbol_index(config_.alphabet);
  std::vector<SymbolId> result;
  result.reserve(text.size());
  for (const char symbol : text) {
    const auto found = symbol_index.find(symbol);
    if (found == symbol_index.end()) {
      throw std::invalid_argument(
          "conditional-reverse text contains an unknown symbol");
    }
    result.push_back(found->second);
  }
  return result;
}

std::string Task::decode(std::span<const SymbolId> symbols) const {
  std::string result;
  result.reserve(symbols.size());
  for (const SymbolId symbol : symbols) {
    if (symbol >= symbol_count()) {
      throw std::out_of_range(
          "conditional-reverse symbol ID is outside the alphabet");
    }
    result.push_back(config_.alphabet[symbol]);
  }
  return result;
}

Example Task::make_example(std::span<const SymbolId> source) const {
  if (source.size() != config_.sequence_length) {
    throw std::invalid_argument(
        "conditional-reverse source has the wrong length");
  }
  for (const SymbolId symbol : source) {
    if (symbol >= symbol_count()) {
      throw std::out_of_range(
          "conditional-reverse source symbol is outside the alphabet");
    }
  }

  Example result;
  result.source.assign(source.begin(), source.end());
  result.target = result.source;
  result.reversed = reverses(result.source.front());
  if (result.reversed) {
    std::reverse(result.target.begin(), result.target.end());
  }
  return result;
}

Example Task::make_example(std::string_view source) const {
  const std::vector<SymbolId> encoded = encode(source);
  return make_example(encoded);
}

std::vector<Example> Task::generate_balanced(std::size_t example_count) const {
  if (example_count == 0) {
    throw std::invalid_argument(
        "conditional-reverse generation needs at least one example");
  }

  std::mt19937 random(config_.seed);
  std::uniform_int_distribution<std::size_t> any_symbol(0, symbol_count() - 1);
  std::uniform_int_distribution<std::size_t> reverse_symbol(
      0, reversing_symbol_ids_.size() - 1);
  std::uniform_int_distribution<std::size_t> copy_symbol(
      0, copying_symbol_ids_.size() - 1);

  std::vector<Example> result;
  result.reserve(example_count);
  for (std::size_t example = 0; example < example_count; ++example) {
    std::vector<SymbolId> source(config_.sequence_length);
    const bool reverse_branch = example % 2 == 0;
    source[0] = reverse_branch ? reversing_symbol_ids_[reverse_symbol(random)]
                               : copying_symbol_ids_[copy_symbol(random)];
    for (std::size_t position = 1; position < source.size(); ++position) {
      source[position] = any_symbol(random);
    }
    result.push_back(make_example(source));
  }
  return result;
}

} // namespace riftco_transformer::experiments::conditional_reverse
