#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace riftco_transformer::experiments::conditional_reverse {

using SymbolId = std::size_t;

struct TaskConfig {
  std::size_t sequence_length = 15;
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  std::string reverse_when_first_is = "aeiou";
  std::uint32_t seed = 42;

  void validate() const;
};

struct Example {
  std::vector<SymbolId> source;
  std::vector<SymbolId> target;
  bool reversed = false;
};

// Deterministic benchmark data for the conditional-reversal circuit. Examples
// are balanced by branch: reverse, copy, reverse, copy, ... . The condition is
// intentionally task metadata rather than compiler metadata; a learned input
// projection converts the first symbol into the program's two condition
// coordinates.
class Task {
public:
  explicit Task(TaskConfig config = {});

  [[nodiscard]] const TaskConfig &config() const noexcept;
  [[nodiscard]] std::size_t symbol_count() const noexcept;
  [[nodiscard]] bool reverses(SymbolId first_symbol) const;

  [[nodiscard]] std::vector<SymbolId> encode(std::string_view text) const;
  [[nodiscard]] std::string decode(std::span<const SymbolId> symbols) const;
  [[nodiscard]] Example make_example(std::span<const SymbolId> source) const;
  [[nodiscard]] Example make_example(std::string_view source) const;

  [[nodiscard]] std::vector<Example>
  generate_balanced(std::size_t example_count) const;

private:
  TaskConfig config_;
  std::vector<bool> reverse_symbols_;
  std::vector<SymbolId> reversing_symbol_ids_;
  std::vector<SymbolId> copying_symbol_ids_;
};

} // namespace riftco_transformer::experiments::conditional_reverse
