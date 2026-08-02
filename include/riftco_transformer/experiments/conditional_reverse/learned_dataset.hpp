#pragma once

#include "riftco_transformer/data/tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace riftco_transformer::experiments::conditional_reverse {

// Teacher-forced protocol used by the paper experiment. A source s and target
// f(s) become tokens [s, delimiter, f(s)], inputs tokens[:-1], and targets
// tokens[1:]. Only target positions [sequence_length, 2 * sequence_length)
// contribute to the training loss and reported accuracy.
struct LearnedProtocolConfig {
  std::size_t sequence_length = 15;
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  std::string reverse_when_first_is = "aeiou";
  char delimiter = '|';
  std::uint32_t seed = 42;

  void validate() const;
  [[nodiscard]] std::size_t vocabulary_size() const noexcept;
  [[nodiscard]] TokenId delimiter_token() const;
  [[nodiscard]] std::size_t context_length() const;
};

struct LearnedSplitSizes {
  std::size_t train = 10'000;
  std::size_t probe = 5'000;
  std::size_t validation = 1'000;
  std::size_t test = 1'000;
  // False matches the paper artifact's independent random examples. The
  // quick lab enables this to make its small splits genuinely source-disjoint.
  bool disjoint_sources = false;

  void validate() const;
};

struct LearnedExample {
  // Complete [source, delimiter, target] protocol sequence, length 2L + 1.
  std::vector<TokenId> tokens;
  // Teacher-forced views, each length 2L.
  std::vector<TokenId> inputs;
  std::vector<TokenId> targets;
  bool reversed = false;
};

class LearnedDataset {
public:
  LearnedDataset(LearnedProtocolConfig config,
                 std::vector<LearnedExample> examples);

  [[nodiscard]] const LearnedProtocolConfig &config() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] bool empty() const noexcept;
  [[nodiscard]] const LearnedExample &operator[](std::size_t index) const;
  [[nodiscard]] std::span<const LearnedExample> examples() const noexcept;

private:
  LearnedProtocolConfig config_;
  std::vector<LearnedExample> examples_;
};

struct LearnedDatasetSplits {
  LearnedDataset train;
  LearnedDataset probe;
  LearnedDataset validation;
  LearnedDataset test;
};

struct LearnedBatch {
  std::size_t batch_size = 0;
  std::size_t context_length = 0;
  std::vector<TokenId> inputs;
  std::vector<TokenId> targets;
  std::vector<std::uint8_t> reversed;
};

// By default all four splits consume one seeded random stream in
// train/probe/validation/test order, matching the paper protocol. With
// disjoint_sources, one without-replacement sample is partitioned in that
// order.
[[nodiscard]] LearnedDatasetSplits
generate_learned_datasets(const LearnedProtocolConfig &config = {},
                          const LearnedSplitSizes &sizes = {});

// Generates an alternating reverse/copy analysis set. It is separate from the
// ordinary split protocol because the paper uses a balanced steering probe.
[[nodiscard]] std::vector<LearnedExample>
generate_balanced_learned_examples(const LearnedProtocolConfig &config,
                                   std::size_t count, std::uint32_t seed);

[[nodiscard]] LearnedBatch
make_learned_batch(std::span<const LearnedExample> examples,
                   const LearnedProtocolConfig &config);

} // namespace riftco_transformer::experiments::conditional_reverse
