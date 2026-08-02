#include "riftco_transformer/experiments/conditional_reverse/learned.hpp"

int main() {
  namespace conditional_reverse =
      riftco_transformer::experiments::conditional_reverse;

  conditional_reverse::LearnedProtocolConfig protocol;
  protocol.sequence_length = 2;
  protocol.alphabet = "ab";
  protocol.reverse_when_first_is = "a";
  protocol.seed = 7;

  const conditional_reverse::LearnedSplitSizes sizes{
      .train = 2,
      .probe = 2,
      .validation = 2,
      .test = 2,
  };
  const auto datasets =
      conditional_reverse::generate_learned_datasets(protocol, sizes);
  return datasets.test.size() == sizes.test ? 0 : 1;
}
