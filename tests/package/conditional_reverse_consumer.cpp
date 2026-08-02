#include "riftco_transformer/experiments/conditional_reverse/conditional_reverse.hpp"

#include <vector>

int main() {
  namespace conditional_reverse =
      riftco_transformer::experiments::conditional_reverse;
  conditional_reverse::CircuitConfig config;
  config.task = {
      .sequence_length = 2,
      .alphabet = "ab",
      .reverse_when_first_is = "a",
      .seed = 3,
  };
  conditional_reverse::Circuit circuit(config);
  const std::vector<conditional_reverse::Example> examples{
      circuit.task().make_example("ab"),
      circuit.task().make_example("ba"),
  };
  const auto result = circuit.evaluate(examples);
  return result.metrics.exact_sequence_accuracy == 1.0 ? 0 : 1;
}
