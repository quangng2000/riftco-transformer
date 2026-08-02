#pragma once

#include "riftco_transformer/compiler/cajal/checker.hpp"
#include "riftco_transformer/compiler/cajal/value.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace riftco_transformer::compiler::cajal {

struct NamedValue {
  std::string name;
  Value value;
};

using Environment = std::vector<NamedValue>;

class EvaluationError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// Type-checks first, then evaluates the deterministic reference semantics.
// Product components are suspended until selected by projection. Dictionary
// values are suspended until selected by a lookup, which requires exactly one
// matching key.
[[nodiscard]] Value evaluate(const Expression &expression,
                             const Environment &environment = {});

} // namespace riftco_transformer::compiler::cajal
