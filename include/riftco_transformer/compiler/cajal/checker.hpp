#pragma once

#include "riftco_transformer/compiler/cajal/expression.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace riftco_transformer::compiler::cajal {

struct Binding {
  std::string name;
  Type type;
};

using Context = std::vector<Binding>;

class TypeError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// Checks linear resource use. Multiplicative constructs split their contexts;
// additive products and case paths reuse the same context because only one
// projection or branch is observed. Every top-level binding must participate
// in the checked expression.
[[nodiscard]] Type type_check(const Expression &expression,
                              const Context &context = {});

} // namespace riftco_transformer::compiler::cajal
