#pragma once

#include "riftco_transformer/compiler/cajal/interpreter.hpp"
#include "riftco_transformer/compiler/cajal/multilinear_map.hpp"

#include <stdexcept>

namespace riftco_transformer::compiler::cajal {

class CompilationError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

class CompiledProgram {
public:
  [[nodiscard]] const Type &output_type() const noexcept;
  [[nodiscard]] const Context &input_context() const noexcept;
  [[nodiscard]] const MultilinearMap &map() const noexcept;

  // Validates and reorders named values to the compiler's explicit context
  // order before applying the coefficient map.
  [[nodiscard]] EncodedValue apply(const Environment &environment) const;

private:
  Type output_type_;
  Context input_context_;
  MultilinearMap map_;

  CompiledProgram(Type output_type, Context input_context, MultilinearMap map);

  friend CompiledProgram compile(const Expression &, const Context &);
};

// Type-checks expression, preserves ordered_context as the map's input-axis
// order, and recursively lowers the deterministic Cajal-lite subset.
[[nodiscard]] CompiledProgram compile(const Expression &expression,
                                      const Context &ordered_context = {});

} // namespace riftco_transformer::compiler::cajal
