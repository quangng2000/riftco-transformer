#pragma once

#include "riftco_transformer/compiler/cajal/compiler.hpp"
#include "riftco_transformer/compiler/cajal/expression.hpp"
#include "riftco_transformer/compiler/cajal/type.hpp"
#include "riftco_transformer/compiler/cajal/value.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace riftco_transformer::experiments::conditional_reverse {

inline constexpr char kConditionInputName[] = "condition";
inline constexpr char kSequenceInputName[] = "sequence";
inline constexpr char kFirstSequenceInputName[] = "xs1";
inline constexpr char kSecondSequenceInputName[] = "xs2";
inline constexpr std::size_t kDefaultMaximumCoefficientElements = 1U << 24U;

class ProgramError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

enum class Condition : std::uint8_t {
  Reverse,
  Copy,
};

struct ProgramConfig {
  std::size_t sequence_length = 4;
  std::size_t symbol_count = 2;
  std::size_t max_coefficient_elements = kDefaultMaximumCoefficientElements;

  void validate() const;
};

struct ResourceMetadata {
  std::size_t condition_dimension = 2;
  std::size_t symbol_dimension = 0;
  std::size_t sequence_dimension = 0;
  std::size_t output_dimension = 0;
  std::array<std::size_t, 2> input_dimensions{};
  std::size_t coefficient_elements = 0;
  std::size_t nonzero_coefficient_elements = 0;
  std::size_t dense_coefficient_bytes = 0;
};

struct ReverseResourceMetadata {
  std::size_t symbol_dimension = 0;
  std::size_t sequence_dimension = 0;
  std::size_t output_dimension = 0;
  std::array<std::size_t, 1> input_dimensions{};
  std::size_t coefficient_elements = 0;
  std::size_t nonzero_coefficient_elements = 0;
  std::size_t dense_coefficient_bytes = 0;
};

struct TwoSequenceResourceMetadata {
  std::size_t symbol_dimension = 0;
  std::size_t sequence_dimension = 0;
  std::size_t output_dimension = 0;
  std::array<std::size_t, 2> input_dimensions{};
  std::size_t coefficient_elements = 0;
  std::size_t nonzero_coefficient_elements = 0;
  std::size_t dense_coefficient_bytes = 0;
};

// Retains the inspectable source program beside its checked dense compilation.
// The context order is condition first and sequence second, matching the
// bilinear map axes and the neural lowering's default query-axis convention.
struct Program {
  ProgramConfig config;
  ResourceMetadata resources;
  compiler::cajal::Type condition_type;
  compiler::cajal::Type symbol_type;
  compiler::cajal::Type sequence_type;
  compiler::cajal::Context input_context;
  compiler::cajal::Expression expression;
  compiler::cajal::CompiledProgram compiled;
};

// The paper's P control: a unary full-sequence reversal map. Keeping this
// separate from Program preserves the compact condition/sequence API used by
// the exact circuit while exposing a linear-lowerable control program.
struct ReverseProgram {
  ProgramConfig config;
  ReverseResourceMetadata resources;
  compiler::cajal::Type symbol_type;
  compiler::cajal::Type sequence_type;
  compiler::cajal::Context input_context;
  compiler::cajal::Expression expression;
  compiler::cajal::CompiledProgram compiled;
};

// The paper-faithful conditional program. Both inputs are complete sequences.
// The first symbol of xs1 controls whether xs2 is reversed or copied.
struct TwoSequenceProgram {
  ProgramConfig config;
  TwoSequenceResourceMetadata resources;
  compiler::cajal::Type symbol_type;
  compiler::cajal::Type sequence_type;
  compiler::cajal::Context input_context;
  compiler::cajal::Expression expression;
  compiler::cajal::CompiledProgram compiled;
};

[[nodiscard]] compiler::cajal::Type make_condition_type();
[[nodiscard]] compiler::cajal::Type make_symbol_type(std::size_t symbol_count);
[[nodiscard]] compiler::cajal::Type
make_sequence_type(std::size_t symbol_count, std::size_t sequence_length);

// Reverse is the left Sum injection; Copy is the right Sum injection.
[[nodiscard]] compiler::cajal::Value make_condition_value(Condition condition);
[[nodiscard]] compiler::cajal::Value
make_symbol_value(std::size_t symbol_count, std::size_t symbol_index);
[[nodiscard]] compiler::cajal::Value
make_sequence_value(std::size_t symbol_count,
                    std::span<const std::size_t> symbol_indices);

// Computes every dense-map size with overflow checks and enforces the
// configured coefficient limit before any recursive type or AST is built.
[[nodiscard]] ResourceMetadata analyze_resources(const ProgramConfig &config);

// Preflights the unary reversal map [L*K] -> [L*K].
[[nodiscard]] ReverseResourceMetadata
analyze_reverse_resources(const ProgramConfig &config);

// Preflights the full conditional map [L*K] x [L*K] -> [L*K].
[[nodiscard]] TwoSequenceResourceMetadata
analyze_two_sequence_resources(const ProgramConfig &config);

// Builds, type-checks, and compiles the exact program. On discrete values the
// left condition reverses the sequence and the right condition copies it.
[[nodiscard]] Program compile_program(const ProgramConfig &config = {});

// Builds the exact unary reversal program used by the paper's P control.
[[nodiscard]] ReverseProgram
compile_reverse_program(const ProgramConfig &config = {});

// Builds the exact full conditional program. Symbol coordinate zero at xs1's
// first position selects reverse(xs2); every other symbol selects xs2.
[[nodiscard]] TwoSequenceProgram
compile_two_sequence_program(const ProgramConfig &config = {});

} // namespace riftco_transformer::experiments::conditional_reverse
