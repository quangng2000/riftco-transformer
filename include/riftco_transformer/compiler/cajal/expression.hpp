#pragma once

#include "riftco_transformer/compiler/cajal/type.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace riftco_transformer::compiler::cajal {

// An immutable, programmatically constructed Cajal-lite expression tree.
// Parsing is deliberately postponed until the checker and compiler semantics
// are stable.
class Expression {
public:
  enum class Kind : std::uint8_t {
    Variable,
    Unit,
    Tuple,
    InjectLeft,
    InjectRight,
    Dictionary,
    Sequence,
    Projection,
    Case,
    Let,
    Lookup,
  };

  using DictionaryEntries =
      std::vector<std::pair<Expression, Expression>>;

  [[nodiscard]] static Expression variable(std::string name);
  [[nodiscard]] static Expression unit();
  [[nodiscard]] static Expression tuple(Expression first, Expression second);
  [[nodiscard]] static Expression inject_left(Expression payload,
                                              Type right_type);
  [[nodiscard]] static Expression inject_right(Type left_type,
                                               Expression payload);
  [[nodiscard]] static Expression dictionary(DictionaryEntries entries);
  [[nodiscard]] static Expression sequence(Expression first,
                                           Expression second);
  [[nodiscard]] static Expression projection(Expression value,
                                             std::size_t index);
  [[nodiscard]] static Expression case_of(Expression scrutinee,
                                          std::string left_name,
                                          Expression left_body,
                                          std::string right_name,
                                          Expression right_body);
  [[nodiscard]] static Expression let(std::string name, Expression bound,
                                      Expression body);
  [[nodiscard]] static Expression lookup(Expression dictionary,
                                         Expression query);

  Expression(const Expression &) noexcept = default;
  Expression &operator=(const Expression &) noexcept = default;
  Expression(Expression &&other) noexcept;
  Expression &operator=(Expression &&other) noexcept;
  ~Expression();

  [[nodiscard]] Kind kind() const noexcept;
  [[nodiscard]] const std::string &variable_name() const;

  [[nodiscard]] const Expression &first() const;
  [[nodiscard]] const Expression &second() const;
  [[nodiscard]] const Expression &payload() const;
  [[nodiscard]] const Type &other_type() const;

  [[nodiscard]] std::size_t dictionary_size() const;
  [[nodiscard]] const Expression &dictionary_key(std::size_t index) const;
  [[nodiscard]] const Expression &dictionary_value(std::size_t index) const;

  [[nodiscard]] const Expression &projected_value() const;
  [[nodiscard]] std::size_t projection_index() const;

  [[nodiscard]] const Expression &scrutinee() const;
  [[nodiscard]] const std::string &left_binding_name() const;
  [[nodiscard]] const Expression &left_body() const;
  [[nodiscard]] const std::string &right_binding_name() const;
  [[nodiscard]] const Expression &right_body() const;

  [[nodiscard]] const std::string &binding_name() const;
  [[nodiscard]] const Expression &bound_expression() const;
  [[nodiscard]] const Expression &body() const;

  [[nodiscard]] const Expression &lookup_dictionary() const;
  [[nodiscard]] const Expression &lookup_query() const;

private:
  struct Node;
  std::shared_ptr<const Node> node_;

  explicit Expression(std::shared_ptr<const Node> node);
};

} // namespace riftco_transformer::compiler::cajal
