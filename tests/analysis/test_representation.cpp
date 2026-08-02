#include "riftco_transformer/analysis/analysis.hpp"

#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace analysis = riftco_transformer::analysis;

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Exception, typename Function>
void require_throws(Function &&function, const std::string &message) {
  bool threw_expected = false;
  try {
    function();
  } catch (const Exception &) {
    threw_expected = true;
  }
  require(threw_expected, message);
}

void capture(analysis::RepresentationTrace &trace, std::string name,
             std::vector<std::size_t> shape, std::vector<float> values) {
  trace.capture(std::move(name), shape, values);
}

void test_ordered_owned_flattened_capture() {
  analysis::RepresentationTrace trace;
  std::vector<float> values(24);
  for (std::size_t index = 0; index < values.size(); ++index) {
    values[index] = static_cast<float>(index);
  }
  const std::vector<std::size_t> first_shape{2, 3, 4};
  trace.capture("blocks.0.residual", first_shape, values);
  values.front() = -100.0F;
  capture(trace, "unbatched", {4}, {1.0F, 2.0F, 3.0F, 4.0F});
  trace.capture({
      "preflattened",
      {2},
      {2, 2, {5.0F, 6.0F, 7.0F, 8.0F}},
  });

  require(trace.contains("blocks.0.residual"), "trace contains first");
  require(trace.contains("unbatched"), "trace contains second");
  require(!trace.contains("missing"), "trace missing lookup");
  const auto entries = trace.entries();
  require(entries.size() == 3, "trace entry count");
  require(entries[0].name == "blocks.0.residual" &&
              entries[1].name == "unbatched" &&
              entries[2].name == "preflattened",
          "trace preserves capture order");

  const analysis::NamedRepresentation &first = trace.at("blocks.0.residual");
  require(first.leading_shape == std::vector<std::size_t>({2, 3}),
          "capture preserves leading shape");
  require(first.observations.rows == 6 && first.observations.columns == 4 &&
              first.observations.values.size() == 24,
          "capture flattens leading axes");
  require(first.observations.values.front() == 0.0F, "capture owns values");
  require(first.observations.values.back() == 23.0F,
          "capture preserves row-major flattening order");

  const auto &unbatched = trace.at("unbatched");
  require(unbatched.leading_shape.empty() && unbatched.observations.rows == 1 &&
              unbatched.observations.columns == 4,
          "feature-only shape becomes one observation");
}

void test_capture_validation_and_transactionality() {
  analysis::RepresentationTrace trace;
  capture(trace, "valid", {1, 2}, {1.0F, 2.0F});
  require_throws<std::invalid_argument>(
      [&] { capture(trace, "valid", {1, 2}, {3.0F, 4.0F}); },
      "trace rejects duplicate names");
  require(trace.entries().size() == 1, "duplicate capture is transactional");
  require_throws<std::invalid_argument>(
      [&] { capture(trace, "", {1}, {1.0F}); }, "trace rejects empty names");
  require_throws<std::invalid_argument>(
      [&] { capture(trace, "no_shape", {}, {1.0F}); },
      "trace rejects missing feature axis");
  require_throws<std::invalid_argument>(
      [&] { capture(trace, "zero_leading", {2, 0, 3}, {}); },
      "trace rejects zero leading dimensions");
  require_throws<std::invalid_argument>(
      [&] { capture(trace, "zero_features", {2, 0}, {}); },
      "trace rejects zero feature count");
  require_throws<std::invalid_argument>(
      [&] { capture(trace, "wrong_values", {2, 2}, {1.0F, 2.0F}); },
      "trace rejects mismatched values");
  require_throws<std::overflow_error>(
      [&] {
        capture(trace, "overflow",
                {std::numeric_limits<std::size_t>::max(), 2, 1}, {});
      },
      "trace rejects leading-shape overflow");
  require_throws<std::invalid_argument>(
      [&] {
        capture(trace, "nonfinite", {1},
                {
                    std::numeric_limits<float>::infinity(),
                });
      },
      "trace rejects nonfinite observations");
  require_throws<std::invalid_argument>(
      [&] {
        trace.capture({
            "bad_flattening",
            {3},
            {2, 2, {1.0F, 2.0F, 3.0F, 4.0F}},
        });
      },
      "trace validates preflattened leading rows");
  require_throws<std::out_of_range>(
      [&] { static_cast<void>(trace.at("missing")); },
      "trace rejects unknown lookup");
  require(trace.entries().size() == 1, "failed captures leave trace intact");
}

} // namespace

int main() {
  try {
    test_ordered_owned_flattened_capture();
    test_capture_validation_and_transactionality();
    std::cout << "Analysis representation tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Analysis representation test failure: " << error.what()
              << '\n';
    return 1;
  }
}
