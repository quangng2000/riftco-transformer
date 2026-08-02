#include "riftco_transformer/experiments/conditional_reverse/task.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace conditional_reverse =
    riftco_transformer::experiments::conditional_reverse;

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Exception, typename Function>
void require_throws_as(Function &&function, const std::string &message) {
  try {
    function();
  } catch (const Exception &) {
    return;
  }
  throw std::runtime_error(message);
}

void test_examples_and_encoding() {
  conditional_reverse::Task task({3, "abcd", "ac", 7});
  const auto reversed = task.make_example("abc");
  require(reversed.reversed && task.decode(reversed.target) == "cba",
          "a trigger prefix must reverse the source");
  const auto copied = task.make_example("bca");
  require(!copied.reversed && task.decode(copied.target) == "bca",
          "a non-trigger prefix must copy the source");
  require(task.decode(task.encode("dcba")) == "dcba",
          "task symbol encoding must round trip");
}

void test_balanced_seeded_generation() {
  conditional_reverse::Task task({4, "abcd", "a", 19});
  const auto first = task.generate_balanced(8);
  const auto second = task.generate_balanced(8);
  require(first.size() == 8 && second.size() == first.size(),
          "balanced generation size");
  for (std::size_t index = 0; index < first.size(); ++index) {
    require(first[index].reversed == (index % 2 == 0),
            "balanced generation must alternate branches");
    require(first[index].source == second[index].source &&
                first[index].target == second[index].target,
            "balanced generation must be deterministic for a fixed seed");
  }
}

void test_validation() {
  require_throws_as<std::invalid_argument>(
      [] { conditional_reverse::TaskConfig{0, "ab", "a", 1}.validate(); },
      "zero length must be rejected");
  require_throws_as<std::invalid_argument>(
      [] { conditional_reverse::TaskConfig{2, "aba", "a", 1}.validate(); },
      "duplicate alphabet symbols must be rejected");
  require_throws_as<std::invalid_argument>(
      [] { conditional_reverse::TaskConfig{2, "ab", "", 1}.validate(); },
      "a missing reverse branch must be rejected");
  require_throws_as<std::invalid_argument>(
      [] { conditional_reverse::TaskConfig{2, "ab", "ab", 1}.validate(); },
      "a missing copy branch must be rejected");

  conditional_reverse::Task task({3, "abc", "a", 1});
  require_throws_as<std::invalid_argument>(
      [&] { static_cast<void>(task.make_example("ab")); },
      "wrong source length must be rejected");
  require_throws_as<std::invalid_argument>(
      [&] { static_cast<void>(task.encode("abz")); },
      "unknown source symbols must be rejected");
}

} // namespace

int main() {
  try {
    test_examples_and_encoding();
    test_balanced_seeded_generation();
    test_validation();
    std::cout << "conditional reverse task tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "conditional reverse task test failure: " << error.what()
              << '\n';
    return 1;
  }
}
