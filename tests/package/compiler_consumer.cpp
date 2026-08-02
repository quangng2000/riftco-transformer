#include "riftco_transformer/compiler/cajal/cajal.hpp"

#include <array>
#include <cstdlib>

int main() {
  namespace cajal = riftco_transformer::compiler::cajal;

  const cajal::MultilinearMap identity = cajal::MultilinearMap::identity(2);
  const std::array inputs{cajal::EncodedValue({2.0, -3.0})};
  const cajal::EncodedValue output = identity.apply(inputs);
  return output.size() == 2 && output.coordinates()[0] == 2.0 &&
                 output.coordinates()[1] == -3.0
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
