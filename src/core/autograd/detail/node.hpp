#pragma once

#include "riftco_transformer/core/autograd.hpp"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace riftco_transformer {

// Private graph representation shared only by the autograd implementation
// translation units. Keeping it here preserves one Node definition without
// exposing graph internals in the public header.
struct Variable::Node {
    Tensor value;
    Tensor gradient;
    bool requires_gradient;
    // Distinguishes untouched zero storage from a successful backward that
    // produced an exactly-zero gradient. Optimizers must consume or
    // explicitly clear the latter before exact-resume state is captured.
    bool gradient_pending = false;
    std::uint64_t sequence;
    std::uint64_t value_version = 0;
    std::vector<std::shared_ptr<Node>> parents;
    std::vector<std::uint64_t> parent_value_versions;
    BackwardFunction backward;

    Node(Tensor node_value, bool node_requires_gradient);
};

}  // namespace riftco_transformer
