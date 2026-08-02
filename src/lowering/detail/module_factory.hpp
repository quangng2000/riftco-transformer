#pragma once

#include "riftco_transformer/lowering/strategy.hpp"

#include <memory>

namespace riftco_transformer::lowering::detail {

[[nodiscard]] std::unique_ptr<LoweredMultilinearModule>
make_builtin_module(const compiler::cajal::MultilinearMap &map,
                    const NeuralLoweringConfig &config,
                    const LoweringAnalysis &analysis);

} // namespace riftco_transformer::lowering::detail
