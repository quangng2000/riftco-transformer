#pragma once

#include "riftco_transformer/compiler/cajal/compiler.hpp"
#include "riftco_transformer/lowering/strategy.hpp"

#include <memory>

namespace riftco_transformer::lowering {

[[nodiscard]] LoweringAnalysis
analyze_neural_lowering(const compiler::cajal::CompiledProgram &program,
                        const NeuralLoweringConfig &config = {});

[[nodiscard]] std::unique_ptr<LoweredMultilinearModule>
lower_to_neural(const compiler::cajal::CompiledProgram &program,
                const NeuralLoweringConfig &config = {});

} // namespace riftco_transformer::lowering
