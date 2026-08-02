#include "riftco_transformer/lowering/cajal.hpp"

namespace riftco_transformer::lowering {

LoweringAnalysis
analyze_neural_lowering(const compiler::cajal::CompiledProgram &program,
                        const NeuralLoweringConfig &config) {
  return analyze_neural_lowering(program.map(), config);
}

std::unique_ptr<LoweredMultilinearModule>
lower_to_neural(const compiler::cajal::CompiledProgram &program,
                const NeuralLoweringConfig &config) {
  return lower_to_neural(program.map(), config);
}

} // namespace riftco_transformer::lowering
