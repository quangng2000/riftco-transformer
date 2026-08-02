#include "riftco_transformer/analysis/analysis.hpp"

#include <vector>

int main() {
  const riftco_transformer::analysis::Matrix observations{
      3,
      2,
      std::vector<float>{-1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F},
  };
  riftco_transformer::analysis::PcaOptions options;
  options.component_count = 1;
  const auto fit =
      riftco_transformer::analysis::fit_pca(observations.view(), options);
  return fit.model.component_count == 1 && fit.scores.rows == 3 ? 0 : 1;
}
