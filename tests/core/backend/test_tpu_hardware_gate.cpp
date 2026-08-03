#include "riftco_transformer/core/backend.hpp"

#include <iostream>
#include <string>
#include <string_view>

int main() {
    using riftco_transformer::ExecutionBackend;
    if (riftco_transformer::execution_backend_available(
            ExecutionBackend::Tpu
        )) {
        std::cerr << "hardware gate accepted the tests-only fake PJRT plugin\n";
        return 1;
    }
    const std::string reason(
        riftco_transformer::execution_backend_unavailability_reason(
            ExecutionBackend::Tpu
        )
    );
    if (reason.find("rejects Riftco's tests-only fake PJRT plugin") ==
        std::string::npos) {
        std::cerr << "unexpected hardware-gate diagnostic: " << reason << '\n';
        return 1;
    }
    std::cout << "TPU hardware gate rejected the repository fake plugin\n";
    return 0;
}
