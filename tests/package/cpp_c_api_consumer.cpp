#include "riftco_transformer/c_api.h"

#include <cstdlib>

int main() {
    return rt_abi_version() == RT_ABI_VERSION &&
                   RT_ABI_VERSION_MINOR == UINT32_C(4) &&
                   RT_BACKEND_CUDA == static_cast<rt_backend>(2) &&
                   RT_BACKEND_TPU == static_cast<rt_backend>(3) &&
                   RT_ACTIVATION_CHECKPOINTING_DISABLED !=
                       RT_ACTIVATION_CHECKPOINTING_TRANSFORMER_BLOCK
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
