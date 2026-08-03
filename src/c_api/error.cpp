#include "internal/error.hpp"

namespace riftco_transformer::c_api::detail {
namespace {

constexpr std::size_t kLastErrorCapacity = 1024;
thread_local std::array<char, kLastErrorCapacity> last_error{};

}  // namespace

void clear_last_error() noexcept {
    last_error[0] = '\0';
}

void set_last_error(const char* message) noexcept {
    std::size_t index = 0;
    if (message != nullptr) {
        while (index + 1 < last_error.size() &&
               message[index] != '\0') {
            last_error[index] = message[index];
            ++index;
        }
    }
    last_error[index] = '\0';
}

const char* last_error_message() noexcept {
    return last_error.data();
}

}  // namespace riftco_transformer::c_api::detail
