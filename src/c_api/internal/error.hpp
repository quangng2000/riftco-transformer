#pragma once

#include "handles.hpp"

namespace riftco_transformer::c_api::detail {

void clear_last_error() noexcept;
void set_last_error(const char* message) noexcept;
[[nodiscard]] const char* last_error_message() noexcept;

class BackendUnavailable final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

template <typename Function>
rt_status guard(Function&& function) noexcept {
    try {
        clear_last_error();
        function();
        return RT_STATUS_OK;
    } catch (const BackendUnavailable& error) {
        set_last_error(error.what());
        return RT_STATUS_BACKEND_UNAVAILABLE;
    } catch (const std::domain_error& error) {
        set_last_error(error.what());
        return RT_STATUS_INVALID_ARGUMENT;
    } catch (const std::invalid_argument& error) {
        set_last_error(error.what());
        return RT_STATUS_INVALID_ARGUMENT;
    } catch (const std::out_of_range& error) {
        set_last_error(error.what());
        return RT_STATUS_OUT_OF_RANGE;
    } catch (const std::overflow_error& error) {
        set_last_error(error.what());
        return RT_STATUS_OVERFLOW;
    } catch (const std::bad_alloc& error) {
        set_last_error(error.what());
        return RT_STATUS_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        set_last_error(error.what());
        return RT_STATUS_RUNTIME_ERROR;
    } catch (...) {
        set_last_error("unknown native error");
        return RT_STATUS_UNKNOWN_ERROR;
    }
}

}  // namespace riftco_transformer::c_api::detail
