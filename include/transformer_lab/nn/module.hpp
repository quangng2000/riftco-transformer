#pragma once

#include "transformer_lab/nn/parameter.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace transformer_lab {

// Base lifecycle for modules that own Parameters and nested Modules.
//
// Registrations retain insertion order. Parameter registrations keep the
// parameter's canonical state alive; child-module registrations are non-owning,
// so a derived module must keep every child alive for at least as long as this
// base object. Modules cannot move or copy because either operation would
// invalidate child registrations held by their parents.
class Module {
public:
    Module() = default;
    virtual ~Module();

    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
    Module(Module&&) = delete;
    Module& operator=(Module&&) = delete;

    // Returns parameters in registration order. Child names are qualified
    // recursively, for example "attention.query.weight".
    [[nodiscard]] ParameterList parameters();

    // Transfers the complete registered parameter tree transactionally.
    // Call before constructing a forward graph or backend-specific optimizer.
    void to(ExecutionBackend backend);

protected:
    // A registration segment must be nonempty and cannot contain '.', which is
    // reserved as the separator in recursively qualified parameter names.
    void register_parameter(
        std::string_view name,
        Parameter& parameter
    );
    void register_module(
        std::string_view name,
        Module& module
    );

    // Dynamic parameter collections that intentionally remain outside the
    // stable parameters() schema can participate in transactional transfer by
    // overriding this hook. Returned names are relative to this module.
    [[nodiscard]] virtual ParameterList
    extra_parameters_for_transfer();

private:
    struct Registration {
        std::string name;
        ParameterHandle parameter;
        Module* module;
    };

    std::vector<Registration> registrations_;
    bool registration_closed_ = false;
    bool registered_as_child_ = false;
};

// Owning ordered container for repeated child modules. append() registers
// children as "0", "1", ... so nesting it as "blocks" produces names such as
// "blocks.0.attention.weight". Shared ownership makes a retained list safe
// even when its caller releases the original child pointer.
class ModuleList final : public Module {
public:
    void append(std::shared_ptr<Module> module);

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept;

private:
    std::vector<std::shared_ptr<Module>> modules_;
};

}  // namespace transformer_lab
