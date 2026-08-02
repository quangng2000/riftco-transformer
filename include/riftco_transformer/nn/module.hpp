#pragma once

#include "riftco_transformer/nn/parameter.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace riftco_transformer {

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

    // Transfers the complete registered module tree transactionally, including
    // both Parameters and non-optimizer backend state exposed through the
    // protected preparation hook. Call before constructing a forward graph or
    // backend-specific optimizer.
    virtual void to(ExecutionBackend backend);

protected:
    // A staged, non-optimizer part of a backend transfer. Derived modules must
    // allocate and copy everything in prepare_extra_backend_transfers()
    // without changing live state. Module::to() calls commit() only after the
    // entire registered tree and every Parameter have been prepared. commit()
    // must therefore remain allocation-free and non-throwing.
    class PreparedBackendTransfer {
    public:
        virtual ~PreparedBackendTransfer() = default;
        virtual void commit() noexcept = 0;

        PreparedBackendTransfer(
            const PreparedBackendTransfer&
        ) = delete;
        PreparedBackendTransfer& operator=(
            const PreparedBackendTransfer&
        ) = delete;

    protected:
        PreparedBackendTransfer() = default;
    };

    using PreparedBackendTransferList =
        std::vector<std::unique_ptr<PreparedBackendTransfer>>;

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

    // Controlled lifecycle for a dense parameter slot that is replaced by an
    // immutable packed weight. The preflight rejects retained external handles
    // so deactivation can actually release the dense allocation.
    void preflight_parameter_deactivation(
        std::string_view name,
        Parameter& parameter
    ) const;
    void deactivate_parameter(
        std::string_view name,
        Parameter& parameter
    );
    void reactivate_parameter(
        std::string_view name,
        Parameter& parameter
    );

    // Dynamic parameter collections that intentionally remain outside the
    // stable parameters() schema can participate in transactional transfer by
    // overriding this hook. Returned names are relative to this module.
    [[nodiscard]] virtual ParameterList
    extra_parameters_for_transfer();

    // Backend-owned state returned here is transfer-only state: it is not
    // added to parameters() and therefore never acquires optimizer state.
    [[nodiscard]] virtual PreparedBackendTransferList
    prepare_extra_backend_transfers(ExecutionBackend backend);

private:
    enum class RegistrationKind {
        Parameter,
        Module,
    };

    struct Registration {
        std::string name;
        RegistrationKind kind;
        ParameterHandle parameter;
        Module* module;
        bool active;
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

}  // namespace riftco_transformer
