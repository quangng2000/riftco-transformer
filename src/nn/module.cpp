#include "riftco_transformer/nn/module.hpp"

#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace riftco_transformer {
namespace {

struct RegisteredStructure {
    std::unordered_set<Module*> modules;
    std::unordered_set<Parameter*> parameters;
};

void require_valid_registration_name(std::string_view name) {
    if (name.empty()) {
        throw std::invalid_argument(
            "module registration name must not be empty"
        );
    }
    if (name.find('.') != std::string_view::npos) {
        throw std::invalid_argument(
            "module registration name must not contain '.'"
        );
    }
}

}  // namespace

Module::~Module() = default;

void Module::register_parameter(
    std::string_view name,
    Parameter& parameter
) {
    if (registration_closed_) {
        throw std::logic_error(
            "cannot mutate a module after it is registered as a child"
        );
    }
    require_valid_registration_name(name);
    const ParameterHandle handle = parameter.handle();
    for (const auto& registration : registrations_) {
        if (registration.name == name) {
            throw std::invalid_argument(
                "module registration names must be unique"
            );
        }
        if (registration.parameter.get() == handle.get()) {
            throw std::invalid_argument(
                "module parameter is already registered"
            );
        }
    }

    std::unordered_set<const Module*> visited_modules;
    std::vector<const Module*> pending{this};
    while (!pending.empty()) {
        const Module* current = pending.back();
        pending.pop_back();
        if (!visited_modules.insert(current).second) {
            continue;
        }
        for (const auto& registration : current->registrations_) {
            if (registration.parameter.get() == handle.get()) {
                throw std::invalid_argument(
                    "module parameter is already registered"
                );
            }
            if (registration.module != nullptr) {
                pending.push_back(registration.module);
            }
        }
    }

    registrations_.push_back({
        std::string(name),
        handle,
        nullptr,
    });
    parameter.mark_registered();
}

void Module::register_module(
    std::string_view name,
    Module& module
) {
    if (registration_closed_) {
        throw std::logic_error(
            "cannot mutate a module after it is registered as a child"
        );
    }
    require_valid_registration_name(name);
    for (const auto& registration : registrations_) {
        if (registration.name == name) {
            throw std::invalid_argument(
                "module registration names must be unique"
            );
        }
        if (registration.module == &module) {
            throw std::invalid_argument(
                "child module is already registered"
            );
        }
    }
    if (&module == this) {
        throw std::invalid_argument(
            "a module cannot register itself as a child"
        );
    }
    if (module.registered_as_child_) {
        throw std::invalid_argument(
            "child module is already attached to a module hierarchy"
        );
    }

    // The traversal helpers are local lambdas because Registration is a
    // private implementation detail. A visited set also makes validation
    // terminate if a malformed graph reaches this boundary.
    const auto collect_structure = [](Module& root) {
        RegisteredStructure result;
        std::vector<Module*> pending{&root};
        while (!pending.empty()) {
            Module* current = pending.back();
            pending.pop_back();
            if (!result.modules.insert(current).second) {
                continue;
            }
            for (const auto& registration : current->registrations_) {
                if (registration.parameter != nullptr) {
                    result.parameters.insert(registration.parameter.get());
                } else if (registration.module != nullptr) {
                    pending.push_back(registration.module);
                }
            }
        }
        return result;
    };

    const RegisteredStructure current = collect_structure(*this);
    const RegisteredStructure candidate = collect_structure(module);
    if (candidate.modules.contains(this)) {
        throw std::invalid_argument(
            "child module registration would create a cycle"
        );
    }
    for (Module* registered : candidate.modules) {
        if (current.modules.contains(registered)) {
            throw std::invalid_argument(
                "child module hierarchy is already registered"
            );
        }
    }
    for (Parameter* registered : candidate.parameters) {
        if (current.parameters.contains(registered)) {
            throw std::invalid_argument(
                "child module hierarchy contains a duplicate parameter"
            );
        }
    }

    registrations_.push_back({
        std::string(name),
        nullptr,
        &module,
    });
    module.registered_as_child_ = true;
    for (Module* registered : candidate.modules) {
        registered->registration_closed_ = true;
    }
}

ParameterList Module::parameters() {
    ParameterList result;
    std::unordered_set<const Module*> active_modules;
    std::unordered_set<const Module*> visited_modules;
    std::unordered_set<Parameter*> visited_parameters;
    std::unordered_set<std::string> visited_names;

    struct PendingModule {
        Module* module;
        std::string prefix;
        std::size_t registration_index;
    };
    std::vector<PendingModule> stack;
    stack.push_back({this, {}, 0});

    while (!stack.empty()) {
        PendingModule& pending = stack.back();
        if (pending.registration_index == 0) {
            if (active_modules.contains(pending.module)) {
                throw std::logic_error(
                    "registered module hierarchy contains a cycle"
                );
            }
            if (!visited_modules.insert(pending.module).second) {
                throw std::logic_error(
                    "registered module hierarchy contains a duplicate module"
                );
            }
            active_modules.insert(pending.module);
        }

        if (pending.registration_index ==
            pending.module->registrations_.size()) {
            active_modules.erase(pending.module);
            stack.pop_back();
            continue;
        }

        const Registration& registration =
            pending.module->registrations_[pending.registration_index];
        ++pending.registration_index;
        const std::string qualified_name =
            pending.prefix.empty()
                ? registration.name
                : pending.prefix + "." + registration.name;

        if (registration.parameter != nullptr) {
            if (!visited_parameters.insert(
                    registration.parameter.get()
                ).second) {
                throw std::logic_error(
                    "registered module hierarchy contains a duplicate parameter"
                );
            }
            if (!visited_names.insert(qualified_name).second) {
                throw std::logic_error(
                    "registered module hierarchy contains a duplicate name"
                );
            }
            result.push_back({
                qualified_name,
                registration.parameter,
            });
            continue;
        }
        if (registration.module == nullptr) {
            throw std::logic_error(
                "registered module hierarchy contains an invalid entry"
            );
        }
        stack.push_back({
            registration.module,
            qualified_name,
            0,
        });
    }

    return result;
}

void Module::to(ExecutionBackend backend) {
    ParameterList transfer_parameters = parameters();
    std::unordered_set<std::string> names;
    names.reserve(transfer_parameters.size());
    for (const auto& named_parameter : transfer_parameters) {
        names.insert(named_parameter.name);
    }

    struct PendingModule {
        Module* module;
        std::string prefix;
    };
    std::vector<PendingModule> pending{{this, {}}};
    while (!pending.empty()) {
        PendingModule current = std::move(pending.back());
        pending.pop_back();

        ParameterList extras =
            current.module->extra_parameters_for_transfer();
        for (auto& named_parameter : extras) {
            if (named_parameter.name.empty()) {
                throw std::invalid_argument(
                    "extra transfer parameter name must not be empty"
                );
            }
            const std::string qualified_name =
                current.prefix.empty()
                    ? named_parameter.name
                    : current.prefix + "." + named_parameter.name;
            if (!names.insert(qualified_name).second) {
                throw std::invalid_argument(
                    "extra transfer parameter name must be unique"
                );
            }
            named_parameter.name = qualified_name;
            transfer_parameters.push_back(
                std::move(named_parameter)
            );
        }

        for (auto registration =
                 current.module->registrations_.rbegin();
             registration !=
                 current.module->registrations_.rend();
             ++registration) {
            if (registration->module == nullptr) {
                continue;
            }
            const std::string child_prefix =
                current.prefix.empty()
                    ? registration->name
                    : current.prefix + "." + registration->name;
            pending.push_back({
                registration->module,
                child_prefix,
            });
        }
    }

    move_parameters_to(transfer_parameters, backend);
}

ParameterList Module::extra_parameters_for_transfer() {
    return {};
}

void ModuleList::append(std::shared_ptr<Module> module) {
    if (module == nullptr) {
        throw std::invalid_argument(
            "module list cannot append a null module"
        );
    }
    if (modules_.size() ==
        std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("module list size overflow");
    }
    const std::string name = std::to_string(modules_.size());
    modules_.push_back(std::move(module));
    try {
        register_module(name, *modules_.back());
    } catch (...) {
        modules_.pop_back();
        throw;
    }
}

std::size_t ModuleList::size() const noexcept {
    return modules_.size();
}

bool ModuleList::empty() const noexcept {
    return modules_.empty();
}

}  // namespace riftco_transformer
