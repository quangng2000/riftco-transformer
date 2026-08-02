#include "riftco_transformer/core/backend.hpp"
#include "riftco_transformer/nn/module.hpp"

#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using riftco_transformer::ExecutionBackend;
using riftco_transformer::Module;
using riftco_transformer::ModuleList;
using riftco_transformer::Parameter;
using riftco_transformer::ParameterHandle;
using riftco_transformer::ParameterList;
using riftco_transformer::Tensor;

static_assert(!std::is_copy_constructible_v<Module>);
static_assert(!std::is_copy_assignable_v<Module>);
static_assert(!std::is_move_constructible_v<Module>);
static_assert(!std::is_move_assignable_v<Module>);

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_throws(Function&& function, const std::string& message) {
    bool threw = false;
    try {
        std::forward<Function>(function)();
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw, message);
}

class TestModule final : public Module {
public:
    void add_parameter(std::string_view name, Parameter& parameter) {
        register_parameter(name, parameter);
    }

    void add_module(std::string_view name, Module& module) {
        register_module(name, module);
    }
};

class ProbeTransferModule final : public Module {
public:
    explicit ProbeTransferModule(bool fail_preparation = false)
        : fail_preparation_(fail_preparation) {}

    [[nodiscard]] bool prepared() const noexcept {
        return prepared_;
    }

    [[nodiscard]] bool committed() const noexcept {
        return committed_;
    }

private:
    class PreparedProbeTransfer final
        : public PreparedBackendTransfer {
    public:
        explicit PreparedProbeTransfer(bool& committed)
            : committed_(&committed) {}

        void commit() noexcept override {
            *committed_ = true;
        }

    private:
        bool* committed_;
    };

    [[nodiscard]] PreparedBackendTransferList
    prepare_extra_backend_transfers(ExecutionBackend) override {
        prepared_ = true;
        if (fail_preparation_) {
            throw std::runtime_error(
                "injected non-parameter transfer failure"
            );
        }
        PreparedBackendTransferList result;
        result.push_back(
            std::make_unique<PreparedProbeTransfer>(committed_)
        );
        return result;
    }

    bool fail_preparation_;
    bool prepared_ = false;
    bool committed_ = false;
};

void require_parameter_order(
    const ParameterList& actual,
    const std::vector<std::string>& expected_names,
    const std::vector<ParameterHandle>& expected_parameters
) {
    require(
        actual.size() == expected_names.size(),
        "registered parameter count"
    );
    require(
        actual.size() == expected_parameters.size(),
        "expected pointer count"
    );
    for (std::size_t index = 0; index < actual.size(); ++index) {
        require(
            actual[index].name == expected_names[index],
            "registered parameter name/order at index " +
                std::to_string(index)
        );
        require(
            actual[index].parameter == expected_parameters[index],
            "registered parameter pointer at index " +
                std::to_string(index)
        );
    }
}

void test_recursive_names_and_stable_registration_order() {
    Parameter root_first(Tensor({1}, 1.0F));
    Parameter child_weight(Tensor({1}, 2.0F));
    Parameter nested_bias(Tensor({1}, 3.0F));
    Parameter root_last(Tensor({1}, 4.0F));

    TestModule nested;
    nested.add_parameter("bias", nested_bias);
    TestModule child;
    child.add_parameter("weight", child_weight);
    child.add_module("projection", nested);
    TestModule root;
    root.add_parameter("first", root_first);
    root.add_module("encoder", child);
    root.add_parameter("last", root_last);

    const std::vector<std::string> names{
        "first",
        "encoder.weight",
        "encoder.projection.bias",
        "last",
    };
    const std::vector<ParameterHandle> pointers{
        root_first.handle(),
        child_weight.handle(),
        nested_bias.handle(),
        root_last.handle(),
    };
    require_parameter_order(root.parameters(), names, pointers);
    require_parameter_order(root.parameters(), names, pointers);
}

void test_module_list_numeric_registration() {
    Parameter first(Tensor({1}, 1.0F));
    Parameter second(Tensor({1}, 2.0F));
    auto first_block = std::make_shared<TestModule>();
    first_block->add_parameter("weight", first);
    auto second_block = std::make_shared<TestModule>();
    second_block->add_parameter("weight", second);
    ModuleList blocks;
    require(blocks.empty(), "new module list is empty");
    blocks.append(first_block);
    blocks.append(second_block);
    require(blocks.size() == 2, "module list size after append");

    TestModule root;
    root.add_module("blocks", blocks);
    require_parameter_order(
        root.parameters(),
        {"blocks.0.weight", "blocks.1.weight"},
        {first.handle(), second.handle()}
    );
    require_throws(
        [&] { blocks.append(first_block); },
        "module list rejects a duplicate child"
    );
    require_throws(
        [&] { blocks.append(nullptr); },
        "module list rejects a null child"
    );
    require(blocks.size() == 2, "failed module-list append is atomic");

    first_block.reset();
    require_parameter_order(
        root.parameters(),
        {"blocks.0.weight", "blocks.1.weight"},
        {first.handle(), second.handle()}
    );
}

void test_invalid_duplicate_and_cyclic_registration() {
    Parameter first(Tensor({1}, 1.0F));
    Parameter second(Tensor({1}, 2.0F));
    TestModule child;
    TestModule root;

    require_throws(
        [&] { root.add_parameter("", first); },
        "empty registration name must be rejected"
    );
    require_throws(
        [&] { root.add_parameter("nested.weight", first); },
        "qualified registration segment must be rejected"
    );
    root.add_parameter("entry", first);
    require_throws(
        [&] { root.add_parameter("entry", second); },
        "duplicate registration name must be rejected"
    );
    require_throws(
        [&] { root.add_parameter("alias", first); },
        "duplicate parameter must be rejected"
    );
    Parameter moved_alias(std::move(first));
    require_throws(
        [&] { root.add_parameter("moved_alias", moved_alias); },
        "moved wrappers with one canonical state must be rejected"
    );
    Parameter replacement(Tensor({1}, 6.0F));
    require_throws(
        [&] { first = std::move(replacement); },
        "registered parameter wrapper identity must be sealed"
    );
    require(
        root.parameters().front().parameter->value().flat(0) == 1.0F,
        "rejected registered wrapper rebind preserves state"
    );
    require_throws(
        [&] { root.add_module("entry", child); },
        "parameter and child names share one namespace"
    );
    require_throws(
        [&] { root.add_module("self", root); },
        "self registration must be rejected"
    );

    TestModule parent;
    TestModule descendant;
    parent.add_module("descendant", descendant);
    require_throws(
        [&] { descendant.add_module("parent", parent); },
        "indirect module cycle must be rejected"
    );
    require(parent.parameters().empty(), "failed cycle leaves graph valid");

    TestModule leaf;
    TestModule branch;
    TestModule tree;
    branch.add_module("leaf", leaf);
    tree.add_module("branch", branch);
    require_throws(
        [&] { tree.add_module("leaf_alias", leaf); },
        "duplicate descendant module must be rejected"
    );

    Parameter shared(Tensor({1}, 5.0F));
    TestModule left;
    TestModule right;
    TestModule shared_root;
    left.add_parameter("value", shared);
    right.add_parameter("value", shared);
    shared_root.add_module("left", left);
    require_throws(
        [&] { shared_root.add_module("right", right); },
        "overlapping child parameters must be rejected"
    );

    Parameter descendant_parameter(Tensor({1}, 7.0F));
    TestModule parameter_child;
    TestModule parameter_root;
    parameter_child.add_parameter("value", descendant_parameter);
    parameter_root.add_module("child", parameter_child);
    require_throws(
        [&] {
            parameter_root.add_parameter(
                "descendant_alias",
                descendant_parameter
            );
        },
        "parameter already reachable through a child must be rejected"
    );

    TestModule late_left;
    TestModule late_right;
    TestModule late_root;
    late_root.add_module("left", late_left);
    late_root.add_module("right", late_right);
    Parameter late_parameter(Tensor({1}, 8.0F));
    require_throws(
        [&] { late_left.add_parameter("value", late_parameter); },
        "attached child registration must be sealed"
    );
    require_throws(
        [&] { late_right.add_module("nested", late_left); },
        "attached child module hierarchy must be sealed"
    );
    require(late_root.parameters().empty(), "sealed tree remains valid");
}

void test_recursive_backend_transfer() {
    Parameter root_parameter(Tensor({2}, {1.25F, -2.5F}));
    Parameter child_parameter(Tensor({1}, 3.75F));
    TestModule child;
    child.add_parameter("value", child_parameter);
    TestModule root;
    root.add_parameter("root", root_parameter);
    root.add_module("child", child);

    root.to(ExecutionBackend::Cpu);
    for (const auto& named_parameter : root.parameters()) {
        require(
            named_parameter.parameter->value().backend() ==
                ExecutionBackend::Cpu,
            "same-backend recursive transfer value"
        );
        require(
            named_parameter.parameter->gradient().backend() ==
                ExecutionBackend::Cpu,
            "same-backend recursive transfer gradient"
        );
    }

    if (!riftco_transformer::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        require_throws(
            [&] { root.to(ExecutionBackend::Metal); },
            "unavailable recursive transfer must throw"
        );
        for (const auto& named_parameter : root.parameters()) {
            require(
                named_parameter.parameter->value().backend() ==
                    ExecutionBackend::Cpu,
                "failed recursive transfer is transactional"
            );
        }
        return;
    }

    root.to(ExecutionBackend::Metal);
    for (const auto& named_parameter : root.parameters()) {
        require(
            named_parameter.parameter->value().backend() ==
                ExecutionBackend::Metal,
            "recursive transfer value reaches Metal"
        );
        require(
            named_parameter.parameter->gradient().backend() ==
                ExecutionBackend::Metal,
            "recursive transfer gradient reaches Metal"
        );
    }
    require(
        root_parameter.value().flat(0) == 1.25F &&
            root_parameter.value().flat(1) == -2.5F &&
            child_parameter.value().flat(0) == 3.75F,
        "recursive transfer preserves values"
    );

    root.to(ExecutionBackend::Cpu);
    for (const auto& named_parameter : root.parameters()) {
        require(
            named_parameter.parameter->value().backend() ==
                ExecutionBackend::Cpu,
            "recursive transfer round trip"
        );
    }
}

void test_non_parameter_transfer_prepares_before_any_commit() {
    ProbeTransferModule prepared_child;
    ProbeTransferModule failing_child(true);
    TestModule root;
    root.add_module("prepared", prepared_child);
    root.add_module("failing", failing_child);

    require_throws(
        [&] { root.to(ExecutionBackend::Cpu); },
        "a failed extra-state preparation must reject the transfer"
    );
    require(
        prepared_child.prepared() && failing_child.prepared(),
        "registered extra state must be prepared in tree order"
    );
    require(
        !prepared_child.committed(),
        "no extra state may commit when a later preparation fails"
    );

    ProbeTransferModule successful_child;
    TestModule successful_root;
    successful_root.add_module("child", successful_child);
    successful_root.to(ExecutionBackend::Cpu);
    require(
        successful_child.prepared() && successful_child.committed(),
        "prepared extra state must commit after a successful transaction"
    );
}

}  // namespace

int main() {
    try {
        test_recursive_names_and_stable_registration_order();
        test_module_list_numeric_registration();
        test_invalid_duplicate_and_cyclic_registration();
        test_recursive_backend_transfer();
        test_non_parameter_transfer_prepares_before_any_commit();
        std::cout << "module tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "module test failure: " << error.what() << '\n';
        return 1;
    }
}
