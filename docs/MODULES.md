# Module and Parameter Lifecycle

`Module` gives reusable layers one common lifecycle without forcing unrelated
forward signatures into one virtual interface. A module registers its direct
parameters and child modules once during construction. The framework can then
derive recursive names, parameter order, and backend transfer from that tree:

```text
typed forward methods ───────────────→ Variable graph ─→ autograd
        │
        └─ registered Module tree ───→ ParameterList
                                        ├─ Adam
                                        ├─ decoder artifact capture
                                        ├─ global gradient norm
                                        └─ backend transfer
```

The implemented module types are `Linear`, `Embedding`, `LayerNorm`,
`LowRankAdapter`, `FeedForward`, `CausalSelfAttention`, `TransformerBlock`,
and `DecoderOnlyTransformer`.

## Registration

A derived module registers owned members in its constructor:

```cpp
class ProjectionPair final : public transformer_lab::Module {
public:
    ProjectionPair(
        std::size_t width,
        std::mt19937& random
    )
        : first_(width, width, random),
          second_(width, width, random) {
        register_module("first", first_);
        register_module("second", second_);
    }

    [[nodiscard]] transformer_lab::Variable forward(
        const transformer_lab::Variable& input
    ) const {
        return second_.forward(first_.forward(input));
    }

private:
    transformer_lab::Linear first_;
    transformer_lab::Linear second_;
};
```

`register_parameter(name, parameter)` and `register_module(name, child)` are
protected because registration is part of a derived module's construction,
not a mutation surface for callers. A name segment must be nonempty and cannot
contain `.`; dots are reserved for recursive qualification. Parameter and
child names share one namespace.

Registration rejects duplicate names, duplicate parameter identities,
duplicate child hierarchies, and direct or indirect cycles. It is therefore a
tree rather than a general graph. Register children only after their own
construction is complete. Attaching a subtree seals its static registration,
so later child mutation cannot introduce a duplicate through an ancestor.

## Repeated children

`ModuleList` is an ordered, shared-owning module container for repeated
children. It accepts `std::shared_ptr<Module>` values and registers them as
`"0"`, `"1"`, and so on. Registering that list as `"blocks"` produces names
such as:

```text
blocks.0.attention.query.weight
blocks.0.attention.query.bias
blocks.1.attention.query.weight
```

The decoder keeps typed block pointers and gives the same shared pointers to
`ModuleList`. Releasing the caller's copy cannot leave a dangling registered
child.

## Lifetime and nonmoving modules

Direct child registrations store module addresses. A derived parent must
therefore keep member children alive for the parent's usable lifetime.
`Module` is non-copyable and non-movable so a registered child cannot silently
change address after its parent records it. Repeated/dynamic children should
use the owning `ModuleList`; ordinary composites store children directly as
members.

Parameter registration has a different ownership rule. It stores a
`ParameterHandle`, which retains the parameter's shared value, gradient, and
autograd leaf state. Moving a `Parameter` wrapper does not change its canonical
registered identity.

Each `NamedParameter` returned in a `ParameterList` also owns a
`ParameterHandle`. Its public `.parameter` member remains a raw `Parameter*`
compatibility view, so existing code can still write:

```cpp
for (const auto& entry : module.parameters()) {
    use(entry.name, entry.parameter->value());
}
```

The owning entry or handle must remain alive while that raw view is used. Do
not extract and retain the raw pointer after all corresponding lists, handles,
and optimizers have been destroyed. Copying a `ParameterList` copies the
owning handles, so an optimizer remains safe if the original list, parameter
wrapper, or originating module later leaves scope.

The raw pointer is a compatibility view of the canonical identity. Use
`set_value`, `zero_gradient`, or the optimizer to mutate its state; attempting
to move-assign a different `Parameter` wrapper into that canonical proxy is
rejected so existing handles cannot be silently rebound.

The `.parameter` pointer itself is read-only and always equals the entry's
owning handle. Likewise, once a wrapper is registered into a `Module`, that
wrapper cannot be move-assigned to a different state. These two rules keep a
custom layer's forward member, recursive parameter tree, and optimizer on one
canonical identity.

## Recursive names and order

`parameters()` returns a fresh owning `ParameterList` in depth-first
registration order. Each child name is joined to its descendants with `.`.
Repeated calls produce the same order as long as the registered structure is
unchanged.

This order is a framework contract used by optimizers and the decoder-specific
artifact code. A composite module should register children in its intended
public order rather than rebuilding names manually. The decoder's base
parameter tree remains independent of LoRA state: dynamically attached adapter
factors are exposed by the explicit `lora_parameters()` path and are not added
to base `parameters()`.

## Transactional backend transfer

`Module::to(backend)` obtains the recursive list and delegates to
`move_parameters_to`. The transfer prepares every destination value and fresh
zero gradient before committing any changed parameter. If preparation fails,
the registered tree remains unchanged. Parameters already on the destination
backend are left untouched.

A derived module may override the protected
`extra_parameters_for_transfer()` hook for dynamic parameter collections that
must not change the stable `parameters()` schema. `Linear` uses this for active
or retained LoRA storage. The base `Module::to()` walks the entire child tree,
invokes that hook polymorphically at each node, and commits base plus dynamic
parameters in one transaction. Calling `to()` through `Module&` is therefore
safe; no derived transfer override is required.

Move a module before constructing a forward graph or an optimizer. A
successful cross-backend move changes leaf versions and resets moved gradients,
so graphs built from the previous values are intentionally invalid.

## Forward and backward responsibilities

`Module` deliberately has no virtual `forward()` because inputs differ across
layers: an embedding consumes token IDs, a linear layer consumes a `Variable`,
and the decoder also has a serving-only cache path. Each concrete type keeps a
strongly typed forward interface.

Modules also do not implement a module-level `backward()`. Their forward
methods compose differentiable `Variable` operations, and central autograd
applies the chain rule over the resulting operation graph. A new fused or
backend-specific operation can join that graph through:

```cpp
custom_gradient(output_tensor, inputs, vjp_callback)
```

The callback receives the upstream gradient and returns one tensor per
positional input. Core validates the complete count, shape, and backend result
before accumulating any contribution, and a failed callback leaves visible
gradients unchanged. This is the public extension seam for a custom VJP; it
does not expose graph nodes and does not require friendship from a model class.
See [AUTOGRAD.md](AUTOGRAD.md) for the full contract and an example.

## Generic consumers

Optimizers and gradient utilities depend on `ParameterList`, not on a concrete
model type. `global_gradient_norm(parameters)` therefore works for a complete
model tree, an explicit LoRA-only list, or a custom registered module with the
same validation rules. This keeps module composition, differentiation, and
parameter update policy separate.
