# Compiling Programs to Transformers

This lab is building a compiler inspired by Joey Velez-Ginorio's
[Compiling to transformers](https://www.engineering.upenn.edu/~joeyv/assets/docs/compiling-to-transformers.pdf).
The long-term goal is to write a small, typed program, compile part of that
program into known Transformer weights, and train the rest of the model around
it. That gives us a ground-truth algorithm inside an otherwise learned model.

The compiler, compact circuit, and learned Section 4 experiment slices are now
implemented. The Cajal-lite symbolic frontend
provides an immutable AST, finite linear types, a Cajal-style linear-context
checker, and a deterministic reference interpreter. The backend-neutral
compiler then encodes finite values and recursively lowers a checked expression
to one dense multilinear map. An optional neural-lowering target now
materializes that map as a differentiable frozen or trainable module using an
exact dense contraction, unary linear map, or bilinear identity-kernel
linear-attention construction. A reusable sequence adapter now surrounds a
program with learned projections, places its result into target positions, and
captures named representations. One conditional-reverse lab is a compact exact
circuit. A second target trains the paper-style F/P/T/I hybrid around a raw
program core. Both use the separate dependency-free interpretation component.
Keeping the symbolic and analysis paths independent of tensors gives both
execution and interpretation trustworthy, reusable boundaries.

## Why this is not a general lambda calculus

Cajal resembles a typed lambda calculus because it has variables, binding,
products, sums, and case analysis. It is first-order, however: there are no
function values, function types, closures, or higher-order application. This is
useful for compilation because every type denotes a finite-dimensional vector
space.

A future lambda-calculus frontend could specialize finite, first-order
functions and lower them into this core. Adding unrestricted closures directly
would lose the simple finite representation needed by this compiler.

## Current pipeline

```mermaid
flowchart LR
    A["Programmatic Cajal-lite AST"] --> B["Linear type checker"]
    B --> C["Reference interpreter"]
    B --> D["Finite value encoding"]
    D --> E["Dense multilinear-map compiler"]
    E --> F["Configurable lowering registry"]
    F --> G["Differentiable programmed module"]
    G --> H["Learned projections + sequence placement"]
    H --> I["Compact exact circuit"]
    H --> M["Learned F/P/T/I hybrid"]
    M --> N["Target-half loss + Adam"]
    I --> J["Named representation capture"]
    M --> J
    J --> K["PCA"]
    I --> L["Ablation + steering"]
    M --> L
```

Every arrow above is implemented. The next boundary is empirical: run and
archive full-scale multi-seed F/P/T/I comparisons before making a claim about
reproducing the paper's reported results.

## Finite types

Let $d(\tau)$ be the number of scalar coordinates used to encode type $\tau$.

| Cajal-lite type | Intuition | Coordinate count |
| --- | --- | --- |
| `Unit` | one trivial value, $()$ | $d(1)=1$ |
| `Sum<A, B>` | either an `A` or a `B` | $d(A\oplus B)=d(A)+d(B)$ |
| `Product<A, B>` | an `A` together with a `B` | $d(A\times B)=d(A)+d(B)$ |
| `Dictionary<K, V>` | a linear map from keys to values | $d(K\rightharpoonup V)=d(K)d(V)$ |

Sums use separate tagged coordinate blocks. Products are direct products, so
their vectors are concatenated. For one-hot enum keys and identity lookup, a
dictionary is represented later as a sum of key/value outer products:

```math
D = \sum_i v_i k_i^\mathsf{T},
\qquad
\mathrm{lookup}(D,q)=Dq.
```

That final equation is linear attention without softmax: orthogonal enum keys
select values by an inner product with the query.

## Dense multilinear maps

For an expression with ordered free-variable dimensions
$d_1,\ldots,d_k$ and output dimension $d_{\mathrm{out}}$, the compiler stores
a dense coefficient tensor

```math
T \in \mathbb{R}^{d_{\mathrm{out}} \times d_1 \times \cdots \times d_k}.
```

The output is the contraction of one input vector with each input axis:

```math
y_o = \sum_{j_1=1}^{d_1}\cdots\sum_{j_k=1}^{d_k}
T_{o,j_1,\ldots,j_k}
\prod_{i=1}^{k} x^{(i)}_{j_i}.
```

The implementation stores coefficients output-major in a flat `double`
vector. A closed expression has $k=0$, so its map is constant and
$y_o=T_o$. `compile(expression, ordered_context)` type-checks the expression,
preserves that context as the input-axis order, and recursively constructs the
map for constants, variables, sums, additive products, sequencing, `let`,
`case`, dictionaries, and lookup.

## Modular neural lowering

The symbolic frontend is the standalone `riftco_transformer::compiler` CMake
target, while neural lowering is the separate
`riftco_transformer::lowering` target. Lowering depends one way on the compiler
and tensor runtime; neither the frontend nor `MultilinearMap` depends on
tensors, autograd, a model, or a hardware backend. Compiler-only tests link
only the compiler archive, so the separation is enforced rather than
conventional. A strategy registry is the extension point for later sparse,
factored, fused, or hardware-specific representations.

The executable interface is representation-neutral too: `metadata().tensors`
lists every owned tensor and `named_tensor(name)` inspects one by name. It does
not force a future factored strategy to pretend it has exactly one weight.
Trainable tensors use the ordinary `Module::parameters()` lifecycle, so Adam
and other optimizers need no compiler-specific adapter.

Link only the concern an application uses:

```cmake
target_link_libraries(symbolic_tool PRIVATE riftco_transformer::compiler)
target_link_libraries(neural_tool PRIVATE riftco_transformer::lowering)
target_link_libraries(analysis_tool PRIVATE riftco_transformer::analysis)
target_link_libraries(sequence_tool PRIVATE riftco_transformer::programmed)
target_link_libraries(reverse_lab PRIVATE riftco_transformer::conditional_reverse)
target_link_libraries(learned_lab PRIVATE riftco_transformer::conditional_reverse_learned)
```

The lowering, programmed, and experiment targets carry their one-way
dependencies transitively. `analysis` and `compiler` each remain
standard-library-only and do not link the tensor runtime.

| Strategy | Accepted arity | Computation | Exactness |
| --- | ---: | --- | --- |
| `dense` | any | $T(x_1\otimes\cdots\otimes x_k)$ | exact subject to the configured FP32 policy |
| `linear` | 1 | $Tx$ | exact subject to the configured FP32 policy |
| `linear_attention` | 2 | form a dynamic linear map from one input, then apply it to the query input | exact identity-kernel contraction; no softmax or scale |
| `mlp` | none yet | capability placeholder | rejected, or explicitly redirected to `dense` by policy |

For a bilinear map with the second input selected as the query, the attention
strategy evaluates

```math
A(x)_{o,j}=\sum_i T_{o,i,j}x_i,
\qquad
y_o=\sum_j A(x)_{o,j}q_j.
```

This is the paper's programmed identity-kernel linear-attention operation, not
the decoder's learned scaled-softmax Q/K/V attention. Calling the current GELU
feed-forward layer an exact lowering would also be misleading, so `mlp`
remains an explicit unsupported capability until a mathematically valid
construction is implemented.

Configuration controls representation without changing the compiler:

```cpp
#include "riftco_transformer/lowering/lowering.hpp"

namespace lowering = riftco_transformer::lowering;

lowering::NeuralLoweringConfig config;
config.strategy = lowering::kAutomaticStrategy;
config.automatic_strategy_order = {
    lowering::kLinearStrategy,
    lowering::kLinearAttentionStrategy,
    lowering::kDenseContractionStrategy,
};
config.unsupported_strategy =
    lowering::UnsupportedStrategyPolicy::Reject;
config.precision = lowering::CoefficientPrecision::RequireExactFloat32;
config.initialization = lowering::CoefficientInitialization::Compiled;
config.trainable = false;
config.backend = riftco_transformer::ExecutionBackend::Cpu;

const lowering::LoweringAnalysis analysis =
    lowering::analyze_neural_lowering(program, config);
auto module = lowering::lower_to_neural(program, config);
```

`auto` tries the configured strategy order, so unary maps select `linear`,
bilinear maps select `linear_attention`, and other arities select `dense` by
default. `DenseFallback` makes a rejected named strategy fall back explicitly
rather than silently. Callers can register custom strategies without editing
the Cajal compiler. Inactive policy fields are independent: an explicit custom
strategy does not need a valid `auto` order, and compiled initialization does
not depend on random-initialization scale.

`Compiled` initialization materializes the compiler coefficients. Exact FP32
mode rejects a coefficient that cannot make a lossless `double` to `float`
round trip; rounded mode records the maximum conversion error in both analysis
and module metadata. `RandomUniform` is a seeded, shape-preserving ablation and
therefore does not claim to preserve the compiled program. With
`trainable=false`, coefficients are frozen graph leaves and no parameter is
registered; with `trainable=true`, the sole named parameter is
`coefficients`, ready for any optimizer that consumes a `ParameterList`.

For nonconstant maps, every input has shape `[..., d_i]`, all leading shapes
must agree, and the result has shape `[..., d_out]`. A zero-arity map uses
`forward_constant(leading_shape)` because it has no input from which to infer
batch or sequence dimensions; `forward({})` is its unbatched shorthand. The
configured CPU, Metal, CUDA, or TPU backend is used only when that backend is
available in the build.

## A tiny checked program

The current API constructs syntax directly in C++; a parser would add surface
syntax but would not change the language semantics.

```cpp
#include "riftco_transformer/compiler/cajal/cajal.hpp"

namespace cajal = riftco_transformer::compiler::cajal;

const cajal::Type bit =
    cajal::Type::sum(cajal::Type::unit(), cajal::Type::unit());

const cajal::Expression identity = cajal::Expression::case_of(
    cajal::Expression::variable("bit"),
    "zero",
    cajal::Expression::inject_left(
        cajal::Expression::variable("zero"), cajal::Type::unit()),
    "one",
    cajal::Expression::inject_right(
        cajal::Type::unit(), cajal::Expression::variable("one")));

const cajal::Value one = cajal::Value::inject_right(
    cajal::Type::unit(), cajal::Value::unit());
const cajal::Value result = cajal::evaluate(identity, {{"bit", one}});
```

`evaluate` derives a type context from the supplied values and invokes the
checker before interpreting the program. The result above is the same right
injection and has type `Sum<Unit, Unit>`.

## What the linear checker protects

The checker distinguishes multiplicative composition from additive
alternatives. This distinction preserves the theorem that a program with $k$
free variables denotes a $k$-linear map:

- `tuple(x, x)` is accepted: both additive product components use the same
  context, and $x\mapsto(x,x)$ is linear.
- `tuple(x, y)` is rejected under context `{x, y}`: its components consume
  different contexts and $(x,y)\mapsto(x,y)$ is not bilinear.
- `sequence(x, y)` splits its context between the two computations; using `x`
  twice there is rejected.
- a `let` binder must appear exactly once in its body.
- both sides of a `case` must consume the same external bindings, because only
  one branch runs.
- all dictionary keys share one context, all values share another context, and
  the key and value contexts are disjoint. This makes every outer-product term
  homogeneous in the same inputs.
- bound names cannot shadow other names, which keeps diagnostics and later
  compilation unambiguous.

Product components are suspended computations. Constructing a tuple does not
run either component; projection evaluates only the selected component. This
matches Cajal's additive product semantics and prevents an error in an
unselected component from changing the result.

Dictionary entries are suspended for the same reason. Lookup evaluates every
key needed to establish a unique match, then evaluates only the selected value.
An error in an unrelated value therefore cannot change a successful lookup.

## Deterministic dictionary semantics

The paper permits nondeterministic choice when several dictionary keys match.
Cajal-lite instead requires exactly one match:

- one matching key returns its value;
- no matching key raises `EvaluationError`;
- multiple matching keys raise `EvaluationError`.

This makes tests reproducible while the compiler is being built. Adding an
explicit set or distribution of possible results is preferable to hiding
nondeterminism in the current `Value` API.

Identity lookup currently accepts only enum keys: `Unit`, or nested sums made
entirely from `Unit`. Their encodings are one-hot and orthogonal, so a dot
product implements exact identity matching. Product keys require the paper's
explicit relation matrix and are rejected for now rather than compiled
incorrectly.

Dictionary coordinates are intentionally algebraic rather than a serialization
of the source entry list. Several lists can sum to the same key/value outer
products, and source dictionary lists are unbounded, so dictionaries have no
generic `decode` or exhaustive `enumerate_values` operation.

There is also a deliberate boundary between discrete source semantics and the
extended algebraic map. The deterministic interpreter throws when lookup finds
no match or multiple matches. The compiled contraction instead produces zero
for no match and a superposition for multiple matches or non-discrete
coordinates. Compiler equivalence is therefore asserted on accepted, total,
unique lookups over discrete finite inputs; behavior outside that domain is the
linear extension, not deterministic interpreter behavior.

## Source map

```text
include/riftco_transformer/compiler/cajal/
  type.hpp          finite type algebra and dimensions
  expression.hpp    immutable program AST
  value.hpp         immutable interpreter values
  checker.hpp       contexts and type-checking contract
  interpreter.hpp   runtime environment and evaluation contract
  encoding.hpp      finite coordinate encoding and discrete decoding
  multilinear_map.hpp dense backend-neutral k-linear coefficient maps
  compiler.hpp      checked AST-to-map compilation contract

src/compiler/cajal/
  type.cpp
  expression.cpp
  value.cpp
  checker.cpp
  interpreter.cpp
  encoding.cpp
  multilinear_map.cpp
  compiler.cpp

include/riftco_transformer/lowering/
  config.hpp        policy, precision, initialization, backend, and limits
  module.hpp        differentiable module and inspectable metadata
  strategy.hpp      strategy interface, registry, analysis, and map bridge
  cajal.hpp         CompiledProgram bridge
  lowering.hpp      aggregate public include

src/lowering/
  config.cpp
  module.cpp
  strategy.cpp
  cajal.cpp

include/riftco_transformer/analysis/
  matrix.hpp        standard-library row-major observation data
  representation.hpp named, owning activation traces
  pca.hpp           deterministic fit/transform/reconstruct PCA
  intervention.hpp row-wise steering and projection removal
  ablation.hpp      paired intervention-effect summaries

src/analysis/
  matrix.cpp
  representation.cpp
  pca.cpp
  intervention.cpp
  ablation.cpp

include/riftco_transformer/programmed/
  sequence_placement.hpp learned projections, placement, capture, interventions

src/programmed/
  sequence_placement.cpp

include/riftco_transformer/experiments/conditional_reverse/
  conditional_reverse.hpp compact-circuit aggregate include
  learned.hpp       learned-experiment aggregate include
  program.hpp       exact finite programs and resource preflight
  task.hpp          deterministic balanced compact-task data
  circuit.hpp       compact compiled circuit and evaluation
  learned_dataset.hpp paper protocol, splits, examples, and batches
  learned_hybrid.hpp F/P/T/I model, capture, intervention, and metrics
  learned_training.hpp Adam training and batched evaluation

src/experiments/conditional_reverse/
  program.cpp
  task.cpp
  circuit.cpp
  learned_dataset.cpp
  learned_hybrid.cpp
  learned_training.cpp

apps/experiments/
  conditional_reverse.cpp
  conditional_reverse_learned.cpp

tests/compiler/cajal/test_cajal.cpp
tests/compiler/cajal/test_multilinear_compiler.cpp
tests/lowering/test_cajal_neural_lowering.cpp
tests/analysis/
tests/experiments/test_conditional_reverse_program.cpp
tests/experiments/test_conditional_reverse_task.cpp
tests/experiments/test_conditional_reverse_circuit.cpp
tests/experiments/learned_hybrid_test.cpp
```

The frontend and multilinear compiler use only the C++ standard library. They
do not depend on `Tensor`, autograd, attention, or a hardware backend. Only the
separate lowering target crosses that boundary.

## Compiler equivalence tests

The compiler test suite checks coordinate layouts and dense maps directly, then
exhaustively enumerates the finite inputs to representative checked programs.
For each environment it requires

```math
\mathrm{compile}(e)(\mathrm{encode}(\rho))
= \mathrm{encode}(\mathrm{evaluate}(e,\rho)).
```

Non-dictionary results are also decoded and compared structurally with the
interpreter value. Separate non-discrete coordinate tests exercise genuine
multilinearity rather than only checking one-hot basis points. Neural-lowering
tests add a third path: interpreter output, dense map application, and lowered
module output must agree. They also check arbitrary leading dimensions,
constant/linear/bilinear/higher-arity execution, input and coefficient
gradients, seeded randomization, precision policy, fallback diagnostics,
parameter registration, backend transfer, and the installed CMake target.

## Conditional reversal with a compiled attention head

For sequence length $L$ and alphabet size $N$, the experiment compiles two
explicit inputs:

- a two-coordinate condition, `Reverse` or `Copy`, projected from the first
  source symbol;
- the complete source sequence encoded in $NL$ coordinates.

The program returns the sequence in reverse order for the left sum injection
and unchanged for the right injection. Its exact bilinear coefficient tensor
has shape

```math
[NL, 2, NL]
```

and therefore $2(NL)^2$ dense elements, of which only $2NL$ are nonzero. All
coefficients are exactly 0 or 1. Automatic neural lowering selects
`linear_attention`, with the source sequence as the query. The adapter owns
Adam-compatible condition, symbol, and output projections, which this exact
circuit initializes to their known solution. It flattens the completed source prefix,
executes the compiled circuit, and places the output only in the target half.
With frozen compilation, Adam sees the six projection weight/bias tensors but
no coefficient parameter. `RandomUniform` supplies a seeded same-shape control;
`trainable=true` deliberately adds the coefficient tensor to the normal
parameter tree.

The lab executable runs the compiled-circuit protocol on disjoint deterministic
train/validation/test splits. PCA is fit on training captures and applied,
without refitting, to validation and test captures; behavioral controls and
interventions are reported on the held-out test split:

```bash
cmake --build --preset debug
./build/debug/riftco-conditional-reverse
```

## Learned F/P/T/I hybrid

The separate `riftco_transformer::conditional_reverse_learned` target follows
the paper artifact's learned architecture without changing the compact circuit.
For a source $s$ and conditional target $f(s)$, the protocol forms

```math
z=[s_1,\ldots,s_L,\mathtt{|},f(s)_1,\ldots,f(s)_L].
```

The model receives $z_{1:2L}$ and predicts the shifted $z_{2:2L+1}$. Only the
last $L$ positions contribute to cross entropy:

```math
\mathcal{L}_{\mathrm{target}}
=-\frac{1}{BL}\sum_{b=1}^{B}\sum_{t=L}^{2L-1}
\log p_\theta\!\left(z_{b,t+1}\mid z_{b,1:t}\right).
```

With default paper dimensions, token and position embeddings produce
$x_1\in\mathbb{R}^{B\times 2L\times20}$. The learned residual paths are

```math
\begin{aligned}
r_1 &= x_1 + W_2\mathrm{ReLU}(W_1x_1+b_1)+b_2,\\
h_1 &= W_A[\mathrm{MHA}_1(r_1);\mathrm{MHA}_2(r_1)]+b_A.
\end{aligned}
```

Each causal MHA has two heads, so there are four learned heads in total. F, P,
and T project the first $L$ residual positions through a biasless
$20\to10$ map, pad the program result into positions $[L,2L)$, and use a
biasless $30\to20$ residual merge. I omits that branch and uses $r_2=h_1+r_1$.

| Variant | Program | Coefficient ownership |
| --- | --- | --- |
| F | full conditional copy/reverse map over two shared-projection inputs | compiled and frozen |
| P | unary unconditional reverse map | compiled and frozen |
| T | F-shaped full map with seeded random coefficients | trainable through ordinary Adam parameters |
| I | no program branch | no program coefficients or merge |

One `LearnedHybridConfig::seed` controls learned initialization and T's
randomized coefficient initialization. F/P override coefficient policy to
compiled and frozen, while T overrides it to randomized and trainable, so a
caller cannot accidentally change a control's scientific meaning through a
generic lowering flag. The learned trainer defaults to practical unclipped
Adam by setting the runtime's finite clipping ceiling to the largest finite
float.

The default config exposes the artifact dimensions ($L=15$, 26 letters plus
delimiter, $d_{model}=20$, two two-head causal attention modules, and
10k/5k/1k/1k train/probe/validation/test splits). The executable defaults to a
smaller, fast configuration so architecture and analysis can be checked during
development. Its finite sources are sampled without replacement across splits,
so its held-out label does not hide train/test source overlap:

```bash
./build/debug/riftco-conditional-reverse-learned --variant F
./build/debug/riftco-conditional-reverse-learned --variant T --steps 32
./build/debug/riftco-conditional-reverse-learned --variant F --paper
```

`--paper` selects the full resource scale and is intentionally opt-in. A clean
paper reproduction still requires completed multi-seed runs, archived configs
and outputs, and comparison against the reported results; architecture support
alone is not that evidence.

One complete local seed-42 `F` run is archived as a machine-readable record at
[`results/experiments/conditional-reverse/m4-max-metal-f-seed-42.json`](../results/experiments/conditional-reverse/m4-max-metal-f-seed-42.json).
On an Apple M4 Max through Metal, 790 Adam steps completed in 277.68 seconds
and reached 100% target-token and exact-sequence accuracy on the 1,000-example
held-out test split. Batch-rolling the program output reduced paired token
accuracy by 96.15 percentage points, while batch-rolling learned attention had
no measured effect. This is a dirty-worktree, single-seed execution record—not
a benchmark or a paper reproduction.

## Interpretation is a separate analysis stage

Yes: PCA, ablation, and steering belong in an interpretation/analysis stage,
not in the compiler or optimizer. They answer different questions:

| Method | Question | Evidential role |
| --- | --- | --- |
| PCA | Which high-variance directions organize captured activations? | Observational summary; correlation, not causal proof |
| Batch-roll ablation | Does the prediction degrade when the representation is reassigned to another example? | Causal necessity test |
| Steering | Does setting or moving a privileged coordinate predictably change behavior? | Causal control/sufficiency test |

The standard-library-only `riftco_transformer::analysis` target owns matrices,
named traces, deterministic covariance/Jacobi PCA, offline interventions, and
paired ablation statistics. It has no tensor or model dependency. The
programmed adapter owns the model-specific execution sites and converts a
captured tensor into the generic `[observation, feature]` representation. This
lets the same analysis component later consume ordinary attention, MLP,
residual-stream, GNN, or other model activations.

PCA must be fit on a fit-only analysis split and only transformed on held-out
examples. The compact circuit uses its training capture split; the learned lab
uses the complete dedicated probe split and transforms complete
validation/test captures in bounded execution batches. PCA uses the raw
$L$-position program output rather than treating the padded zero half as
observations, and the lab reports held-out PC1 associations with branch,
position, and token labels. Ablation and steering results are reported beside
an unaltered baseline and matched controls. Learned-attention, program-output,
and simultaneous attention-plus-program batch-roll effects are separate
measurements. F steering uses an independently generated balanced reverse/copy
set and the artifact's near-exclusive $(0,100)$ / $(100,0)$ scales. The three
methods together are much stronger than a PCA plot alone.

## Current status and next evidence milestone

The current circuit deliberately factors the known condition into a
two-coordinate input and the source into an $NL$-coordinate input. That is
behaviorally exact on valid one-hot task encodings, but it is a compact
compiled-circuit baseline rather than a reproduction of the paper's learned
Model F. The paper model projects the same learned residual sequence into two
full program inputs, and its projection sharing, latent geometry, training
dynamics, and steering basis are therefore different.

That separate learned hybrid is now implemented: 26 letters, length 15, token
and position embeddings, a $20\to80\to20$ ReLU path, four learned causal
heads, shared biasless $20\to10$ program projection, F/P/T/I controls,
target-half loss, Adam, branch-stratified held-out metrics, stable captures,
probe-fit PCA, paired resample ablations, and privileged-basis steering.

The archived M4 Max record establishes that the paper-scale `F` configuration
executes end to end on the native Metal path. It does not establish variance
across seeds or the relative behavior of `P`, `T`, and `I`.

The next milestone is not another architectural layer. It is a reproducible
full-scale experiment suite: run every variant across declared seeds and
hyperparameters, retain validation-based selection, evaluate test once after
selection, archive metrics/checkpoints, and compare PCA/ablation/steering
effects with the paper. Until that evidence exists, the implementation should
be described as paper-faithful experiment support rather than a reproduced
result.

## Deliberate omissions

- no parser or textual syntax
- no general functions or closures
- no nondeterministic choice or relation-valued lookup
- no generic dictionary decoding or exhaustive dictionary enumeration
- no sparse or factored multilinear-map representation
- no generic exact GELU-MLP lowering
- no attachment to the production `DecoderOnlyTransformer` or KV-cached decode
- no archived full-scale multi-seed reproduction, checkpoint, or
  hyperparameter-sweep result yet; only one local seed-42 `F` execution record
  is archived
- no fused hardware kernel specifically for the programmed contraction; the
  current differentiable tensor operations dispatch through the selected
  runtime backend

These omissions keep the completed finite compiler and the optional neural
bridge small enough to audit. They are integration stages, not hidden behavior
in the interpreter or strategy names.
