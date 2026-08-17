# FastForward Design Choices

This document records the *fundamental design choices* behind FastForward and
the rationale for each. See [§9](#9-status-summary) for which of them are
implemented today and which are still planned.

---

## 1. What FastForward is

FastForward is a Python package built on top of PyTorch for neural-network
quantization. It aims to serve as a foundation for research and prototyping in
quantization, and second as a reliable path to deployment. It leverages
PyTorch's eager mode, so working with a quantized network is meant to feel like
working with any other `torch.nn.Module`.

In practice, a model moves through FastForward in stages: a float module is
converted into a *quant-ready* tree of quantizer stubs, those stubs are resolved
into concrete *quant-aware* quantizers, ranges are estimated to produce a fitted
*quantized* model, and the result can be exported to a deployment artifact. Each
stage is separately customizable, and users are not required to make every
decision up front ([§2](#2-design-principles)).

These stages run on CPU/GPU, while the exported artifact targets accelerators
such as NPUs; GPU references elsewhere in this document refer to the former.

Three properties shape the library as a whole:

- **Ease of use and extensibility as primary goals**, chosen ahead of breadth of
  features.
- **Minimal dependencies and easy installability**, so it composes with many
  PyTorch / OS / library versions.
- **A single library, configured for hardware**, rather than divergent
  forks — HW-aware behavior belongs in (possibly private) configuration,
  not in a separate codebase.

A complementary principle bounds the scope: the **quantization core does one
thing — quantization**. Hardware targets, model conversion, logging, and export
are all supported, but they live in layers on top of the core rather than being
baked into it. This "do one thing well / pull complexity down" stance is the
reason the codebase is split into cooperating layers
([§3](#3-architecture-at-a-glance)) rather than a monolith.

---

## 2. Design principles

These principles are visible throughout the codebase:

1. **Eager-mode transparency.** A quantized model should be as inspectable as
   any PyTorch module — breakpoints, `print`, and standard introspection all
   work. This is the reason Autoquant emits plain Python code rather than a traced
   graph ([§6](#6-autoquant-quantization-by-source-transpilation)).
2. **PyTorch-native, define-by-run.** Quantization is expressed in ordinary
   Python/PyTorch, deliberately *not* requiring `torch.compile` or trace-only
   capture as the primary path.
3. **Enable, don't prescribe.** The library exposes mechanism (custom
   quantizers, custom kernels, custom export stages) rather than hard-coding
   policy. "Simple ≠ easy": the common path is unsurprising, but non-standard
   tasks are supported at the cost of some user investment.
4. **Deferred quantization specification.** A model progresses
   float → quant-ready (stubs) → quant-aware (concrete quantizers) → quantized
   (fitted). Users are not forced to make all decisions up front ([§5](#5-quantized-modules-ffnn)).
5. **Separation of concerns / layering.** Core quantization, dispatch,
   range-setting, modules, orchestration, and export are distinct layers that
   can be adopted incrementally.

---

## 3. Architecture at a glance

FastForward is organized as cooperating layers, low-level to high-level.
Packages are given relative to `src/fastforward/`:

| Layer | Package | Responsibility |
|---|---|---|
| Quantized Tensor & Ops Dispatch | `quantized_tensor.py`, `dispatcher.py`, `_quantops/` | Represent quantized values + route operators |
| Quantization logic | `quantization/`, `range_setting/` | Quantize/dequantize functions, granularity, range estimation |
| Modules | `nn/` | Quantization-aware module wrappers + functional API |
| Model selection | `mpath/` | Select/update many submodules at once |
| Autoquant | `_autoquant/`, `autoquant.py` | Generate quantized-ready module source code from original torch model source |
| Orchestration | `_orchestration/`, `orchestration.py` | Schedule and execute layer-by-layer algorithms (GPTQ, etc.) over a graph IR |
| Export | `export/` | Produce deployable artifacts (QNN/ONNX) |
| Cross-cutting | `serialization.py`, `flags.py`, `overrides.py`, `testing/` | State I/O, global modes, helpers |

---

## 4. Quantization core

### 4.1 `QuantizedTensor` carries data *and* metadata

**Choice.** The central abstraction is `QuantizedTensor` (a `torch.Tensor`
subclass) that stores the integer representation together with a
`QuantizationContext` (the quantization function + its parameters: scale,
offset, num_bits, tile size).

**Why.** Keeping parameters attached to the tensor lets operators and modules
understand format, range, and granularity without out-of-band bookkeeping.

### 4.2 Autograd semantics: identity-Jacobian dequantize, gradients live in *quantize*

**Choice.** A quantized tensor represents *real values* in quantized form.
Dequantization changes representation, not value, so its backward pass is the
**identity**. *All* gradient behavior — exact or approximate (STE/LSQ-style) —
is implemented in the **quantize** function's backward pass.

**Why.** This makes gradient approximations mathematically well-defined and
composable: any custom quantization function that follows the same convention
interoperates cleanly. The convention is documented in-code and encouraged (not
strictly enforced) for new quantization functions.

### 4.3 Operator dispatch extends `__torch_function__` with predicate routing

**Choice.** Rather than replace PyTorch's operator routing, FastForward extends
it: `QuantizedTensor.__torch_function__` calls a predicate-based `dispatch()`.
Kernels register with a priority (`DEFAULT` → `FALLBACK` →
`NOT_IMPLEMENTED_FALLBACK`); the first matching predicate wins. If nothing
matches, a dequantization fallback runs (subject to strict mode, [§4.6](#46-quantization-safety-strict-mode-overrides)).

**Why.** Quantized and float ops coexist in one eager graph — essential for
debugging, gradual adoption, and mixed-precision experiments.

### 4.4 A data-driven YAML operator table

**Choice.** Quantized operators are declared *as data* in
`_quantops/quantized_operators.yaml` (each entry: an `op` signature using
`Quantized`/`MaybeQuantized` symbolic types, a `fallback` torch callable, and
optional `aliases`), not by subclassing per operator.

**Why.** Declarative operator specs decouple the *set* of quantized operators
from runtime dispatch and from Autoquant's code generation. New operators, or
extra qualified names for the same op, are added without touching dispatch
logic.

### 4.5 Granularity as a tiling abstraction

**Choice.** `Granularity` is an abstract base with `PerTensor`, `PerChannel`,
`PerBlock`, and `PerTile` implementations. Each computes `tile_size(shape)`,
from which parameter dimensionality is derived.

**Why.** A single tiling abstraction expresses all granularities uniformly, so
quantization logic need not special-case per-channel vs. per-block.

### 4.6 Quantization safety: strict mode + overrides

**Choice.** A global `strict_quantization` flag (default **on**) turns implicit
dequantization into an error; context managers (`disable_quantization`, etc.)
relax it locally.

**Why.** Catch common quantization mistakes early by default, while still
allowing exploratory, partially-quantized runs.

### 4.7 No in-place ops on `QuantizedTensor`

**Choice.** In-place tensor operators (`+=`, etc.) are disabled and fall back to
out-of-place semantics.

**Why.** In-place mutation would generally push values off the quantization
grid, violating the "quantized tensor = real values on a grid" invariant.

### 4.8 Export mode dequantizes on attach

**Choice.** With `export_mode` set, attaching a `QuantizationContext` yields a
dequantized float tensor instead of a `QuantizedTensor`.

**Why.** Lets capture/export machinery operate without the eager-mode
`QuantizedTensor` infrastructure being active.

### 4.9 Range setting is a pluggable subsystem

**Choice.** Range-estimation strategies (min/max, min-error, smoothed min/max)
live in `range_setting/`, invoked via an `estimate_ranges` context manager, so
all quantizers estimate/update ranges consistently. Algorithms such as GPTQ
([§7](#7-orchestration-algorithms)) build on this subsystem rather than
re-implementing estimation.

---

## 5. Quantized modules (`ff.nn`)

**Choice.** Quantized modules inherit from both `QuantizedModule` and their
float counterpart (e.g. `QuantizedLinear(QuantizedModule, torch.nn.Linear)`). A
metaclass calls `__init_quantization__`, which installs input/weight/bias/output
quantizers — initially as `QuantizerStub` placeholders. A functional API
(`ff.nn.functional`) mirrors the operator table. `mpath` provides a selector DSL
to configure many submodules at once.

**Why.** This realizes *deferred specification* ([§2](#2-design-principles)): converting a model
produces a quant-ready module tree of stubs, which are later resolved to
concrete quantizers. Familiar `torch.nn` APIs are preserved so adoption is
incremental.

---

## 6. Autoquant — quantization by source transpilation

**Choice.** Autoquant converts a float module into quantization-ready **module
Python source code**: it reads the module's source, parses it to a Concrete Syntax
Tree (libcst), applies passes, and emits formatted Python.

Key sub-choices:

- **Emit source, not a runtime graph transform.** Preserves eager-mode
  execution (dispatcher, `QuantizedTensor`, range-setting all stay live and
  debuggable), and keeps control flow, loops, and comments intact.
- **libcst (CST) over AST.** Preserves formatting/comments through the
  round-trip.
- **Optional mypy type inference** to decide which variables are quantized
  ("quant vs. nonquant").
- **Correct operation order under variable reassignment.** SSA-like versioning
  and walrus-bound temporaries avoid re-quantizing shared subexpressions and
  keep quantizer insertion at safe points.
- **Collision-safe naming** for generated classes and aliases.

**Roadmap (design material, not yet in code):** policy-driven Autoquant
(generate alternative rewrites — a `PatternRule` extension point exists but is
not yet exercised); structured Autoquant *reports* (currently only logging);
and agent-in-the-loop Autoquant. Generation of freeze-parameter-ready quantizers
is likewise a documented interaction question rather than implemented codegen.

---

## 7. Orchestration & algorithms

**Choice.** A dedicated orchestration layer separates *execution policy*
(ordering, scheduling, memory movement) from *algorithm logic* (GPTQ, local-error
methods). A model is represented as an explicit DAG (`GraphModule`) whose node
arguments are symbolic references, lowered by an `InstructionScheduler` into a
linear `InstructionProgram` executed by a register-based `InstructionEngine`.

Key sub-choices:

- **Graph IR + topological scheduling**, not control flow, so execution order is
  explicit and analyzable.
- **Instruction VM** with a small instruction set (call/optimize module, load
  attribute, store/return, move module/activations, delete register entries).
- **Multi-context execution via `Delegate` + `Contexts`.** Local-error methods
  (e.g. GPTQ) need activations in several regimes (float calibration vs.
  quantized forward); contexts propagate backward through the graph so each node
  produces what downstream optimization needs.
- **Lifetime & offloading passes** free activations at last use and move
  weights/activations between compute/storage devices.

**Roadmap:** Work on orchestration is primarily algorithm-driven. We plan on adding
support for richer forward-data flows to support algorithms like AdaRound and BrecQ,
and backward-data flows to support algorithms like YAQA. Secondary to this, we want to
add optimizations for distributed workflows (multi-GPU offloading, instruction placement
on specific devices, automatic load balancing). Finally, although we have a tracing
mechanism for automatic GraphModule creation, this is still experimental and we would
like to make this feature stable.

---

## 8. Export

**Choice.** Export is a **DAG of stages** resolved by topological sort, selected
from a **registry keyed by `(target, format)`**. Quantization is serialized as
**ONNX `metadata_props` side-channel** (plus an `.encodings` file), keeping the
graph itself float and standard. Model capture uses **`torch.export` (PT2E)**.

Key sub-choices:

- **Stage DAG over a monolithic exporter**, so targets add, replace, and
  re-order stages independently.
- **`(target, format)` registry keying**, so one target can support multiple
  formats side-by-side. Two QNN pipelines ship: `("qnn", "onnx")` (metadata) and
  `("qnn", "onnx_qdq")` (embedded Q/DQ nodes).
- **Metadata-based encodings, not fake-quant nodes**, so downstream tooling
  reconstructs quantization from side-channel data on a clean float graph.
  Encoding schemas are **versioned** (Legacy 0.6.1 / V1 / V2) behind a handler
  protocol; V1 is the default.
- **PT2E / `torch.export` capture**, aligning with PyTorch's canonical
  quantization IR.
- **User-extensible by construction** (custom stages, custom
  `pipeline_factory` / `registry`).
- **Module-level export** (`ModuleIORecorder`) for per-layer export and
  numerical validation.
- **QNN-specific transforms stay out of tree by design**, keeping core export
  generic.

**Roadmap:** **ExecuTorch** and **GGUF** targets are planned but **not** yet
registered in the default registry.

---

## 9. Status summary

| Area | Choice | Status |
|---|---|---|
| Core | `QuantizedTensor` = data + metadata | Implemented |
| Core | Identity-Jacobian dequantize; gradients in quantize | Implemented |
| Core | `__torch_function__` + predicate dispatch | Implemented |
| Core | YAML operator table | Implemented |
| Core | Granularity tiling abstraction | Implemented |
| Core | Strict mode + overrides | Implemented |
| Core | No in-place ops; export-mode dequantize | Implemented |
| Modules | Dual-inheritance modules + quantizer stubs (deferred spec) | Implemented |
| Autoquant | Source transpilation (libcst, optional mypy, SSA order) | Implemented |
| Autoquant | Policy-driven / reports / agent-in-loop | Roadmap |
| Orchestration | Graph IR + instruction VM + multi-context delegates | Implemented |
| Orchestration | Lifetime management + offload passes | Implemented |
| Orchestration | Forward data flow support for AdaRound style algorithms | Roadmap |
| Orchestration | Backward data flow support for YAQA style algorithms | Roadmap |
| Orchestration | Automatic load balancing for multi-GPU setups | Roadmap |
| Export | Stage DAG + `(target,format)` registry | Implemented |
| Export | Metadata encodings (versioned) + PT2E capture | Implemented |
| Export | QNN ONNX + ONNX-QDQ pipelines | Implemented |
| Export | ExecuTorch target | Roadmap |
| Export | GGUF target | Roadmap |
