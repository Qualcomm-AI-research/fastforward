# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,.nb.py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Export Pipelines
#
# > ⚠️ **WARNING**: Export is an experimental feature and is currently under active development. Please expect API changes. We encourage you to file bug reports if you run into any problems.
#
# The high-level [`ff.export.export`](export_llama.nb.py) function is a thin wrapper around a more general
# *pipeline* machinery. A `Pipeline` is a directed acyclic graph (DAG) of *stages*; FastForward's built-in
# `qnn_onnx_pipeline` is one such graph that captures a quantized FastForward module, runs cleanup passes,
# converts to ONNX, and emits a QNN-compatible encodings file.
#
# The pipeline framework itself is target-agnostic: it is just stages-with-dependencies plus a registry
# keyed by `(target, format)`. Nothing about it is QNN-specific. FastForward already ships a second built-in
# pipeline, `qnn_onnx_qdq_pipeline`, which is a plain ONNX export with the quantizers embedded as standard
# `QuantizeLinear`/`DequantizeLinear` (QDQ) nodes — no QNN-specific encodings file at all. The same
# machinery could host an export to any other backend or format.
#
# > 💡 We export to QNN by default because QNN (the Qualcomm AI Engine Direct SDK) is Qualcomm's primary
# > runtime for on-device AI inference, so QNN-formatted artifacts are what most FastForward users need
# > to deploy to Qualcomm hardware.
#
# This tutorial focuses on the pipeline layer underneath `export`. By the end you will know:
#
# 1. How to build a pipeline from scratch by registering stages and wiring up dependencies.
# 2. How to run the built-in QNN/ONNX pipeline directly, without going through `export`.
# 3. How to manipulate an existing pipeline: insert stages before/after a target, replace a stage in place,
#    and add or remove dependency edges.
# 4. How to register your own pipeline factory in a `PipelineRegistry` so `export(...)` picks it up.
#
# We use a tiny convolutional network so the focus stays on the pipeline mechanics rather than on model setup.
# If you want to see the same machinery applied to a real LLM, see the [LLaMA export tutorial](export_llama.nb.py).

# %% [markdown]
# ## Setup

# %%
import json
import logging
import warnings

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import fastforward as ff
import torch

from fastforward.export import (
    ExportArtifacts,
    ExportRequest,
    QnnOnnxOptions,
    export,
    export_with_pipeline,
)
from fastforward.export.pipeline import (
    Pipeline,
    build_default_registry,
)
from fastforward.export.pipeline.qnn_onnx_pipeline import qnn_onnx_pipeline
from fastforward.export.stages.onnx.onnx_export_stages import stage_save_onnx_proto

warnings.filterwarnings("ignore")
logging.getLogger("torch.onnx._internal._registration").setLevel(logging.ERROR)
logging.getLogger("torch.onnx._internal.exporter").setLevel(logging.ERROR)

torch.set_grad_enabled(False);  # fmt: skip  # noqa: E703

# %% [markdown]
# ### A small quantized ConvNet
#
# We define a 4-layer ConvNet using FastForward's quantized building blocks (`QuantizedConv2d`,
# `QuantizedRelu`, `QuantizedLinear`). Then we use `ff.QuantizationConfig` to attach `LinearQuantizer`s
# to all weight, input and output positions, and run a single forward pass under `estimate_ranges` to
# calibrate the quantization ranges. This is the same pattern used in the
# [Quantizing Networks](quantizing_networks.nb.py) tutorial, just on a smaller model.


# %%
class SimpleConvNet(torch.nn.Module):
    """A tiny ConvNet with two strided conv blocks followed by a classifier head."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Use stride-2 convs to reduce spatial dims so we do not need a pooling op,
        # which keeps the graph entirely composed of FastForward-quantized modules.
        self.conv1 = ff.nn.QuantizedConv2d(3, 8, kernel_size=3, stride=2, padding=1)
        self.relu1 = ff.nn.QuantizedRelu()
        self.conv2 = ff.nn.QuantizedConv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.relu2 = ff.nn.QuantizedRelu()
        # 32x32 input -> 16x16 -> 8x8, flattened to 1024 features.
        self.classifier = ff.nn.QuantizedLinear(16 * 8 * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.flatten(1)
        return self.classifier(x)


model = SimpleConvNet()

config = ff.QuantizationConfig()
config.add_rule(
    "**/[quantizer:parameter/weight]",
    ff.nn.LinearQuantizer,
    num_bits=8,
    symmetric=True,
    granularity=ff.PerChannel(),
)
config.add_rule(
    "**/[quantizer:activation/output]",
    ff.nn.LinearQuantizer,
    num_bits=8,
    symmetric=False,
    granularity=ff.PerTensor(),
)
config.add_rule(
    "conv1/[quantizer:activation/input]",
    ff.nn.LinearQuantizer,
    num_bits=8,
    symmetric=False,
    granularity=ff.PerTensor(),
)
config.initialize(model)

calibration_data = torch.randn(2, 3, 32, 32)
# Range estimation runs the model in floating point with quantization disabled at the
# tensor level. We disable strict_quantization here so that any non-quantized ops in the
# model (such as `flatten`) do not raise. The `export(...)` entry point wraps its
# internals in the same flag automatically.
with (
    ff.strict_quantization(False),
    ff.range_setting.estimate_ranges(model, ff.range_setting.smoothed_minmax),
):
    model(calibration_data)

model.eval()
sample_input = torch.randn(1, 3, 32, 32)

# %% [markdown]
# We now have a quantized, calibrated `SimpleConvNet` and a `sample_input` we can trace through it.

# %% [markdown]
# ## 1. Anatomy of a `Pipeline`
#
# A `Pipeline` is a DAG of stages. Each stage is a callable with the signature:
#
# ```python
# def stage(modules: tuple[Any, ...], sample_inputs: list[tuple[args, kwargs]], context: dict[str, Any]) -> Any: ...
# ```
#
# - `modules` is a tuple containing the outputs of the stage's dependencies (or, for stages with no
#   dependencies, the pipeline input). The pipeline input for `Pipeline.__call__(module, sample_inputs)`
#   is the `module`.
# - `sample_inputs` is a list of `(args, kwargs)` pairs, the same value passed to `Pipeline.__call__`.
# - `context` is the pipeline-wide kwargs dict passed to `Pipeline(pipeline_kwargs=...)`.
#
# The pipeline runs stages in topological order, threading each stage's return value to its dependents.
# Stages registered with `capture_stage_output=True` show up in the pipeline's return value.
#
# Let's build a minimal pipeline with three toy stages over our quantized ConvNet.


# %%
def stage_passthrough(
    modules: tuple[torch.nn.Module, ...],
    sample_inputs: list[tuple[tuple[Any, ...], dict[str, Any]]],
    context: dict[str, Any],
) -> torch.nn.Module:
    del sample_inputs, context
    (module,) = modules
    return module


def stage_count_params(
    modules: tuple[torch.nn.Module, ...],
    sample_inputs: list[tuple[tuple[Any, ...], dict[str, Any]]],
    context: dict[str, Any],
) -> int:
    del sample_inputs, context
    (module,) = modules
    return sum(p.numel() for p in module.parameters())


def stage_summarize(
    modules: tuple[torch.nn.Module | int, ...],
    sample_inputs: list[tuple[tuple[Any, ...], dict[str, Any]]],
    context: dict[str, Any],
) -> dict[str, Any]:
    del sample_inputs
    module, num_params = modules
    return {
        "model_name": context["model_name"],
        "num_params": num_params,
        "module_classes": sorted({type(m).__name__ for m in module.modules()}),
    }


toy_pipeline = Pipeline(pipeline_kwargs={"model_name": "simple_convnet"})
src = toy_pipeline.register_stage(stage_passthrough, "src")
counts = toy_pipeline.register_stage(stage_count_params, "count_params").depends_on(src)
toy_pipeline.register_stage(stage_summarize, "summary", capture_stage_output=True).depends_on(
    src, counts
)

# Note: pipelines expect a *list* of (args, kwargs) sample inputs.
toy_results, _ = toy_pipeline(model, [((sample_input,), {})])
toy_results

# %% [markdown]
# ✅ Only the stage we marked with `capture_stage_output=True` shows up in the returned dict. The pipeline
# resolved the order `src -> count_params -> summary` automatically from the dependency edges; you do not
# need to register stages in any particular order.

# %% [markdown]
# ## 2. Running the built-in QNN/ONNX pipeline
#
# FastForward ships a `qnn_onnx_pipeline` factory that builds the full export DAG: capture, cleanup,
# ONNX conversion, encoding extraction. There are three layers of API for running it, in order of decreasing
# abstraction:
#
# 1. `export(model, data, output_dir, model_name, ...)` — the convenience entry point used in the
#    [LLaMA tutorial](export_llama.nb.py).
# 2. `export_with_pipeline(ExportRequest(...))` — same machinery, but you build an `ExportRequest` yourself.
#    Useful when you want to swap in a custom registry or pipeline factory without touching `export`'s kwargs.
# 3. Constructing a `Pipeline` directly from `qnn_onnx_pipeline(context)` and calling it. This is what you
#    drop down to when you want to inspect or mutate the DAG.
#
# Let us run all three on the same model to confirm they produce equivalent artifacts.

# %%
work_dir = TemporaryDirectory()
work_root = Path(work_dir.name)


def list_artifacts(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


# Layer 1: the high-level `export` function.
out_dir_high = work_root / "high_level"
export(
    model=model,
    data=(sample_input,),
    output_directory=out_dir_high,
    model_name="convnet_high",
    verbose=False,
)
list_artifacts(out_dir_high)

# %%
# Layer 2: build an ExportRequest and call `export_with_pipeline`.
out_dir_mid = work_root / "mid_level"
out_dir_mid.mkdir(parents=True, exist_ok=True)
options = QnnOnnxOptions(verbose=False)
request = ExportRequest(
    model=model,
    sample_inputs=[((sample_input,), {})],
    output_dir=out_dir_mid,
    model_name="convnet_mid",
    target="qnn",
    format="onnx",
    options=options.to_context(),
)
artifacts: ExportArtifacts = export_with_pipeline(request)
print("pipeline_name:", artifacts.pipeline_name)
list_artifacts(out_dir_mid)

# %%
# Layer 3: build the pipeline directly. The factory takes a context dict, the same one
# the orchestrator would have built from your ExportRequest.
out_dir_low = work_root / "low_level"
out_dir_low.mkdir(parents=True, exist_ok=True)
context = {
    "output_dir": out_dir_low,
    "model_name": "convnet_low",
    **QnnOnnxOptions(verbose=False).to_context(),
}
pipeline = qnn_onnx_pipeline(context)
print("registered stages:")
for stage in pipeline._stages:
    deps = [dep.name for dep in stage.dependencies]
    print(f"  - {stage.name:40s} deps={deps}")

with ff.export_mode(True):
    stage_outputs, eval_results = pipeline(model, [((sample_input,), {})])

list_artifacts(out_dir_low)

# %% [markdown]
# ✅ All three calls produce the same `<model_name>.onnx` and `<model_name>.encodings` pair. The difference
# is only how much of the plumbing you exposed. Layers 2 and 3 are where pipeline manipulation becomes
# possible.
#
# A few details worth noting about layer 3:
#
# - We had to enter `ff.export_mode(True)` ourselves. The `export` and `export_with_pipeline` entry points
#   do this for you. When you call a `Pipeline` directly you are responsible for the right context managers.
# - The `_stages` mapping is internal but convenient for visualization. The supported way to look up a
#   stage by name is `pipeline.get_stage("name")`.

# %% [markdown]
# ### 2.1 A second built-in pipeline: ONNX QDQ
#
# To make the point that the framework is not QNN-specific, the default registry actually holds *two*
# pipelines under the `qnn` target:
#
# - `("qnn", "onnx")` → `qnn_onnx_pipeline` — ONNX graph plus a side-channel `.encodings` file (what we ran
#   above).
# - `("qnn", "onnx_qdq")` → `qnn_onnx_qdq_pipeline` — a plain ONNX export where each FastForward
#   quantize/dequantize op is lowered to a standard ONNX `QuantizeLinear`/`DequantizeLinear` pair. The
#   quantization parameters live in the graph topology itself, so **no encodings file is produced**.
#
# Selecting it is just a matter of the `format` argument. Note the QDQ flow requires ONNX opset 21+
# (where INT4/INT16 storage in Q/DQ nodes was introduced); the pipeline defaults to opset 21 for you.

# %%
out_dir_qdq = work_root / "qdq"
export(
    model=model,
    data=(sample_input,),
    output_directory=out_dir_qdq,
    model_name="convnet_qdq",
    target="qnn",
    format="onnx_qdq",
    verbose=False,
)
list_artifacts(out_dir_qdq)

# %% [markdown]
# ✅ Compare the artifact list with the QNN/ONNX runs above: there is a `convnet_qdq.onnx` but no
# `.encodings` file, because the quantization information is baked into the ONNX graph as Q/DQ nodes. Same
# pipeline framework, different export format — and nothing forces a pipeline to target QNN or even ONNX.

# %% [markdown]
# ## 3. Manipulating the built-in pipeline
#
# `Pipeline` exposes four primitives for mutating an already-built DAG:
#
# - `insert_stage_before(target, fn, name)` — splice a new stage so that `target` now depends on it.
# - `insert_stage_after(target, fn, name)` — splice a new stage so that everything that used to depend on
#   `target` now depends on the new stage instead.
# - `replace_stage(target, fn, name=None)` — drop-in swap that preserves both incoming and outgoing edges
#   plus capture/eval flags.
# - `add_dependency(stage, dep)` / `remove_dependency(stage, dep)` — edit individual edges. Cycles are
#   rejected at the call site rather than at build time.
#
# We will demonstrate each on a fresh `qnn_onnx_pipeline`.

# %% [markdown]
# ### 3.1 Insert before
#
# Suppose we want to log the size of the ONNX proto right before it is written to disk. The natural place
# is *before* the `save_onnx_proto` stage. The default behavior of `insert_stage_before` is to inherit the
# target's current dependencies and rewire the target to depend on the new stage — i.e. it splices into
# the chain.


# %%
def stage_log_proto_size(
    modules: tuple[Any, ...],
    sample_inputs: list[tuple[tuple[Any, ...], dict[str, Any]]],
    context: dict[str, Any],
) -> Any:
    del sample_inputs, context
    (proto,) = modules
    print(f"  [hook] onnx proto has {len(proto.graph.node)} nodes before save")
    return proto


out_dir_insert = work_root / "insert_before"
out_dir_insert.mkdir(parents=True, exist_ok=True)
context = {
    "output_dir": out_dir_insert,
    "model_name": "convnet_insert_before",
    **QnnOnnxOptions(verbose=False).to_context(),
}
pipeline = qnn_onnx_pipeline(context)
pipeline.insert_stage_before("save_onnx_proto", stage_log_proto_size, name="log_proto_size")

with ff.export_mode(True):
    pipeline(model, [((sample_input,), {})])

# %% [markdown]
# ✅ The hook ran between `copy_metadata_props_from_ir_to_proto` and `save_onnx_proto` — the original
# upstream of `save_onnx_proto`. Because we did not pass `depends_on=`, `save_onnx_proto` was rewired
# to depend on our new stage. If you instead pass `depends_on=`, the new stage is wired only to those
# dependencies and the target's own dependencies are left untouched (useful for side-branch stages).

# %% [markdown]
# ### 3.2 Insert after
#
# `insert_stage_after` is the mirror image: every stage that previously depended on `target` is rewired
# to depend on the new stage. This is the right primitive when you want every downstream consumer to see
# your modified output.


# %%
def stage_strip_metadata(
    modules: tuple[Any, ...],
    sample_inputs: list[tuple[tuple[Any, ...], dict[str, Any]]],
    context: dict[str, Any],
) -> Any:
    """Toy transform: clear all node-level metadata after the rename pass."""
    del sample_inputs, context
    (program,) = modules
    for node in program.model.graph:
        node.metadata_props.clear()
    return program


out_dir_after = work_root / "insert_after"
out_dir_after.mkdir(parents=True, exist_ok=True)
context = {
    "output_dir": out_dir_after,
    "model_name": "convnet_insert_after",
    **QnnOnnxOptions(verbose=False).to_context(),
}
pipeline = qnn_onnx_pipeline(context)
pipeline.insert_stage_after(
    "rename_onnx_input_output_names", stage_strip_metadata, name="strip_node_metadata"
)

# Confirm that `onnx_program_to_proto` and `copy_metadata_props_from_ir_to_proto`,
# the two stages that previously depended on `rename_onnx_input_output_names`, are
# now wired to our new stage.
for name in ("onnx_program_to_proto", "copy_metadata_props_from_ir_to_proto"):
    deps = [dep.name for dep in pipeline.get_stage(name).dependencies]
    print(f"  {name}.dependencies = {deps}")

# %% [markdown]
# ⚠️ Use `insert_stage_after` with care — it changes what every downstream stage sees. If you only want
# a side-branch that observes `target` without displacing existing consumers, use `register_stage` plus
# `add_dependency` instead.

# %% [markdown]
# ### 3.3 Replace a stage in place
#
# `replace_stage` swaps a stage's callable while preserving its position in the graph: the replacement
# inherits the original's dependencies, every existing dependent is rewired to the replacement, and the
# `capture_stage_output` / evaluation-stage flags carry over by default.
#
# Here we replace `save_onnx_proto` with a variant that also writes a small JSON sidecar describing the
# saved model.


# %%
def stage_save_onnx_proto_with_sidecar(
    modules: tuple[Any, ...],
    sample_inputs: list[tuple[tuple[Any, ...], dict[str, Any]]],
    context: dict[str, Any],
) -> Any:
    proto = stage_save_onnx_proto(modules, sample_inputs, context)
    sidecar = Path(context["output_dir"]) / f"{context['model_name']}.summary.json"
    sidecar.write_text(json.dumps({"num_nodes": len(proto.graph.node)}, indent=2))
    return proto


out_dir_replace = work_root / "replace"
out_dir_replace.mkdir(parents=True, exist_ok=True)
context = {
    "output_dir": out_dir_replace,
    "model_name": "convnet_replace",
    **QnnOnnxOptions(verbose=False).to_context(),
}
pipeline = qnn_onnx_pipeline(context)
pipeline.replace_stage("save_onnx_proto", stage_save_onnx_proto_with_sidecar)

with ff.export_mode(True):
    pipeline(model, [((sample_input,), {})])

list_artifacts(out_dir_replace)

# %% [markdown]
# ✅ Alongside `convnet_replace.onnx` and `convnet_replace.encodings`, we now have a
# `convnet_replace.summary.json` sidecar produced by our replacement stage. The
# `onnx_proto_to_encodings` stage downstream still ran correctly because its dependency on
# `save_onnx_proto` was rewired transparently to the replacement.

# %% [markdown]
# ### 3.4 Add and remove dependency edges
#
# When you need finer control than insert/replace, `add_dependency` and `remove_dependency` edit
# individual edges. Adding an edge that would introduce a cycle is rejected immediately:

# %%
context = {
    "output_dir": work_root / "edges",
    "model_name": "convnet_edges",
    **QnnOnnxOptions(verbose=False).to_context(),
}
pipeline = qnn_onnx_pipeline(context)

# Adding a redundant edge is a no-op.
pipeline.add_dependency("save_onnx_proto", "copy_metadata_props_from_ir_to_proto")

# Adding an edge that creates a cycle raises immediately.
try:
    pipeline.add_dependency("capture_ff", "save_onnx_proto")
except ValueError as exc:
    print("rejected cycle:", exc)

# Edges can also be removed. Here we drop the secondary dependency that
# `cleanup_ff_quantizer_artifacts` has on `source_ff_module`. The pipeline will then fail to build because
# the cleanup stage's signature still expects two inputs - cycles aside, edge edits are not validated
# against stage signatures.
pipeline.remove_dependency("cleanup_ff_quantizer_artifacts", "source_ff_module")
remaining = [dep.name for dep in pipeline.get_stage("cleanup_ff_quantizer_artifacts").dependencies]
print("cleanup_ff_quantizer_artifacts deps after removal:", remaining)

# %% [markdown]
# ⚠️ `remove_dependency` will not stop you from putting the pipeline into a state that fails at execution
# time (a stage that expects N inputs but only has M dependencies). Edge edits are mechanical; keeping
# stage signatures consistent is your responsibility.

# %% [markdown]
# ## 4. Custom pipeline factories and the registry
#
# The `ExportOrchestrator` resolves which pipeline to build by looking up the `(target, format)` pair on
# a `PipelineRegistry`. The default registry holds the two built-in entries we saw earlier —
# `("qnn", "onnx") -> qnn_onnx_pipeline` and `("qnn", "onnx_qdq") -> qnn_onnx_qdq_pipeline`. You can register
# additional factories under new keys, or replace an existing entry, and `export(...)` will pick them up
# via the `registry=` keyword.
#
# Below we wrap the built-in pipeline in a thin custom factory that adds a logging stage, register it
# under a new key, and run `export` against that key.


# %%
def my_logging_pipeline(pipeline_kwargs: dict[str, Any]) -> Pipeline:
    pipeline = qnn_onnx_pipeline(pipeline_kwargs)
    pipeline.insert_stage_before("save_onnx_proto", stage_log_proto_size, name="log_proto_size")
    return pipeline


registry = build_default_registry()
registry.register("qnn", "onnx-with-logging", my_logging_pipeline)

# %% [markdown]
# Now `export(...)` will resolve to our custom pipeline whenever it is called with
# `target="qnn"` and `format="onnx-with-logging"`:

# %%
out_dir_custom = work_root / "custom_factory"
export(
    model=model,
    data=(sample_input,),
    output_directory=out_dir_custom,
    model_name="convnet_custom",
    target="qnn",
    format="onnx-with-logging",
    registry=registry,
    verbose=False,
)
list_artifacts(out_dir_custom)

# %% [markdown]
# You can also pass a `pipeline_factory=` directly to `export(...)` if you only want a one-off override
# without touching the registry. Use the registry when you want the factory to be discoverable by
# `(target, format)` and reusable across calls.

# %% [markdown]
# ## Cleanup

# %%
work_dir.cleanup()

# %% [markdown]
# ## Conclusion
#
# We have walked from the ground up: building a `Pipeline` by hand, running the built-in `qnn_onnx_pipeline`
# at three different abstraction levels, mutating it with the four manipulation primitives, and finally
# registering a custom factory in a `PipelineRegistry` so it surfaces through the standard `export(...)`
# entry point.
#
# Pick the right level of abstraction for the task at hand:
#
# - Use `export(...)` when you just want artifacts on disk.
# - Drop down to `export_with_pipeline(ExportRequest(...))` when you need a custom registry or pipeline
#   factory.
# - Construct a `Pipeline` directly when you are inspecting or mutating the DAG, writing tests, or
#   prototyping new stages.
#
# Related tutorials:
#
# - [Exporting tinyLlama](export_llama.nb.py) — the same machinery applied to a real LLM.
# - [MPath: a utility for submodule selection](mpath.nb.py) — useful when authoring stages that need to
#   target specific parts of a model.

# %% [markdown]
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
