# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import cast

import fastforward as ff
import pytest
import torch
import torch.nn.functional as F

from fastforward.export.stages.base_pipeline_stages import (
    _SampleInputsT,
    stage_capture_impl_ff,
    stage_cleanup_ff_quantizer_artifacts,
    stage_convert_captured_impl_ff,
    stage_convert_captured_impl_ff_qdq,
    stage_fp_eval,
    stage_fuse_qdq_weights,
    stage_passthrough_ff_module,
    stage_quantized_eval,
)
from tests._core_package_version_utils import is_torch_version_at_least


class _QuantizerModule(ff.nn.Quantizer):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def quantize(self, data: torch.Tensor) -> torch.Tensor:
        return data


class _HostModule(ff.nn.QuantizedModule):
    def __init__(self) -> None:
        super().__init__()

    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.used_quantizer = _QuantizerModule()
        self.orphan_quantizer = _QuantizerModule()
        self.call_quantizer = _QuantizerModule()


class _AliasedQuantizerHost(ff.nn.QuantizedModule):
    def __init__(self) -> None:
        super().__init__()

    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        shared_quantizer = _QuantizerModule()
        self.primary_quantizer = shared_quantizer
        self.alias_quantizer = shared_quantizer


class _NestedQuantizerBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner_quantizer = _QuantizerModule()


class _NestedQuantizerHost(ff.nn.QuantizedModule):
    def __init__(self) -> None:
        super().__init__()
        self.block = _NestedQuantizerBlock()


class _DequantizableOutput:
    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def dequantize(self) -> torch.Tensor:
        return self._tensor


class _TensorOutputModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


class _DequantizableOutputModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> _DequantizableOutput:
        return _DequantizableOutput(x + 2)


class _IdentityQuantizedModule(ff.nn.QuantizedModule):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _LinearProbeQuantizedModule(ff.nn.QuantizedModule):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 4))
        self.bias = torch.nn.Parameter(torch.randn(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class _MockExportedProgram:
    def __init__(self, graph_module: torch.fx.GraphModule) -> None:
        self._graph_module = graph_module

    def run_decompositions(
        self, _decomp_table: dict[torch._ops.OperatorBase, object]
    ) -> "_MockExportedProgram":
        return self

    def module(self) -> torch.fx.GraphModule:
        return self._graph_module


class _QuantParamHost(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor([0.5], dtype=torch.float32))
        self.register_buffer("offset", torch.tensor([10.0], dtype=torch.float32))


def _build_mock_exported_program_with_quantize_nodes() -> _MockExportedProgram:
    root = _QuantParamHost()
    graph = torch.fx.Graph()

    input_node = graph.placeholder("x")
    scale_node = graph.get_attr("scale")
    offset_node = graph.get_attr("offset")
    quantize_node = graph.call_function(
        torch.ops.fastforward.quantize_by_tile.default,
        args=(input_node, scale_node, (1,), 8.0, torch.int8, offset_node),
    )
    dequantize_node = graph.call_function(
        torch.ops.fastforward.dequantize_by_tile.default,
        args=(quantize_node, scale_node, (1,), offset_node, torch.float32),
    )
    graph.output(dequantize_node)

    module: torch.fx.GraphModule = torch.fx.GraphModule(root, graph)
    return _MockExportedProgram(module)


def _build_graph_module_with_unused_get_attr_quantizer_reference() -> tuple[
    torch.fx.GraphModule, ff.nn.QuantizedModule
]:
    root = _HostModule()
    graph = torch.fx.Graph()

    input_node = graph.placeholder("x")
    graph.get_attr("used_quantizer.scale")
    graph.output(input_node)

    return torch.fx.GraphModule(root, graph), root


def _build_graph_module_with_get_attr_quantizer_reference() -> tuple[
    torch.fx.GraphModule, ff.nn.QuantizedModule
]:
    root = _HostModule()
    graph = torch.fx.Graph()

    input_node = graph.placeholder("x")
    used_scale = graph.get_attr("used_quantizer.scale")
    output_node = graph.call_function(torch.add, args=(input_node, used_scale))
    graph.output(output_node)

    return torch.fx.GraphModule(root, graph), root


def _build_graph_module_with_call_module_quantizer_reference() -> tuple[
    torch.fx.GraphModule, ff.nn.QuantizedModule
]:
    root = _HostModule()
    graph = torch.fx.Graph()

    input_node = graph.placeholder("x")
    output_node = graph.call_module("call_quantizer", args=(input_node,))
    graph.output(output_node)

    return torch.fx.GraphModule(root, graph), root


def _build_graph_module_with_nested_get_attr_quantizer_reference() -> tuple[
    torch.fx.GraphModule, ff.nn.QuantizedModule
]:
    root = _NestedQuantizerHost()
    graph = torch.fx.Graph()

    input_node = graph.placeholder("x")
    used_scale = graph.get_attr("block.inner_quantizer.scale")
    output_node = graph.call_function(torch.add, args=(input_node, used_scale))
    graph.output(output_node)

    return torch.fx.GraphModule(root, graph), root


def _build_graph_module_with_live_aliased_quantizer_reference() -> tuple[
    torch.fx.GraphModule, ff.nn.QuantizedModule
]:
    root = _AliasedQuantizerHost()
    graph = torch.fx.Graph()

    input_node = graph.placeholder("x")
    alias_scale = graph.get_attr("alias_quantizer.scale")
    output_node = graph.call_function(torch.add, args=(input_node, alias_scale))
    graph.output(output_node)

    return torch.fx.GraphModule(root, graph), root


def _build_graph_module_with_live_direct_quantizer_get_attr_reference() -> tuple[
    torch.fx.GraphModule, ff.nn.QuantizedModule
]:
    root = _AliasedQuantizerHost()
    graph = torch.fx.Graph()

    quantizer_node = graph.get_attr("alias_quantizer")
    graph.output(quantizer_node)

    return torch.fx.GraphModule(root, graph), root


def _build_graph_module_with_live_direct_nested_quantizer_get_attr_reference() -> tuple[
    torch.fx.GraphModule, ff.nn.QuantizedModule
]:
    root = _NestedQuantizerHost()
    graph = torch.fx.Graph()

    quantizer_node = graph.get_attr("block.inner_quantizer")
    graph.output(quantizer_node)

    return torch.fx.GraphModule(root, graph), root


def _graph_targets(module: torch.fx.GraphModule, op: str) -> list[str]:
    return [
        str(node.target)
        for node in module.graph.nodes
        if node.op == op and isinstance(node.target, str)
    ]


def test_stage_capture_impl_ff_raises_when_sample_inputs_is_empty() -> None:
    # GIVEN: A module with no sample inputs provided.
    # WHEN: Capturing the module for export.
    # THEN: The stage should raise a ValueError.
    with pytest.raises(ValueError, match="sample_inputs cannot be empty"):
        stage_capture_impl_ff((_IdentityQuantizedModule(),), [], context={})


def test_stage_capture_impl_ff_returns_exported_program_with_valid_sample_inputs() -> None:
    # GIVEN: A module with a valid sample input.
    module = _IdentityQuantizedModule()
    sample_inputs: _SampleInputsT = [((torch.randn(1, 4),), {})]

    # WHEN: Capturing the module for export.
    exported_program = stage_capture_impl_ff((module,), sample_inputs, context={})

    # THEN: The stage should return an ExportedProgram containing the captured sample inputs.
    assert isinstance(exported_program, torch.export.ExportedProgram)
    captured_args, captured_kwargs = exported_program.example_inputs
    expected_args, expected_kwargs = sample_inputs[0]
    assert len(captured_args) == len(expected_args)
    assert torch.equal(captured_args[0], expected_args[0])
    assert captured_kwargs == expected_kwargs

    # THEN: Replaying the exported program with captured inputs should match the module output.
    with torch.no_grad():
        expected_output = module(*expected_args, **expected_kwargs)
        captured_output = exported_program.module()(*captured_args, **captured_kwargs)
    assert torch.equal(captured_output, expected_output)


def test_stage_convert_captured_impl_ff_returns_graph_module() -> None:
    # GIVEN: A mocked exported program with FF quantize/dequantize nodes.
    sample_inputs: _SampleInputsT = [((torch.randn(1, 4),), {})]
    exported = _build_mock_exported_program_with_quantize_nodes()

    # WHEN: Converting captured export to a quantization-free graph module.
    captured_module = stage_convert_captured_impl_ff(
        (cast(torch.export.ExportedProgram, exported),), sample_inputs, context={}
    )
    call_targets = [
        node.target for node in captured_module.graph.nodes if node.op == "call_function"
    ]

    # THEN: The stage should return a captured FX GraphModule with FF quant nodes removed.
    assert isinstance(captured_module, torch.fx.GraphModule)
    assert torch.ops.fastforward.quantize_by_tile.default not in call_targets
    assert torch.ops.fastforward.dequantize_by_tile.default not in call_targets


def test_stage_convert_captured_impl_ff_qdq_preserves_ff_quant_nodes() -> None:
    # GIVEN: A mocked exported program with FF quantize/dequantize nodes.
    sample_inputs: _SampleInputsT = [((torch.randn(1, 4),), {})]
    exported = _build_mock_exported_program_with_quantize_nodes()

    # WHEN: Converting captured export to a qdq-oriented graph module.
    captured_module = stage_convert_captured_impl_ff_qdq(
        (cast(torch.export.ExportedProgram, exported),), sample_inputs, context={}
    )
    call_targets = [
        node.target for node in captured_module.graph.nodes if node.op == "call_function"
    ]

    # THEN: FF custom quant ops should be preserved for ONNX custom lowering.
    assert isinstance(captured_module, torch.fx.GraphModule)
    assert torch.ops.fastforward.quantize_by_tile.default in call_targets
    assert torch.ops.fastforward.dequantize_by_tile.default in call_targets


def test_stage_capture_impl_ff_respects_torch_export_decomp_table() -> None:
    # GIVEN: A module with a linear op that is decomposed by default.
    module = _LinearProbeQuantizedModule()
    sample_inputs: _SampleInputsT = [((torch.randn(1, 4),), {})]
    if not is_torch_version_at_least("2.6"):
        default_decomp_table = torch._decomp.core_aten_decompositions()
    else:
        default_decomp_table = torch.export.default_decompositions()  # type: ignore[attr-defined,unused-ignore]
    if torch.ops.aten.linear.default not in default_decomp_table:
        pytest.skip("aten.linear.default is not present in the default decomposition table")

    # WHEN: Capturing once and converting with full decompositions,
    # and once excluding linear decomposition.
    exported_default = stage_capture_impl_ff((module,), sample_inputs, context={})
    exported_linear_preserved = stage_capture_impl_ff((module,), sample_inputs, context={})
    captured_default = stage_convert_captured_impl_ff(
        (exported_default,), sample_inputs, context={}
    )
    captured_linear_preserved = stage_convert_captured_impl_ff(
        (exported_linear_preserved,),
        sample_inputs,
        context={"torch_export_decomp_table": {torch.ops.aten.linear.default: True}},
    )
    default_targets = [
        str(node.target) for node in captured_default.graph.nodes if node.op == "call_function"
    ]
    preserved_targets = [
        str(node.target)
        for node in captured_linear_preserved.graph.nodes
        if node.op == "call_function"
    ]

    # THEN: Excluding linear from the decomposition table should keep aten.linear in the captured graph.
    assert "aten.linear.default" not in default_targets
    assert "aten.linear.default" in preserved_targets


def test_stage_cleanup_ff_quantizer_artifacts_prunes_unused_get_attrs_and_succeeds() -> None:
    # GIVEN: A graph module with an unused quantizer get_attr reference.
    module, source_module = _build_graph_module_with_unused_get_attr_quantizer_reference()
    before_targets = _graph_targets(module, "get_attr")
    assert "used_quantizer.scale" in before_targets
    assert hasattr(module, "used_quantizer")
    assert hasattr(source_module, "orphan_quantizer")
    assert hasattr(source_module, "call_quantizer")
    # WHEN: Cleanup stage runs after capture.
    output_module = stage_cleanup_ff_quantizer_artifacts(
        (module, source_module), sample_inputs=[], context={}
    )
    after_targets = _graph_targets(output_module, "get_attr")

    # THEN: Cleanup should remove the unused get_attr and complete successfully.
    assert output_module is module
    assert "used_quantizer.scale" not in after_targets
    assert not hasattr(output_module, "used_quantizer")


def test_stage_cleanup_ff_quantizer_artifacts_raises_with_live_get_attr_reference() -> None:
    # GIVEN: A graph module that still has a get_attr reference to a quantizer.
    module, source_module = _build_graph_module_with_get_attr_quantizer_reference()
    # WHEN: Cleanup stage runs with a live quantizer get_attr reference.
    # THEN: The stage should fail with a clear runtime error.
    with pytest.raises(RuntimeError, match="still references quantizer submodules"):
        stage_cleanup_ff_quantizer_artifacts((module, source_module), sample_inputs=[], context={})


def test_stage_cleanup_ff_quantizer_artifacts_raises_with_live_call_module_reference() -> None:
    # GIVEN: A graph module that still has a call_module reference to a quantizer.
    module, source_module = _build_graph_module_with_call_module_quantizer_reference()
    # WHEN: Cleanup stage runs with a live quantizer call_module reference.
    # THEN: The stage should fail with a clear runtime error.
    with pytest.raises(RuntimeError, match="still references quantizer submodules"):
        stage_cleanup_ff_quantizer_artifacts((module, source_module), sample_inputs=[], context={})


def test_stage_cleanup_ff_quantizer_artifacts_raises_with_live_nested_get_attr_reference() -> None:
    # GIVEN: A graph module with a nested get_attr reference to a quantizer.
    module, source_module = _build_graph_module_with_nested_get_attr_quantizer_reference()
    # WHEN: Cleanup stage runs with a live nested quantizer get_attr reference.
    # THEN: The stage should fail with a clear runtime error.
    with pytest.raises(RuntimeError, match="still references quantizer submodules"):
        stage_cleanup_ff_quantizer_artifacts((module, source_module), sample_inputs=[], context={})


def test_stage_cleanup_ff_quantizer_artifacts_raises_with_live_aliased_quantizer_reference() -> (
    None
):
    # GIVEN: A graph with a live get_attr reference via an alias to a shared quantizer instance.
    module, source_module = _build_graph_module_with_live_aliased_quantizer_reference()
    # WHEN: Cleanup stage validates live quantizer references.
    # THEN: The stage should detect the aliased quantizer reference and fail.
    with pytest.raises(RuntimeError, match="still references quantizer submodules"):
        stage_cleanup_ff_quantizer_artifacts((module, source_module), sample_inputs=[], context={})


def test_stage_cleanup_ff_quantizer_artifacts_raises_with_live_direct_quantizer_get_attr_reference() -> (
    None
):
    # GIVEN: A graph with a live direct get_attr reference to a quantizer module.
    module, source_module = _build_graph_module_with_live_direct_quantizer_get_attr_reference()
    # WHEN: Cleanup stage validates live quantizer references.
    # THEN: The stage should detect the direct quantizer reference and fail.
    with pytest.raises(RuntimeError, match="still references quantizer submodules"):
        stage_cleanup_ff_quantizer_artifacts((module, source_module), sample_inputs=[], context={})


def test_stage_cleanup_ff_quantizer_artifacts_raises_with_live_direct_nested_quantizer_get_attr_reference() -> (
    None
):
    # GIVEN: A graph with a live direct get_attr reference to a nested quantizer module path.
    module, source_module = (
        _build_graph_module_with_live_direct_nested_quantizer_get_attr_reference()
    )
    # WHEN: Cleanup stage validates live quantizer references.
    # THEN: The stage should detect the direct nested quantizer reference and fail.
    with pytest.raises(RuntimeError, match="still references quantizer submodules"):
        stage_cleanup_ff_quantizer_artifacts((module, source_module), sample_inputs=[], context={})


def test_stage_passthrough_ff_module_returns_input_module() -> None:
    # GIVEN: An FF-like module input.
    module = _HostModule()
    # WHEN: The passthrough stage is executed.
    output_module = stage_passthrough_ff_module((module,), sample_inputs=[], context={})
    # THEN: The original module should be returned unchanged.
    assert output_module is module


def test_stage_fp_eval_returns_tensor_outputs() -> None:
    # GIVEN: A module producing tensor outputs and sample inputs.
    module = _TensorOutputModule()
    sample_inputs: _SampleInputsT = [
        ((torch.tensor([1.0]),), {}),
        ((torch.tensor([3.0]),), {}),
    ]

    # WHEN: Running floating-point evaluation stage.
    outputs = stage_fp_eval((module,), sample_inputs=sample_inputs, context={})

    # THEN: Outputs should be evaluated tensors for each input.
    assert len(outputs) == 2
    assert torch.equal(outputs[0], torch.tensor([2.0]))
    assert torch.equal(outputs[1], torch.tensor([4.0]))


def test_stage_quantized_eval_dequantizes_outputs() -> None:
    # GIVEN: A module producing outputs with a dequantize() method and sample inputs.
    module = _DequantizableOutputModule()
    sample_inputs: _SampleInputsT = [
        ((torch.tensor([1.0]),), {}),
        ((torch.tensor([3.0]),), {}),
    ]

    # WHEN: Running quantized evaluation stage.
    outputs = stage_quantized_eval((module,), sample_inputs=sample_inputs, context={})

    # THEN: Stage should return dequantized tensors for each input.
    assert len(outputs) == 2
    assert torch.equal(outputs[0], torch.tensor([3.0]))
    assert torch.equal(outputs[1], torch.tensor([5.0]))


def _quantized_linear_model() -> ff.nn.QuantizedSequential:
    """Build a small initialized quantized Sequential for fuse-stage tests."""
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.Linear(4, 4),
    )
    ff.quantize_model(model)
    assert isinstance(model, ff.nn.QuantizedSequential)
    ff.find_quantizers(model, "**/[quantizer:parameter/weight]").initialize(
        ff.nn.LinearQuantizer, num_bits=4, granularity=ff.PerTensor()
    )
    with ff.estimate_ranges(model, ff.range_setting.smoothed_minmax), ff.strict_quantization(False):
        model(torch.randn(4, 4))
    return model


def _expected_qdq(layer: ff.nn.QuantizedLinear) -> torch.Tensor:
    with ff.strict_quantization(False):
        return layer.weight_quantizer(layer.weight).dequantize()


def _quantized_layers(model: ff.nn.QuantizedSequential) -> list[ff.nn.QuantizedLinear]:
    layers: list[ff.nn.QuantizedLinear] = []
    for layer in model:
        assert isinstance(layer, ff.nn.QuantizedLinear)
        layers.append(layer)
    return layers


def test_stage_fuse_qdq_weights_passes_through_when_flag_absent() -> None:
    # GIVEN: An initialized quantized model.
    module = _quantized_linear_model()

    # WHEN: The fuse stage runs with no context flag.
    result = stage_fuse_qdq_weights((module,), sample_inputs=[], context={})

    # THEN: The original module is returned unchanged (same identity).
    assert result is module


def test_stage_fuse_qdq_weights_fuses_in_place_when_flag_set() -> None:
    # GIVEN: An initialized quantized model.
    module = _quantized_linear_model()
    layers = _quantized_layers(module)
    expected_weights = [_expected_qdq(layer) for layer in layers]

    # WHEN: The fuse stage runs with the flag set.
    result = stage_fuse_qdq_weights(
        (module,), sample_inputs=[], context={"store_weights_as_qdq": True}
    )

    # THEN: The same model object is returned with grid-snapped weights.
    assert result is module
    for layer, expected in zip(_quantized_layers(result), expected_weights):
        torch.testing.assert_close(layer.weight, expected)

    # THEN: Weight quantizers remain active LinearQuantizers.
    for layer in _quantized_layers(result):
        assert isinstance(layer.weight_quantizer, ff.nn.LinearQuantizer)
        assert not layer.weight_quantizer.is_stub()
