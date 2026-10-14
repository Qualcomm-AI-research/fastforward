# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for the quant-dequant kernel."""

from typing import Any, Callable, Sequence, TypeAlias

import fastforward as ff
import pytest
import torch

from fastforward.nn.linear_quantizer import LinearQuantizer
from fastforward.quantization.affine import _ops
from fastforward.quantization.affine._autograd import (
    affine_dequantize_fn,
    affine_static_qdq_fn,
    affine_static_quantize_fn,
)
from torch._functorch.aot_autograd import aot_module_simplified

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

IntTuple: TypeAlias = tuple[int, ...]

# `(data_shape, tile_size)` pairs covering per-tensor, per-channel over each
# axis, and multi-dimensional tiling.
SHAPES_AND_TILES: list[tuple[IntTuple, IntTuple]] = [
    ((8,), (8,)),
    ((4, 6), (4, 6)),
    ((4, 6), (1, 6)),
    ((4, 6), (4, 1)),
    ((2, 4, 6), (1, 4, 6)),
    ((2, 4, 6), (2, 1, 6)),
    ((2, 4, 6), (2, 4, 1)),
    ((2, 4, 6), (1, 1, 6)),
    ((8, 16), (4, 8)),
    ((6, 10, 4), (3, 5, 2)),
]


def _quant_params(
    data_shape: IntTuple,
    tile_size: IntTuple,
    device: str,
    with_offset: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Build data, scale and (optionally) offset for a shape/tile combination."""
    generator = torch.Generator(device="cpu").manual_seed(1234)
    num_tiles = torch.Size(data_shape).numel() // torch.Size(tile_size).numel()

    data = torch.randn(*data_shape, generator=generator) * 2.0
    scale = torch.rand(num_tiles, generator=generator) * 0.05 + 0.01
    offset = torch.round(torch.randn(num_tiles, generator=generator) * 5.0) if with_offset else None

    data = data.to(device)
    scale = scale.to(device)
    offset = offset.to(device) if offset is not None else None
    return data, scale, offset


# ------------------------------------------------------------------------------
# Op level: fused QdQ vs. the reference two-step compositions
# ------------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("data_shape", "tile_size"), SHAPES_AND_TILES)
@pytest.mark.parametrize("with_offset", [False, True], ids=["symmetric", "asymmetric"])
def test_qdq_op_matches_reference_quantize_then_dequantize(
    device: str, data_shape: IntTuple, tile_size: IntTuple, with_offset: bool
) -> None:
    # GIVEN data and quantization parameters for a tile configuration
    data, scale, offset = _quant_params(data_shape, tile_size, device, with_offset)

    # WHEN quantizing then dequantizing with the reference two-step path, and
    # quantizing-and-dequantizing in one call with the fused broadcast op
    quantized = _ops.affine_static_quantize_op(data, scale, tile_size, 8, None, offset)
    expected = _ops.affine_dequantize_op(quantized, scale, tile_size, offset, None)
    actual = _ops.affine_static_qdq_op(data, scale, tile_size, 8, None, offset)

    # THEN the results are bit-identical
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("num_bits", [2, 4, 8, 16])
def test_qdq_op_respects_bitwidth(device: str, num_bits: int) -> None:
    # GIVEN data that exceeds the quantization range on both sides
    data, scale, offset = _quant_params((4, 16), (4, 16), device, with_offset=False)

    # WHEN quantize-dequantizing at a given bitwidth with the fused op, and
    # with the reference two-step path
    quantized = _ops.affine_static_quantize_op(data, scale, (4, 16), num_bits, None, offset)
    expected = _ops.affine_dequantize_op(quantized, scale, (4, 16), offset, None)
    actual = _ops.affine_static_qdq_op(data, scale, (4, 16), num_bits, None, offset)

    # THEN the fused op clamps to the same integer grid as the reference path
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("data_shape", "tile_size"), SHAPES_AND_TILES)
@pytest.mark.parametrize("with_offset", [False, True], ids=["symmetric", "asymmetric"])
def test_export_op_matches_plain_python_op(
    device: str, data_shape: IntTuple, tile_size: IntTuple, with_offset: bool
) -> None:
    # GIVEN data, quantization parameters and an output gradient for a tile configuration
    data, scale, offset = _quant_params(data_shape, tile_size, device, with_offset)
    output_grad = torch.cos(data)

    # WHEN the qdq op and its backward are run through the default (plain) implementation,
    # and through the registered torch op selected by `export_mode`
    plain_qdq = _ops.affine_static_qdq_op(data, scale, tile_size, 8, None, offset)
    plain_grads = _ops.affine_quantize_backward_op(data, output_grad, scale, tile_size, 8, offset)
    with ff.export_mode(True):
        export_qdq = _ops.affine_static_qdq_op(data, scale, tile_size, 8, None, offset)
        export_grads = _ops.affine_quantize_backward_op(
            data, output_grad, scale, tile_size, 8, offset
        )

    # THEN the outputs and all gradients are bit-identical
    assert torch.equal(plain_qdq, export_qdq)
    assert len(plain_grads) == len(export_grads) == 3
    for plain_grad, custom_grad in zip(plain_grads, export_grads, strict=True):
        assert torch.equal(plain_grad, custom_grad)


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:.*should not be instantiated:DeprecationWarning")
def test_qdq_export_keeps_registered_op() -> None:
    # GIVEN a module applying the fused qdq function
    class QdqModule(torch.nn.Module):
        def forward(self, data: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            return affine_static_qdq_fn(data, scale, None, torch.Size((4, 6)), 8, None)

    # WHEN it is exported
    with ff.export_mode(True), torch.no_grad():
        exported = torch.export.export(QdqModule(), (torch.randn(4, 6), torch.ones(1)))

    # THEN the registered qdq op is kept as an opaque node
    targets = [n.target for n in exported.graph_module.graph.nodes if n.op == "call_function"]
    assert torch.ops.fastforward.affine_static_qdq.default in targets


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:.*should not be instantiated:DeprecationWarning")
def test_quantize_dequantize_export_keep_registered_ops() -> None:
    # GIVEN a module applying the (non-fused) quantize and dequantize functions
    class QuantDequantModule(torch.nn.Module):
        def forward(self, data: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            quantized = affine_static_quantize_fn(data, scale, None, torch.Size((4, 6)), 8, None)
            return affine_dequantize_fn(quantized, scale, None, torch.Size((4, 6)), None)

    # WHEN it is exported with export mode enabled
    with ff.export_mode(True), torch.no_grad():
        exported = torch.export.export(QuantDequantModule(), (torch.randn(4, 6), torch.ones(1)))

    # THEN the registered quantize and dequantize ops are kept as opaque nodes
    targets = [n.target for n in exported.graph_module.graph.nodes if n.op == "call_function"]
    assert torch.ops.fastforward.affine_static_quantize.default in targets
    assert torch.ops.fastforward.affine_dequantize.default in targets


# --------------------------------------------------------------------------------------
# Autograd wrapper level: the full quantize-dequantize matrix, forward and backward
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("data_shape", "tile_size"), SHAPES_AND_TILES)
@pytest.mark.parametrize("with_offset", [False, True], ids=["symmetric", "asymmetric"])
def test_mode_matches_reference_end_to_end(
    device: str, data_shape: IntTuple, tile_size: IntTuple, with_offset: bool
) -> None:
    """Every implementation must reproduce the reference output and gradients."""
    # GIVEN the same data and parameters, quantized and dequantized through the
    # reference implementation and qdq mode
    data, scale, offset = _quant_params(data_shape, tile_size, device, with_offset)
    num_bits = 8

    ref_dequant, ref_data_grad, ref_scale_grad, ref_offset_grad = _run_quant_dequant(
        data, scale, offset, tile_size, num_bits, use_qdq_fn=False
    )
    actual_dequant, actual_data_grad, actual_scale_grad, actual_offset_grad = _run_quant_dequant(
        data, scale, offset, tile_size, num_bits, use_qdq_fn=True
    )

    # THEN the dequantized output and data gradient are bit-identical, and the
    # scale/offset gradients agree up to floating point reassociation
    assert torch.equal(actual_dequant, ref_dequant)
    assert torch.equal(actual_data_grad, ref_data_grad)
    torch.testing.assert_close(actual_scale_grad, ref_scale_grad, rtol=1e-4, atol=1e-4)
    if with_offset:
        assert ref_offset_grad is not None
        assert actual_offset_grad is not None
        torch.testing.assert_close(actual_offset_grad, ref_offset_grad, rtol=1e-4, atol=1e-4)


def _run_quant_dequant(
    data: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor | None,
    tile_size: IntTuple,
    num_bits: int,
    use_qdq_fn: bool = False,
    qdq_fn: Callable[..., torch.Tensor] = affine_static_qdq_fn,
    quantized_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run a full quantize-then-dequantize computation.

    Returns:
        A `(dequantized, data_grad, scale_grad, offset_grad)` tuple, where
        `offset_grad` is `None` if `offset` is `None`.
    """
    size = torch.Size(tile_size)
    data = data.detach().clone().requires_grad_(True)
    scale = scale.detach().clone().requires_grad_(True)
    offset = offset.detach().clone().requires_grad_(True) if offset is not None else None

    with ff.qdq_mode(use_qdq_fn):
        if use_qdq_fn:
            dequantized = qdq_fn(data, scale, offset, size, num_bits, quantized_dtype)
        else:
            quantized = affine_static_quantize_fn(
                data, scale, offset, size, num_bits, quantized_dtype
            )
            dequantized = affine_dequantize_fn(quantized, scale, offset, size, None)

    # A non-uniform objective makes the scale/offset gradients non-trivial.
    dequantized.pow(2).sum().backward()
    offset_grad = offset.grad if offset is not None else None
    assert data.grad is not None
    assert scale.grad is not None
    return dequantized.detach(), data.grad, scale.grad, offset_grad


# --------------------------------------------------------------------------------------
# End to end through the quantizer
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "granularity_factory",
    [ff.PerTensor, lambda: ff.PerChannel(0), lambda: ff.PerChannel(1)],
    ids=["per-tensor", "per-channel-0", "per-channel-1"],
)
@pytest.mark.parametrize("symmetric", [True, False], ids=["symmetric", "asymmetric"])
def test_qdq_mode_matches_reference_end_to_end(
    device: str, granularity_factory: Any, symmetric: bool
) -> None:
    """`qdq_mode` alone must not change quantizer results."""
    data = torch.randn(8, 16, device=device, requires_grad=True)
    granularity = granularity_factory()
    dimensionality = granularity.parameter_dimensionality(torch.Size((8, 16)))
    lower = torch.full((dimensionality,), -3.0, device=device)
    quantizer = LinearQuantizer(
        num_bits=8, granularity=granularity, symmetric=symmetric, device=torch.device(device)
    )
    quantizer.quantization_range = (lower, -lower)

    # GIVEN the same quantizer and data, run with `qdq_mode` off and on
    ref_out, ref_data_grad, ref_scale_grad = _run_fwd_bck(data, quantizer, qdq=False)
    qdq_out, qdq_data_grad, qdq_scale_grad = _run_fwd_bck(data, quantizer, qdq=True)

    # THEN outputs and data gradients are bit-identical and scale gradients match
    assert torch.equal(qdq_out, ref_out)
    assert torch.equal(qdq_data_grad, ref_data_grad)
    torch.testing.assert_close(qdq_scale_grad, ref_scale_grad, rtol=1e-4, atol=1e-4)


def _run_fwd_bck(
    data: torch.Tensor, quantizer: LinearQuantizer, qdq: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with ff.qdq_mode(qdq):
        output = quantizer(data)
    if not qdq:
        output = output.dequantize()

    # A non-uniform objective makes the scale gradient non-trivial.
    output.pow(2).sum().backward()

    assert data.grad is not None
    assert quantizer.scale.grad is not None
    return output, data.grad, quantizer.scale.grad


# --------------------------------------------------------------------------------------
# Compiled path: under torch.compile the qdq autograd function traces the plain torch
# implementation instead of the custom op. The results must not change.
# --------------------------------------------------------------------------------------


def _capturing_backend(graphs: list[str]) -> Callable[..., Any]:
    """Return a torch.compile backend that records the code of every graph it compiles."""

    def capture(gm: torch.fx.GraphModule, example_inputs: Sequence[Any]) -> Any:
        del example_inputs
        graphs.append(gm.code)

        def boxed_forward(args: list[torch.Tensor]) -> Any:
            return gm.forward(*args)

        # AOTAutograd expects a compiled callable that takes its inputs as one list.
        boxed_forward._boxed_call = True  # type: ignore[attr-defined]
        return boxed_forward

    def backend(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]) -> Any:
        return aot_module_simplified(gm, example_inputs, fw_compiler=capture, bw_compiler=capture)

    return backend


@pytest.mark.slow
# Torch 2.8 - 2.13 instantiate autograd.Function classes while dynamo traces them,
# emitting a DeprecationWarning.
@pytest.mark.filterwarnings("ignore:.*should not be instantiated:DeprecationWarning")
# Importing torch._inductor on torch 2.11+ emits the torch.jit.script_method DeprecationWarning.
@pytest.mark.filterwarnings("ignore:.*torch.jit.script_method. is deprecated:DeprecationWarning")
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("data_shape", "tile_size"), SHAPES_AND_TILES)
@pytest.mark.parametrize("num_bits", [4, 8])
@pytest.mark.parametrize("with_offset", [False, True], ids=["symmetric", "asymmetric"])
def test_qdq_compiled_matches_eager_custom_op(
    device: str, data_shape: IntTuple, tile_size: IntTuple, num_bits: int, with_offset: bool
) -> None:
    # GIVEN the eager output and gradients, computed through the custom op
    torch._dynamo.reset()
    data, scale, offset = _quant_params(data_shape, tile_size, device, with_offset)
    ref_dequant, ref_data_grad, ref_scale_grad, ref_offset_grad = _run_quant_dequant(
        data, scale, offset, tile_size, num_bits, use_qdq_fn=True
    )

    # WHEN the same computation is compiled, forward and backward
    graphs: list[str] = []
    compiled_qdq = torch.compile(affine_static_qdq_fn, backend=_capturing_backend(graphs))
    actual_dequant, actual_data_grad, actual_scale_grad, actual_offset_grad = _run_quant_dequant(
        data, scale, offset, tile_size, num_bits, use_qdq_fn=True, qdq_fn=compiled_qdq
    )

    # THEN the compiled graphs use plain torch ops instead of the custom op
    assert graphs
    assert all("torch.ops.fastforward" not in code for code in graphs)

    # AND the compiled results match the eager custom op
    assert torch.equal(actual_dequant, ref_dequant)
    assert torch.equal(actual_data_grad, ref_data_grad)
    torch.testing.assert_close(actual_scale_grad, ref_scale_grad, rtol=1e-4, atol=1e-4)
    if with_offset:
        assert ref_offset_grad is not None
        assert actual_offset_grad is not None
        torch.testing.assert_close(actual_offset_grad, ref_offset_grad, rtol=1e-4, atol=1e-4)


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:.*should not be instantiated:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:.*torch.jit.script_method. is deprecated:DeprecationWarning")
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "quantized_dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.float64],
    ids=["float16", "bfloat16", "float32", "float64"],
)
def test_qdq_compiled_matches_eager_with_output_dtype(
    device: str, quantized_dtype: torch.dtype
) -> None:
    # GIVEN data and quantization parameters, and an output dtype that differs from the input
    torch._dynamo.reset()
    data, scale, offset = _quant_params((4, 6), (4, 6), device, with_offset=True)
    ref_dequant, ref_data_grad, ref_scale_grad, ref_offset_grad = _run_quant_dequant(
        data, scale, offset, (4, 6), 8, use_qdq_fn=True, quantized_dtype=quantized_dtype
    )

    # WHEN the same computation is compiled, forward and backward
    graphs: list[str] = []
    compiled_qdq = torch.compile(affine_static_qdq_fn, backend=_capturing_backend(graphs))
    actual_dequant, actual_data_grad, actual_scale_grad, actual_offset_grad = _run_quant_dequant(
        data,
        scale,
        offset,
        (4, 6),
        8,
        use_qdq_fn=True,
        qdq_fn=compiled_qdq,
        quantized_dtype=quantized_dtype,
    )

    # THEN the output has the requested dtype, and the compiled results match the eager ones
    assert graphs
    assert ref_dequant.dtype == quantized_dtype
    assert actual_dequant.dtype == quantized_dtype
    torch.testing.assert_close(actual_dequant, ref_dequant)
    torch.testing.assert_close(actual_data_grad, ref_data_grad)
    torch.testing.assert_close(actual_scale_grad, ref_scale_grad, rtol=1e-4, atol=1e-4)
    if ref_offset_grad is not None:
        assert actual_offset_grad is not None
        torch.testing.assert_close(actual_offset_grad, ref_offset_grad, rtol=1e-4, atol=1e-4)
