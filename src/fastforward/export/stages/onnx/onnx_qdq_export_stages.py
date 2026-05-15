# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import functools

from typing import Any, Callable, TypeAlias

import onnx
import onnxscript
import torch

from fastforward.exceptions import ExportError
from fastforward.export.stages.onnx.onnx_export_stages import stage_fx_to_onnx_program

_SampleInputsT: TypeAlias = list[tuple[tuple[Any, ...], dict[str, Any]]]

_QDQ_DTYPE_BY_BITWIDTH = {
    # Smallest signed-int ONNX storage that matches FF's signed-range clamping.
    # FF's quantize_by_tile clamps to [-2**(N-1), 2**(N-1)-1]; ONNX QuantizeLinear
    # saturates to the storage dtype's full range. The two only agree when the
    # storage dtype is at least N bits wide, so we pick the tight signed dtype
    # for each supported bitwidth. INT4/INT16 in QuantizeLinear/DequantizeLinear
    # require opset 21+ (enforced in `_with_ff_qdq_lowerings`).
    4: onnx.TensorProto.INT4,
    8: onnx.TensorProto.INT8,
    16: onnx.TensorProto.INT16,
}
_SUPPORTED_QDQ_BITWIDTHS = frozenset(_QDQ_DTYPE_BY_BITWIDTH)
# Minimum ONNX opset that supports INT4/INT16 in QuantizeLinear/DequantizeLinear.
# Lower opsets emit a structurally-malformed graph that ORT rejects at load time
# (no clear export-time error), so we both default to this and validate user input.
_QDQ_MIN_OPSET_VERSION = 21


def _ff_zero_point_from_offset(scale: Any, offset: Any | None, op: Any) -> Any:
    # `offset` is a SymbolicTensor at lowering time, not an eager torch tensor,
    # so we express `-round(offset)` as ONNX ops; the optimizer folds the chain
    # to a constant initializer in the saved graph.
    if offset is None:
        # Same shape as scale; values zero.
        return op.Sub(scale, scale)
    return op.Neg(op.Round(offset))


def _ff_axis_from_tile_size(
    tile_size: list[int] | tuple[int, ...] | torch.Size,
    data: Any | None = None,
) -> int | None:
    """Map FF's ``tile_size`` to ONNX QDQ ``axis``.

    For per-channel quantization FF emits a ``tile_size`` with ``1`` at the
    channel axis; for per-tensor it uses the data shape. The channel axis is
    therefore the (unique) dim where ``tile_size[i] == 1``.

    For 1x1-conv weights ``[O, I, 1, 1]`` quantized per-output-channel, FF
    emits ``tile_size = [1, I, 1, 1]`` — three candidate ``1``-axes, but axes
    2 and 3 are coincidental (the data dim is also 1 there). The disambiguator
    prefers the candidate whose data dim is genuinely non-1.
    """
    axes = [idx for idx, dim in enumerate(tile_size) if int(dim) == 1]
    if len(axes) == 0:
        return None
    if len(axes) == 1:
        return axes[0]

    # Disambiguator for the multi-`1` case (e.g. 1x1 conv weights). At lowering
    # time `data` is an onnxscript SymbolicTensor; we read `.shape` directly
    # rather than via getattr so that an unexpected callsite fails loudly
    # instead of silently falling through to `axes[0]`.
    shape = data.shape if data is not None else None
    if shape is not None:
        for axis in axes:
            try:
                dim = int(shape[axis])
            except (IndexError, TypeError, ValueError):
                # Dynamic / symbolic / out-of-bounds dim: can't use this axis
                # as a tiebreaker. Skip and try the next candidate.
                continue
            if dim != 1:
                return axis

    # Reachable only if every candidate axis has data dim 1 or unresolvable.
    # For supported FF tilings (per-tensor, single-axis per-channel) this is
    # the degenerate all-ones case.
    return axes[0]


def _ff_quant_dtype_for_bitwidth(num_bits: int | float) -> int:
    """Return the ONNX storage dtype for FF's signed-range clamping at ``num_bits``.

    ``num_bits`` is typed ``int | float`` to match FF's op signature, but QDQ
    storage requires an integer-valued bitwidth — fractional bitwidths cannot be
    represented by an ONNX integer dtype. Fractional or unsupported values raise
    ``ExportError`` so callers (and ``ExportError``-handling consumers) get a
    clean failure mode rather than a silent truncation or a bare ``ValueError``.
    """
    bitwidth_float = float(num_bits)
    if not bitwidth_float.is_integer():
        msg = f"QDQ ONNX lowering requires integer-valued num_bits, got {num_bits!r}"
        raise ExportError(msg)
    bitwidth = int(bitwidth_float)
    dtype = _QDQ_DTYPE_BY_BITWIDTH.get(bitwidth)
    if dtype is None:
        msg = (
            "QDQ ONNX lowering supports only bitwidths "
            f"{sorted(_SUPPORTED_QDQ_BITWIDTHS)}, got {num_bits!r}"
        )
        raise ExportError(msg)
    return dtype


def _make_ff_quantize_by_tile_onnx(op: Any) -> Callable[..., Any]:
    """Return a `quantize_by_tile -> QuantizeLinear` lowering bound to ``op``.

    ``op`` is the onnxscript opset namespace (e.g. ``onnxscript.opset21``)
    selected at runtime from the user's ``onnx_export_options["opset_version"]``.
    The integer storage dtype is derived from ``num_bits`` (signed, matching
    FF's signed-range clamping). The Q output is the int storage dtype and is
    expected to flow as-is into a downstream ``DequantizeLinear``.
    """

    def lowering(
        input: Any,
        scale: Any,
        tile_size: list[int] | tuple[int, ...] | torch.Size,
        num_bits: int | float,
        output_dtype: Any | None = None,
        offset: Any | None = None,
    ) -> Any:
        del output_dtype
        bitwidth = int(num_bits)
        axis = _ff_axis_from_tile_size(tile_size, data=input)
        zero_point = _ff_zero_point_from_offset(scale, offset, op)
        zero_point = op.Cast(zero_point, to=_ff_quant_dtype_for_bitwidth(bitwidth))
        if axis is None:
            return op.QuantizeLinear(input, scale, zero_point)
        return op.QuantizeLinear(input, scale, zero_point, axis=axis)

    return lowering


def _make_ff_dequantize_by_tile_onnx(op: Any) -> Callable[..., Any]:
    """Return a `dequantize_by_tile -> DequantizeLinear` lowering bound to ``op``.

    Expects the lowering's ``input`` to already be the integer-typed output of an
    upstream ``QuantizeLinear`` (i.e. no float-cast roundtrip between Q and DQ).
    The zero_point is cast to match ``input``'s dtype, so any of INT4/INT8/INT16
    storage flows through correctly.
    """

    def lowering(
        input: Any,
        scale: Any,
        tile_size: list[int] | tuple[int, ...] | torch.Size,
        offset: Any | None = None,
        output_dtype: Any | None = None,
    ) -> Any:
        del output_dtype
        axis = _ff_axis_from_tile_size(tile_size, data=input)
        zero_point = _ff_zero_point_from_offset(scale, offset, op)
        zero_point = op.CastLike(zero_point, input)
        if axis is None:
            return op.DequantizeLinear(input, scale, zero_point)
        return op.DequantizeLinear(input, scale, zero_point, axis=axis)

    return lowering


@functools.lru_cache(maxsize=None)
def _resolve_qdq_lowerings(
    opset_version: int,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Return cached ``(Q, DQ)`` lowerings bound to onnxscript's opset namespace.

    The opset namespace is fetched at call time from
    ``onnxscript.opset{opset_version}`` so that a user-supplied opset version is
    threaded through to the actual ops emitted by the lowering. Caller is
    responsible for validating ``opset_version >= _QDQ_MIN_OPSET_VERSION`` —
    that gate lives in :func:`_with_ff_qdq_lowerings`.
    """
    attr = f"opset{opset_version}"
    op = getattr(onnxscript, attr, None)
    if op is None:
        msg = (
            f"onnxscript does not provide {attr}; upgrade onnxscript or pick "
            f"a supported opset (>= {_QDQ_MIN_OPSET_VERSION})."
        )
        raise ExportError(msg)
    return _make_ff_quantize_by_tile_onnx(op), _make_ff_dequantize_by_tile_onnx(op)


def _with_ff_qdq_lowerings(context: dict[str, Any]) -> dict[str, Any]:
    """Return a context with FF Q/DQ lowerings merged into ``onnx_export_options``.

    Builds a fresh context dict and a fresh ``onnx_export_options`` /
    ``custom_translation_table`` at every level it touches, so neither the input
    context nor any user-owned nested dict is mutated. User-supplied lowerings
    for the same op take precedence over the FF defaults.

    Defaults ``onnx_export_options["opset_version"]`` to
    ``_QDQ_MIN_OPSET_VERSION`` (21) when unset, and rejects user-supplied values
    below that — INT4/INT16 in ``QuantizeLinear`` / ``DequantizeLinear`` were
    introduced in opset 21, and lower opsets silently produce a graph that ORT
    refuses to load. The validated opset version is then used to resolve the
    onnxscript opset namespace that the FF lowerings emit through, so passing
    ``opset_version=22`` (or higher) routes the lowerings through the matching
    ``onnxscript.opsetN`` instead of the minimum.
    """
    raw_options = context.get("onnx_export_options") or {}
    if not isinstance(raw_options, dict):
        msg = "`onnx_export_options` must be a dictionary if provided in context."
        raise TypeError(msg)
    raw_table = raw_options.get("custom_translation_table") or {}
    if not isinstance(raw_table, dict):
        msg = "`onnx_export_options['custom_translation_table']` must be a dictionary."
        raise TypeError(msg)

    options = {**raw_options}
    options.setdefault("opset_version", _QDQ_MIN_OPSET_VERSION)
    if int(options["opset_version"]) < _QDQ_MIN_OPSET_VERSION:
        msg = (
            f"qnn/onnx_qdq pipeline requires opset_version >= {_QDQ_MIN_OPSET_VERSION} "
            f"(got {options['opset_version']}). INT4/INT16 in "
            "QuantizeLinear/DequantizeLinear were introduced in opset 21; "
            "lower opsets produce a graph that ONNX Runtime cannot load."
        )
        raise ExportError(msg)

    q_lowering, dq_lowering = _resolve_qdq_lowerings(int(options["opset_version"]))

    table = dict(raw_table)
    table.setdefault(torch.ops.fastforward.quantize_by_tile.default, q_lowering)
    table.setdefault(torch.ops.fastforward.dequantize_by_tile.default, dq_lowering)
    options["custom_translation_table"] = table
    return {**context, "onnx_export_options": options}


def stage_fx_to_onnx_program_qdq(
    modules: tuple[torch.fx.GraphModule, ...],
    sample_inputs: _SampleInputsT,
    context: dict[str, Any],
) -> torch.onnx.ONNXProgram:
    """Variant of ``stage_fx_to_onnx_program`` that injects FF Q/DQ ONNX lowerings.

    Builds a fresh context with the FF custom translation table merged into
    ``onnx_export_options`` (see :func:`_with_ff_qdq_lowerings`) and delegates
    to ``stage_fx_to_onnx_program``. The input context and any user-owned
    nested dicts are left unmodified.
    """
    return stage_fx_to_onnx_program(modules, sample_inputs, _with_ff_qdq_lowerings(context))
