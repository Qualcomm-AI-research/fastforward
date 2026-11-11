# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tensor splitting logic backing the :class:`~fastforward.export.stages.gguf.rules.Split` rule.

Each tensor whose name matches the rule's pattern is cut along one axis into
consecutive parts, which replace it in the tensor list. Float tensors can be
split along any axis. Quantized tensors can only be split along axis 0: whole
rows move together with their per-block scales and offsets, so every block
stays intact and each part packs to the matching rows of the source's packed
bytes.
"""

from __future__ import annotations

import re

from collections.abc import Sequence

import torch

from fastforward.exceptions import ExportError
from fastforward.export.stages.gguf._extract import ExtractedTensor
from fastforward.export.stages.gguf._fusion import _resolve_target_name


def _contiguous_copy(tensor: torch.Tensor) -> torch.Tensor:
    """Copy a split part into freshly allocated row-major memory.

    ``Tensor.contiguous()`` is not enough: a part whose dimensions all have
    size 1 counts as contiguous whatever its strides, so it would keep a
    non-unit stride from a strided source and break the packers' dtype views.
    """
    return tensor.clone(memory_format=torch.contiguous_format)


def _resolve_part_sizes(
    hf_name: str, length: int, n_parts: int, sizes: Sequence[int] | None
) -> list[int]:
    """Return the size of every part, checking that they tile ``length`` exactly."""
    if sizes is None:
        if length % n_parts != 0:
            msg = (
                f"Cannot split '{hf_name}' into {n_parts} equal parts: "
                + f"its length {length} along the split axis is not divisible by {n_parts}"
            )
            raise ExportError(msg)
        return [length // n_parts] * n_parts

    part_sizes = list(sizes)
    if len(part_sizes) != n_parts:
        msg = f"Cannot split '{hf_name}': got {len(part_sizes)} sizes for {n_parts} targets"
        raise ExportError(msg)
    if any(size <= 0 for size in part_sizes) or sum(part_sizes) != length:
        msg = (
            f"Cannot split '{hf_name}': sizes {part_sizes} must be positive and add up to "
            + f"its length {length} along the split axis"
        )
        raise ExportError(msg)
    return part_sizes


def _split_float(
    tensor: ExtractedTensor,
    names: list[str],
    axis: int,
    sizes: Sequence[int] | None,
    squeeze: bool,
) -> list[ExtractedTensor]:
    """Split a float tensor along any axis, optionally dropping that axis."""
    assert tensor.float_data is not None
    data = tensor.float_data
    if not -data.ndim <= axis < data.ndim:
        msg = f"Cannot split '{tensor.hf_name}' along axis {axis}: it has {data.ndim} dimension(s)"
        raise ExportError(msg)

    part_sizes = _resolve_part_sizes(tensor.hf_name, data.shape[axis], len(names), sizes)
    if squeeze and (data.ndim < 2 or any(size != 1 for size in part_sizes)):
        msg = (
            f"Cannot split '{tensor.hf_name}' with squeeze: every part must have size 1 "
            + f"along axis {axis} and keep at least one dimension, got sizes {part_sizes} "
            + f"of a {data.ndim}-D tensor"
        )
        raise ExportError(msg)

    parts: list[ExtractedTensor] = []
    for name, part in zip(names, torch.split(data, part_sizes, dim=axis)):
        if squeeze:
            part = part.squeeze(axis)
        part = _contiguous_copy(part)
        parts.append(
            ExtractedTensor(
                hf_name=name,
                kind="float",
                rows=part.shape[0],
                cols=part.shape[1] if part.ndim > 1 else None,
                float_data=part,
            )
        )
    return parts


def _split_quantized(
    tensor: ExtractedTensor,
    names: list[str],
    axis: int,
    sizes: Sequence[int] | None,
    squeeze: bool,
) -> list[ExtractedTensor]:
    """Split a quantized tensor into row ranges, keeping codes, scales and offsets paired."""
    assert tensor.int_codes is not None and tensor.scales is not None
    if axis != 0 or squeeze:
        msg = (
            f"Cannot split quantized tensor '{tensor.hf_name}' along axis {axis}"
            + (" with squeeze" if squeeze else "")
            + ": quantized tensors can only be split along axis 0 (whole rows), "
            + "which keeps every quantization block intact"
        )
        raise ExportError(msg)

    part_sizes = _resolve_part_sizes(tensor.hf_name, tensor.rows, len(names), sizes)
    # View every per-row quantity as (rows, -1) so a row range selects whole blocks.
    codes = torch.split(tensor.int_codes.reshape(tensor.rows, -1), part_sizes)
    scales = torch.split(tensor.scales.reshape(tensor.rows, -1), part_sizes)
    offsets: Sequence[torch.Tensor | None] = (
        [None] * len(names)
        if tensor.offsets is None
        else torch.split(tensor.offsets.reshape(tensor.rows, -1), part_sizes)
    )

    return [
        ExtractedTensor(
            hf_name=name,
            kind="quantized",
            rows=rows,
            cols=tensor.cols,
            int_codes=_contiguous_copy(part_codes),
            scales=_contiguous_copy(part_scales),
            offsets=_contiguous_copy(part_offsets) if part_offsets is not None else None,
            quant_format=tensor.quant_format,
        )
        for name, rows, part_codes, part_scales, part_offsets in zip(
            names, part_sizes, codes, scales, offsets
        )
    ]


def split_matching(
    tensors: list[ExtractedTensor],
    pattern: str,
    targets: tuple[str, ...],
    axis: int,
    sizes: Sequence[int] | None,
    squeeze: bool,
) -> list[ExtractedTensor]:
    r"""Split every tensor whose ``hf_name`` fully matches ``pattern``.

    Args:
        tensors: Extracted tensors.
        pattern: Regex matched (``re.fullmatch``) against ``hf_name``.
        targets: Name templates for the parts, in order. They may contain
            back-references (``\1``, ``\2``, ...) to the pattern's groups.
        axis: The axis to split along. Quantized tensors only allow 0.
        sizes: Size of every part along ``axis``, or ``None`` for equal parts.
        squeeze: Whether to drop ``axis`` from every part (each part must
            have size 1 along it). Float tensors only.

    Returns:
        A new list in which every matching tensor is replaced, in place, by
        its parts. The parts take their resolved target names as ``hf_name``
        and have no GGUF name yet.

    Raises:
        ExportError: If a matching tensor cannot be split as requested.
    """
    compiled = re.compile(pattern)
    result: list[ExtractedTensor] = []
    for tensor in tensors:
        match = compiled.fullmatch(tensor.hf_name)
        if match is None:
            result.append(tensor)
            continue
        names = [_resolve_target_name(target, match.groups()) for target in targets]
        split = _split_quantized if tensor.kind == "quantized" else _split_float
        result.extend(split(tensor, names, axis, sizes, squeeze))
    return result
