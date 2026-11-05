# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tensor fusion logic backing the :class:`~fastforward.export.stages.gguf.rules.Fuse` rule.

Given source patterns and a list of extracted tensors, groups matching tensors
and concatenates them along the specified axis.
"""

from __future__ import annotations

import re

import torch

from fastforward.exceptions import ExportError
from fastforward.export.stages.gguf._extract import ExtractedTensor


def _resolve_target_name(target_template: str, groups: tuple[str, ...]) -> str:
    """Substitute back-references in a target name template."""
    result = target_template
    for i, group_value in enumerate(groups, start=1):
        result = result.replace(f"\\{i}", group_value)
    return result


def _match_sources(
    sources: tuple[str, ...],
    tensor_names: list[str],
) -> dict[tuple[str, ...], list[str | None]]:
    """Match tensor names against a fusion's source patterns.

    Single pass over tensor names: each name is tested against every source
    pattern once. Results are grouped by captured groups (e.g. layer index).

    Returns:
        Mapping from captured groups to a list of matched names per source
        pattern (None for unmatched positions). Only groups with at least one
        match are included.
    """
    compiled = [re.compile(p) for p in sources]
    n_sources = len(sources)
    groups_map: dict[tuple[str, ...], list[str | None]] = {}

    for name in tensor_names:
        for src_idx, pattern in enumerate(compiled):
            m = pattern.fullmatch(name)
            if m:
                groups = m.groups()
                if groups not in groups_map:
                    groups_map[groups] = [None] * n_sources
                groups_map[groups][src_idx] = name

    return groups_map


def _logical_shape(tensor: ExtractedTensor) -> tuple[int | None, ...]:
    """Return the tensor's shape; float tensors use their data, as ``cols`` may be unset."""
    if tensor.float_data is not None:
        return tuple(tensor.float_data.shape)
    return (tensor.rows, tensor.cols)


def _validate_fusion_group(
    source_tensors: list[ExtractedTensor],
    axis: int,
    groups: tuple[str, ...],
) -> None:
    """Validate that a matched set of tensors is compatible for fusion.

    Raises:
        ExportError: On mixed kinds or incompatible shapes on the non-concat axis.
    """
    kinds = {t.kind for t in source_tensors}
    if len(kinds) > 1:
        msg = (
            f"Tensor fusion for instance {groups}: cannot fuse tensors "
            f"of mixed kinds {kinds}. All sources must be quantized or "
            f"all must be float."
        )
        raise ExportError(msg)

    shapes = [_logical_shape(t) for t in source_tensors]
    if axis == 0:
        cols_set = {shape[1:] for shape in shapes}
        if len(cols_set) > 1:
            msg = (
                f"Tensor fusion for instance {groups}: cannot row-concat "
                f"tensors with different column counts {cols_set}"
            )
            raise ExportError(msg)
    elif axis == 1:
        if any(len(shape) < 2 for shape in shapes):
            msg = f"Tensor fusion for instance {groups}: cannot col-concat 1-D tensors"
            raise ExportError(msg)
        rows_set = {shape[0] for shape in shapes}
        if len(rows_set) > 1:
            msg = (
                f"Tensor fusion for instance {groups}: cannot col-concat "
                f"tensors with different row counts {rows_set}"
            )
            raise ExportError(msg)


def _fuse_quantized(tensors: list[ExtractedTensor], target_name: str, axis: int) -> ExtractedTensor:
    """Concatenate quantized tensors along the given axis."""
    formats = {t.quant_format for t in tensors} - {None}
    if len(formats) > 1:
        msg = (
            f"Tensor fusion '{target_name}': cannot fuse tensors with "
            f"different quant formats {formats}"
        )
        raise ExportError(msg)
    fused_format = next(iter(formats)) if formats else None

    codes_list = []
    scales_list = []
    offsets_list = []
    has_offsets = tensors[0].offsets is not None
    for t in tensors:
        assert t.int_codes is not None and t.scales is not None
        codes_list.append(t.int_codes)
        scales_list.append(t.scales)
        if has_offsets:
            assert t.offsets is not None
            offsets_list.append(t.offsets)

    fused_codes = torch.cat(codes_list, dim=axis)
    fused_scales = torch.cat(scales_list, dim=axis)
    fused_offsets = torch.cat(offsets_list, dim=axis) if has_offsets else None

    if axis == 0:
        total_rows = sum(t.rows for t in tensors)
        total_cols = tensors[0].cols
    else:
        total_rows = tensors[0].rows
        total_cols = sum(t.cols for t in tensors if t.cols is not None)

    return ExtractedTensor(
        hf_name=target_name,
        kind="quantized",
        rows=total_rows,
        cols=total_cols,
        int_codes=fused_codes,
        scales=fused_scales,
        offsets=fused_offsets,
        quant_format=fused_format,
    )


def _fuse_float(tensors: list[ExtractedTensor], target_name: str, axis: int) -> ExtractedTensor:
    """Concatenate float tensors along the given axis."""
    data_list = []
    for t in tensors:
        assert t.float_data is not None
        data_list.append(t.float_data)

    fused_data = torch.cat(data_list, dim=axis)
    return ExtractedTensor(
        hf_name=target_name,
        kind="float",
        rows=fused_data.shape[0],
        # Float tensors can be 1D (norms, biases), so cols may not exist.
        cols=fused_data.shape[1] if fused_data.ndim > 1 else None,
        float_data=fused_data,
    )


def fuse_matching(
    tensors: list[ExtractedTensor],
    sources: tuple[str, ...],
    target_name: str,
    axis: int,
) -> list[ExtractedTensor]:
    """Fuse every group of tensors matching ``sources`` into one tensor per group.

    Matches the source patterns against tensor ``hf_name`` values, groups by
    the captured layer/instance index, validates compatibility, and
    concatenates each group along ``axis``. Unmatched tensors pass through
    unchanged.

    Args:
        tensors: Extracted tensors.
        sources: Regex patterns for the tensors to fuse, in concat order.
        target_name: Name template for the fused tensors (may contain
            back-references to the captured groups).
        axis: Concatenation axis.

    Returns:
        A new list: unmatched tensors in their original order, followed by the
        fused tensors.

    Raises:
        ExportError: If sources within a group have mismatched kinds, if
            shapes are incompatible on the non-concat dimension, or if some but
            not all sources match for a given instance.
    """
    tensor_by_name: dict[str, ExtractedTensor] = {t.hf_name: t for t in tensors}
    matches = _match_sources(sources, list(tensor_by_name))
    consumed_names: set[str] = set()
    fused_tensors: list[ExtractedTensor] = []

    for groups, matched_names in matches.items():
        missing = [i for i, name in enumerate(matched_names) if name is None]
        if missing:
            present = [n for n in matched_names if n is not None]
            msg = (
                f"Tensor fusion incomplete for instance {groups}: "
                + f"source pattern(s) at index {missing} had no match. "
                + f"Matched sources: {present}"
            )
            raise ExportError(msg)

        source_tensors = [tensor_by_name[name] for name in matched_names]  # type: ignore[index]
        _validate_fusion_group(source_tensors, axis, groups)

        resolved_name = _resolve_target_name(target_name, groups)
        if source_tensors[0].kind == "quantized":
            fused = _fuse_quantized(source_tensors, resolved_name, axis)
        else:
            fused = _fuse_float(source_tensors, resolved_name, axis)

        fused_tensors.append(fused)
        consumed_names.update(name for name in matched_names if name is not None)

    result = [t for t in tensors if t.hf_name not in consumed_names]
    result.extend(fused_tensors)
    return result
