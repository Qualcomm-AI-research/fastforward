# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import functools

from typing import Any, Callable, List, ParamSpec, Sequence, TypeAlias, TypeVar

import torch

from fastforward.flags import get_compiled_quant_funcs
from fastforward.quantization import tiled_tensor
from fastforward.quantization.ste import round_ste

from . import affine
from .tiled_tensor import rows_to_tiles, tiles_to_rows

SizeT = Sequence[int]

_T = TypeVar("_T")
_P = ParamSpec("_P")


def conditional_compile(func: Callable[_P, _T]) -> Callable[_P, _T]:
    compiled_func = None

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        nonlocal compiled_func
        if "num_bits" in kwargs:
            kwargs["num_bits"] = float(kwargs["num_bits"])  # type: ignore[arg-type]

        if get_compiled_quant_funcs():
            if compiled_func is None:
                compiled_func = torch.compile(func)
            return compiled_func(*args, **kwargs)  # type: ignore[no-any-return]
        return func(*args, **kwargs)

    return wrapper


if torch.__version__ < "2.4":
    torch.library.define(
        "fastforward::quantize_by_tile",
        "("
        "Tensor input,"
        "Tensor scale,"
        "int[] tile_size,"
        "int num_bits,"
        "ScalarType? output_dtype = None,"
        "Tensor? offset = None"
        ") -> Tensor",
    )

    torch.library.define(
        "fastforward::dequantize_by_tile",
        "("
        "Tensor input,"
        "Tensor scale,"
        "int[] tile_size,"
        "Tensor? offset = None,"
        "ScalarType? output_dtype = None"
        ") -> Tensor",
    )

    torch.library.define(
        "fastforward::quantize_by_tile_backward",
        "("
        "Tensor input,"
        "Tensor doutput,"
        "Tensor scale,"
        "int[] tile_size,"
        "int num_bits,"
        "Tensor? offset"
        ") -> (Tensor, Tensor, Tensor)",
    )

    torch.library.define(
        "fastforward::quantize_dynamic_by_tile",
        "("
        "Tensor data,"
        "int[] tile_size,"
        "int num_bits,"
        "ScalarType output_dtype"
        ") -> (Tensor, Tensor, Tensor)",
    )


def quant_operator(name: str) -> Callable[[Callable[..., Any]], Any]:
    def decorator(func: Callable[..., Any]) -> Any:
        if torch.__version__ >= "2.4":
            return torch.library.custom_op(name, mutates_args=())(func)  # type: ignore[attr-defined]
        else:
            return torch.library.impl(name, ("cpu", "cuda"), func=func)

    return decorator


register_fake = getattr(torch.library, "register_fake", torch.library.impl_abstract)


def _infer_offset(offset: torch.Tensor | None, scale: torch.Tensor) -> torch.Tensor:
    return torch.round(offset.reshape(-1)) if offset is not None else torch.zeros_like(scale)


@quant_operator("fastforward::quantize_by_tile")
@conditional_compile
def quantize_by_tile_impl(
    data: torch.Tensor,
    scale: torch.Tensor,
    tile_size: SizeT,
    num_bits: float,
    output_dtype: torch.dtype | None,
    offset: torch.Tensor | None = None,
) -> torch.Tensor:
    scale = scale.reshape(-1)
    offset = _infer_offset(offset, scale)
    tile_size = torch.Size(tile_size)

    min_threshold = -(2 ** (num_bits - 1))
    max_threshold = -min_threshold - 1
    row_representation = tiled_tensor.tiles_to_rows(data, tile_size)
    quantized = round_ste(row_representation / scale[:, None] - offset[:, None])
    quantized = torch.clamp(quantized, min_threshold, max_threshold)
    result = tiled_tensor.rows_to_tiles(quantized, data.shape, tile_size)
    result = result.to(output_dtype or result.dtype)
    return result


@quant_operator("fastforward::dequantize_by_tile")
@conditional_compile
def dequantize_by_tile_impl(
    data: torch.Tensor,
    scale: torch.Tensor,
    tile_size: SizeT,
    offset: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    scale = scale.reshape(-1)
    offset = _infer_offset(offset, scale)
    tile_size = torch.Size(tile_size)

    row_representation = tiles_to_rows(data, tile_size)
    dequantized = (row_representation + offset[:, None]) * scale[:, None]
    dequantized = tiled_tensor.rows_to_tiles(dequantized, data.shape, tile_size)
    if output_dtype:
        dequantized = dequantized.to(output_dtype)
    return dequantized


@quant_operator("fastforward::quantize_by_tile_backward")
@conditional_compile
def quant_dequant_by_tile_grad_impl(
    data: torch.Tensor,
    output_grad: torch.Tensor,
    scale: torch.Tensor,
    tile_size: SizeT,
    num_bits: float,
    offset: torch.Tensor | None = None,
) -> List[torch.Tensor]:
    param_shape = scale.shape
    scale = scale.reshape(-1)
    offset_is_none = offset is None
    offset = _infer_offset(offset, scale)
    tile_size = torch.Size(tile_size)

    min_threshold = -(2 ** (num_bits - 1))
    max_threshold = -min_threshold - 1

    data_as_rows = tiles_to_rows(data, tile_size)
    grad_as_rows = tiles_to_rows(output_grad, tile_size)

    pre_round = (data_as_rows / scale[:, None]) - round_ste(offset[:, None])
    quantized = torch.round(pre_round)
    clip_mask = torch.logical_or(quantized < min_threshold, quantized > max_threshold)

    dinput = torch.where(clip_mask, 0, grad_as_rows)

    if offset_is_none:
        doffset = torch.Tensor()
    else:
        doffset = torch.where(clip_mask, scale[:, None] * grad_as_rows, 0)
        doffset = doffset.sum(1).reshape(param_shape)

    dscale = torch.empty(quantized.shape, dtype=scale.dtype, device=scale.device)
    min_thresh = torch.tensor([min_threshold], dtype=scale.dtype, device=scale.device)
    max_thresh = torch.tensor([max_threshold], dtype=scale.dtype, device=scale.device)
    torch.where(quantized < min_threshold, min_thresh, max_thresh, out=dscale)
    dscale.add_(offset[:, None].to(dscale.dtype))
    torch.where(clip_mask, dscale, (quantized - pre_round).to(dscale.dtype), out=dscale)
    dscale.mul_(grad_as_rows)

    dinput = rows_to_tiles(dinput, data.shape, tile_size)
    dscale = dscale.sum(1).reshape(param_shape)
    return [dinput, dscale, doffset]


RT: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@quant_operator("fastforward::quantize_dynamic_by_tile")
@conditional_compile
def quantize_dynamic_by_tile_impl(
    data: torch.Tensor, tile_size: SizeT, num_bits: float, output_dtype: torch.dtype | None
) -> RT:
    tile_size = torch.Size(tile_size)

    min_threshold = -(2 ** (num_bits - 1))
    max_threshold = -min_threshold - 1
    row_representation = tiled_tensor.tiles_to_rows(data, tile_size)

    min_range = torch.min(row_representation, dim=1).values
    max_range = torch.max(row_representation, dim=1).values
    scale, offset = affine.parameters_for_range(
        min_range, max_range, num_bits, symmetric=False, allow_one_sided=True
    )
    assert offset is not None
    offset = torch.round(offset)

    quantized = torch.round(row_representation / scale[:, None] - offset[:, None])
    quantized = torch.clamp(quantized, min_threshold, max_threshold)
    result = tiled_tensor.rows_to_tiles(quantized, data.shape, tile_size)
    result = result.to(output_dtype or result.dtype)
    return result, scale, offset


@register_fake("fastforward::quantize_by_tile")  # type: ignore[misc]
def quantize_by_tile_meta(
    input: torch.Tensor,
    scale: torch.Tensor,
    tile_size: SizeT,
    num_bits: float,
    output_dtype: torch.dtype | None,
    offset: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty(input.shape)


@register_fake("fastforward::dequantize_by_tile")  # type: ignore[misc]
def dequantize_by_tile_meta(
    input: torch.Tensor,
    scale: torch.Tensor,
    tile_size: SizeT,
    offset: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty(input.shape)


@register_fake("fastforward::quantize_by_tile_backward")  # type: ignore[misc]
def quantize_by_tile_backward_meta(
    input: torch.Tensor,
    output_grad: torch.Tensor,
    scale: torch.Tensor,
    tile_size: SizeT,
    num_bits: float,
    offset: torch.Tensor | None = None,
) -> List[torch.Tensor]:
    return [torch.empty(input.shape), torch.empty(scale.shape), torch.empty(scale.shape)]


@register_fake("fastforward::quantize_dynamic_by_tile")  # type: ignore[misc]
def quantize_dynamic_by_tile_meta(
    input: torch.Tensor,
    scale: torch.Tensor,
    tile_size: SizeT,
    num_bits: float,
    output_dtype: torch.dtype | None,
    offset: torch.Tensor | None = None,
) -> RT:
    num_params = int(input.numel() / torch.Size(tile_size).numel())
    scale = torch.empty(num_params)
    offset = torch.empty(num_params)
    return torch.empty(input.shape), scale, offset