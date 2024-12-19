# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

"""
This file holds temporary implementations for view-like operations that test for
linear quantizers. It should be refactord to use a more robust dispatching
system once that lands.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import torch

import fastforward as ff

from fastforward.dispatcher import Predicate, register
from fastforward.quantized_tensor import QuantizedTensor, apply_and_reattach

Size: TypeAlias = torch.Size | tuple[int, ...]

if TYPE_CHECKING:
    from fastforward.quantization.affine import StaticAffineQuantParams


def _is_affine_per_tensor(input: QuantizedTensor, *args: Any, **kwargs: Any) -> bool:
    # As part of the predicate we verify input is acutally a QuantizedTensor
    if not isinstance(input, QuantizedTensor):
        return False  # type: ignore[unreachable]

    # Lazy import to break circular import
    from fastforward.quantization.affine import AffineQuantizationFunction, StaticAffineQuantParams

    context = input.quantization_context
    granularity = getattr(context.quantization_params, "granularity", None)

    is_affine = issubclass(context.quantization_fn, AffineQuantizationFunction)
    has_affine_params = isinstance(context.quantization_params, StaticAffineQuantParams)
    is_per_tensor = isinstance(granularity, ff.PerTensor)
    return is_affine and has_affine_params and is_per_tensor


affine_per_tensor_predicate = Predicate(_is_affine_per_tensor)


def _affine_params(tensor: QuantizedTensor) -> "StaticAffineQuantParams":
    """Perform implicit cast ont quant_args"""
    return tensor.quant_args()  # type: ignore[return-value]


def _is_affine(input: QuantizedTensor, *args: Any, **kwargs: Any) -> bool:
    # As part of the predicate we verify input is acutally a QuantizedTensor
    if not isinstance(input, QuantizedTensor):
        return False  # type: ignore[unreachable]

    # Lazy import to break circular import
    from fastforward.quantization.affine import AffineQuantizationFunction, StaticAffineQuantParams

    context = input.quantization_context
    is_affine = issubclass(context.quantization_fn, AffineQuantizationFunction)
    has_affine_params = isinstance(context.quantization_params, StaticAffineQuantParams)
    return is_affine and has_affine_params


affine_predicate = Predicate(_is_affine)


# See https://morpheus-gitlab.qualcomm.com/compression/fastforward/-/issues/67
# for why the type: ignore[arg-type] is required
@register("contiguous", None)  # type: ignore[arg-type]
def contiguous(input: QuantizedTensor) -> QuantizedTensor:
    return apply_and_reattach(lambda x: x.contiguous(), input)


@register("view", affine_per_tensor_predicate)
def view(input: QuantizedTensor, *args: Any) -> QuantizedTensor:
    if len(args) > 0 and isinstance(args[0], torch.dtype):
        raise TypeError("QuantizedTensor.view(dtype) is not supported")
    return apply_and_reattach(lambda x: x.view(*args), input)


@register("view_as", affine_per_tensor_predicate)
def view_as(input: QuantizedTensor, *args: Any) -> QuantizedTensor:
    if len(args) > 0 and isinstance(args[0], torch.dtype):
        raise TypeError("QuantizedTensor.view_as(dtype) is not supported")
    return apply_and_reattach(lambda x: x.view_as(*args), input)


@register("reshape", affine_per_tensor_predicate)
def reshape(input: QuantizedTensor, *args: Any) -> QuantizedTensor:
    return apply_and_reattach(lambda x: x.reshape(*args), input)


@register("transpose", affine_per_tensor_predicate)
def transpose(input: QuantizedTensor, *args: Any) -> QuantizedTensor:
    return apply_and_reattach(lambda x: x.transpose(*args), input)


# See https://morpheus-gitlab.qualcomm.com/compression/fastforward/-/issues/67
# for why the type: ignore[arg-type] is required
@register("ones_like", None)  # type: ignore[arg-type]
def ones_like(input: QuantizedTensor, **kwargs: Any) -> torch.Tensor:
    return torch.ones_like(input.raw_data, **{"dtype": torch.float, **kwargs})


class _ScaleGradient(torch.autograd.Function):
    """
    Scale the gradient by `scalar` during backward pass.
    """

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, scalar: float) -> torch.Tensor:
        ctx.scalar = scalar
        return input

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        return grad * ctx.scalar, None


def _is_scalar_other(input: QuantizedTensor, other: float, **kwargs) -> bool:  # type: ignore[no-untyped-def]
    return isinstance(other, (float, int))


def _no_output_quantizer(
    input: QuantizedTensor, other: float, output_quantizer: Any = None, **kwargs: Any
) -> bool:
    # Lazy import to break circular import
    from fastforward.nn import QuantizerStub

    return output_quantizer is None or isinstance(output_quantizer, QuantizerStub)


scalar_other_predicate = Predicate(_is_scalar_other)
no_output_quantizer_predicate = Predicate(_no_output_quantizer)


@register(
    "mul",
    affine_per_tensor_predicate & scalar_other_predicate & no_output_quantizer_predicate,
    # type: ignore[misc, call-overload]
)
def scalar_multiply(
    input: QuantizedTensor, other: float, *args: Any, **kwargs: Any
) -> QuantizedTensor:
    """
    Multiply quantized `input` tensor by a scalar.

    Mutliplication by a scalar is a special case for affine quanitzed tensors
    since it can be implemented by only multiplying the scaling factor.
    """
    quant_params = _affine_params(input)
    new_scale = quant_params.scale * other
    scaled_out = cast(torch.Tensor, _ScaleGradient.apply(input.raw_data, other))

    new_context = input.quantization_context.with_changes(scale=new_scale)
    return new_context.attach(scaled_out)


def cat_predicate(
    tensors: Sequence[QuantizedTensor],
    dim: int = 0,
    output_quantizer: Any = None,
    strict_quantization: bool = False,
) -> bool:
    """
    Predicate for quantized concatenation

    Returns True if all elements of `tensors` are quantized using affine quantization
    using the exact same quantization params.
    """
    if output_quantizer is not None:
        return False

    # Lazy import to break circular import
    from fastforward.quantization.affine import AffineQuantizationFunction, StaticAffineQuantParams

    scale: torch.Tensor | float | None = None
    offset: torch.Tensor | float | None = None

    for tensor in tensors:
        if not isinstance(tensor, QuantizedTensor):
            return False  # type: ignore[unreachable]

        if not issubclass(tensor.quant_func, AffineQuantizationFunction):
            return False

        quant_params = _affine_params(tensor)
        if not isinstance(quant_params, StaticAffineQuantParams):
            return False  # type: ignore[unreachable]

        if not isinstance(quant_params.granularity, ff.PerTensor):
            return False

        scale_match = scale is None or quant_params.scale == scale
        offset_match = offset is None or quant_params.offset == offset
        if not (scale_match and offset_match):
            return False
        scale = quant_params.scale
        offset = quant_params.offset

    return True


@register("cat", Predicate(cat_predicate))
def cat(
    tensors: Sequence[QuantizedTensor], dim: int = 0, *args: Any, **kwargs: Any
) -> QuantizedTensor:
    output = torch.cat([t.raw_data for t in tensors], dim=dim)
    return tensors[0].quantization_context.attach(output)


@register("__getitem__", affine_per_tensor_predicate)
def getitem(input: QuantizedTensor, *args: Any) -> QuantizedTensor:
    return apply_and_reattach(lambda x: x.__getitem__(*args), input)


@register("expand", affine_per_tensor_predicate)
def expand(input: QuantizedTensor, *args: Any) -> QuantizedTensor:
    return apply_and_reattach(lambda x: x.expand(*args), input)


@register("unsqueeze", affine_predicate)
def unsqueeze(input: QuantizedTensor, dim: int) -> QuantizedTensor:
    quant_params = _affine_params(input)
    data = input.raw_data.unsqueeze(dim)

    match quant_params.granularity:
        case ff.PerTensor():
            new_context = input.quantization_context
        case ff.PerChannel(axis):
            new_axis = tuple(ax + ax >= dim for ax in axis)
            new_context = input.quantization_context.with_changes(
                granularity=ff.PerChannel(new_axis)
            )
        case ff.PerTile(tile_size):
            new_size = [s for s in tile_size]
            new_size.insert(dim % (len(new_size) + 1), 1)
            new_context = input.quantization_context.with_changes(
                granularity=ff.PerTile(tuple(new_size))
            )
        case _:
            raise ValueError(
                f"unsqueeze: unsupported granularity: {type(quant_params.granularity).__name__}"
            )

    return new_context.attach(data)


@register("take_along_dim", affine_per_tensor_predicate)
def take_along_dim(
    input: QuantizedTensor, indices: torch.LongTensor, dim: int | None = None
) -> QuantizedTensor:
    return apply_and_reattach(lambda x: torch.take_along_dim(x, indices, dim=dim), input)


@register("topk", affine_per_tensor_predicate)  # type: ignore[arg-type]
def topk(
    input: QuantizedTensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
) -> torch.return_types.topk:
    values, indices = torch.topk(input, k, dim=dim, largest=largest, sorted=sorted)
    values = input.quantization_context.attach(values)
    return torch.return_types.topk((values, indices))
