# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

"""
This file holds temporary implementations for view-like operations that test for
linear quantizers. It should be refactord to use a more robust dispatching
system once that lands.
"""

from typing import Any, Optional, Sequence, TypeAlias

import torch

from fastforward.dispatcher import Predicate, register
from fastforward.nn.quantizer import Quantizer, QuantizerStub
from fastforward.quantized_tensor import QuantizedTensor, apply_and_reattach

Size: TypeAlias = torch.Size | tuple[int, ...]


def _numelof(data: torch.Tensor | float | None) -> int:
    if data is None:
        return 0
    if isinstance(data, (float, int)):
        return 1
    return data.numel()


def _is_linear_per_tensor(input: QuantizedTensor, *args, **kwargs) -> bool:  # type: ignore[no-untyped-def]
    # Lazy import to break circular import
    from fastforward.quantization.affine import TiledAffineQuantizationFunction
    from fastforward.quantization.dynamic import TiledDynamicAffineQuantizationFunction

    if not isinstance(input, QuantizedTensor):
        return False  # type: ignore[unreachable]

    quant_args = input.quant_args()
    return (
        issubclass(
            input.quant_func,
            (TiledAffineQuantizationFunction, TiledDynamicAffineQuantizationFunction),
        )
        and _numelof(quant_args.scale) == 1
        and _numelof(quant_args.offset) in (0, 1)
        and quant_args.tile_size == "data_shape"
    )


linear_per_tensor_predicate = Predicate(_is_linear_per_tensor)


def _is_linear_or_dynamic_linear(input: QuantizedTensor, *args, **kwargs) -> bool:  # type: ignore[no-untyped-def]
    # Lazy import to break circular import
    from fastforward.quantization.affine import TiledAffineQuantizationFunction
    from fastforward.quantization.dynamic import TiledDynamicAffineQuantizationFunction

    if not isinstance(input, QuantizedTensor):
        return False  # type: ignore[unreachable]
    if not (
        issubclass(input.quant_func, TiledAffineQuantizationFunction)
        or issubclass(input.quant_func, TiledDynamicAffineQuantizationFunction)
    ):
        return False
    return True


linear_or_dynamic_linear_predicate = Predicate(_is_linear_or_dynamic_linear)


# See https://morpheus-gitlab.qualcomm.com/compression/fastforward/-/issues/67
# for why the type: ignore[arg-type] is required
@register("contiguous", None)  # type: ignore[arg-type]
def contiguous(input: QuantizedTensor) -> QuantizedTensor:
    return apply_and_reattach(lambda x: x.contiguous(), input)


@register("view", linear_per_tensor_predicate)
def view(input: QuantizedTensor, *args: Any) -> QuantizedTensor:
    if len(args) > 0 and isinstance(args[0], torch.dtype):
        raise TypeError("QuantizedTensor.view(dtype) is not supported")
    return apply_and_reattach(lambda x: x.view(*args), input)


@register("view_as", linear_per_tensor_predicate)
def view_as(input: QuantizedTensor, *args: Any) -> QuantizedTensor:
    if len(args) > 0 and isinstance(args[0], torch.dtype):
        raise TypeError("QuantizedTensor.view_as(dtype) is not supported")
    return apply_and_reattach(lambda x: x.view_as(*args), input)


@register("reshape", linear_per_tensor_predicate)
def reshape(input: QuantizedTensor, *args: Any) -> QuantizedTensor:
    return apply_and_reattach(lambda x: x.reshape(*args), input)


@register("transpose", linear_per_tensor_predicate)
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
    input: QuantizedTensor, other: float, output_quantizer: None | Quantizer = None, **kwargs: Any
) -> bool:
    return output_quantizer is None or isinstance(output_quantizer, QuantizerStub)


scalar_other_predicate = Predicate(_is_scalar_other)
no_output_quantizer_predicate = Predicate(_no_output_quantizer)


@register(
    "mul",
    linear_per_tensor_predicate & scalar_other_predicate & no_output_quantizer_predicate,
    # type: ignore[misc, call-overload]
)
def scalar_multiply(
    input: QuantizedTensor, other: float, *args: Any, **kwargs: Any
) -> QuantizedTensor:
    quant_args = input.quant_args()
    quant_args.scale = quant_args.scale * other
    scaled_out = _ScaleGradient.apply(input.raw_data, other)
    return input._quantization_function.rebind(**quant_args).attach(scaled_out)


def cat_predicate(
    tensors: Sequence[QuantizedTensor],
    dim: int = 0,
    output_quantizer: Optional[Quantizer] = None,
    strict_quantization: bool = False,
) -> bool:
    if output_quantizer is not None:
        return False

    # Lazy import to break circular import
    from fastforward.quantization.affine import TiledAffineQuantizationFunction
    from fastforward.quantization.dynamic import TiledDynamicAffineQuantizationFunction

    scale: torch.Tensor | float | None = None
    offset: torch.Tensor | float | None = None

    for tensor in tensors:
        if not isinstance(tensor, QuantizedTensor):
            return False  # type: ignore[unreachable]
        quant_args = tensor.quant_args()
        if not issubclass(
            tensor.quant_func,
            (TiledAffineQuantizationFunction, TiledDynamicAffineQuantizationFunction),
        ):
            return False
        scale_match = scale is None or quant_args.scale == scale
        offset_match = offset is None or quant_args.offset == offset
        if not (scale_match and offset_match):
            return False
        scale = quant_args.scale
        offset = quant_args.offset

    return True


@register("cat", Predicate(cat_predicate))
def cat(
    tensors: Sequence[QuantizedTensor], dim: int = 0, *args: Any, **kwargs: Any
) -> QuantizedTensor:
    output = torch.cat([t.raw_data for t in tensors], dim=dim)
    tile_size = tuple(output.shape)
    quant_args = tensors[0].quant_args()
    quant_args.tile_size = tile_size
    new_quant_func = tensors[0]._quantization_function.rebind(**quant_args)
    return new_quant_func.attach(output)


@register("__getitem__", linear_per_tensor_predicate)
def getitem(input: QuantizedTensor, *args: Any) -> QuantizedTensor:
    return apply_and_reattach(lambda x: x.__getitem__(*args), input)


@register("expand", linear_per_tensor_predicate)
def expand(input: QuantizedTensor, *args: Any) -> QuantizedTensor:
    return apply_and_reattach(lambda x: x.expand(*args), input)


@register("unsqueeze", linear_or_dynamic_linear_predicate)
def unsqueeze(input: QuantizedTensor, dim: int) -> QuantizedTensor:
    tile_size = input.quant_args().tile_size
    data = input.raw_data.unsqueeze(dim)

    if tile_size == "data_shape":
        return input._quantization_function.attach(data)

    elif isinstance(tile_size, tuple):
        new_size = [s for s in tile_size]
        new_size.insert(dim % (len(new_size) + 1), 1)

        new_bound_quant_func = input._quantization_function.rebind(tile_size=torch.Size(new_size))
        return new_bound_quant_func.attach(data)

    else:
        raise TypeError("Unsupported tile_size.")


@register("take_along_dim", linear_per_tensor_predicate)
def take_along_dim(
    input: QuantizedTensor, indices: torch.LongTensor, dim: int | None = None
) -> QuantizedTensor:
    return apply_and_reattach(lambda x: torch.take_along_dim(x, indices, dim=dim), input)


@register("topk", linear_per_tensor_predicate)  # type: ignore[arg-type]
def topk(
    input: QuantizedTensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
) -> torch.return_types.topk:
    values, indices = torch.topk(input, k, dim=dim, largest=largest, sorted=sorted)
    values = input._quantization_function.attach(values)
    return torch.return_types.topk((values, indices))
