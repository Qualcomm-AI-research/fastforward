# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

#
# Warning: you should not make changes to this file directly.
# This file is generated based on 'src/fastforward/_quantops/quantized_operators.yaml'.
#

from typing import TYPE_CHECKING, Optional, Sequence, TypeAlias, Union

import torch

from fastforward.exceptions import QuantizationError
from fastforward.quantized_tensor import QuantizedTensor

if TYPE_CHECKING:
    from fastforward.nn.quantizer import Quantizer
Size: TypeAlias = Union[torch.Size, list[int], tuple[int, ...]]
__all__ = [
    "linear",
    "conv1d",
    "conv2d",
    "conv3d",
    "softmax",
    "relu",
    "sigmoid",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "avg_pool1d",
    "avg_pool2d",
    "avg_pool3d",
    "embedding",
    "layer_norm",
    "matmul",
    "mm",
    "bmm",
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "sum",
    "bitwise_not",
    "negative",
    "positive",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "floor_divide",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "remainder",
    "silu",
    "gelu",
    "scaled_dot_product_attention",
    "dropout",
    "permute",
    "cat",
    "index_add",
    "cumsum",
]


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:1
def linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    if strict_quantization and not isinstance(weight, QuantizedTensor):
        raise QuantizationError(
            "Expected 'weight' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()

    if bias is not None:
        if isinstance(bias, QuantizedTensor):
            bias = bias.dequantize()

    output = torch.nn.functional.linear(input=input, weight=weight, bias=bias)
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:4
def conv1d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: Union[Size, int] = 1,
    padding: Union[Size, int, str] = 0,
    dilation: Union[Size, int] = 1,
    groups: int = 1,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    if strict_quantization and not isinstance(weight, QuantizedTensor):
        raise QuantizationError(
            "Expected 'weight' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()

    if bias is not None:
        if isinstance(bias, QuantizedTensor):
            bias = bias.dequantize()

    output = torch.nn.functional.conv1d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:7
def conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: Union[Size, int] = 1,
    padding: Union[Size, int, str] = 0,
    dilation: Union[Size, int] = 1,
    groups: int = 1,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    if strict_quantization and not isinstance(weight, QuantizedTensor):
        raise QuantizationError(
            "Expected 'weight' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()

    if bias is not None:
        if isinstance(bias, QuantizedTensor):
            bias = bias.dequantize()

    output = torch.nn.functional.conv2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:10
def conv3d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: Union[Size, int] = 1,
    padding: Union[Size, int, str] = 0,
    dilation: Union[Size, int] = 1,
    groups: int = 1,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    if strict_quantization and not isinstance(weight, QuantizedTensor):
        raise QuantizationError(
            "Expected 'weight' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()

    if bias is not None:
        if isinstance(bias, QuantizedTensor):
            bias = bias.dequantize()

    output = torch.nn.functional.conv3d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:13
def softmax(
    input: torch.Tensor,
    dim: int,
    dtype: torch.dtype | None = None,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    output = torch.softmax(input=input, dim=dim, dtype=dtype)
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:16
def relu(
    input: torch.Tensor,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    output = torch.relu(input=input)
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:19
def sigmoid(
    input: torch.Tensor,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    output = torch.sigmoid(input=input)
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:22
def conv_transpose1d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: Union[Size, int] = 1,
    padding: Union[Size, int] = 0,
    output_padding: Union[Size, int] = 0,
    dilation: Union[Size, int] = 1,
    groups: int = 1,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    if strict_quantization and not isinstance(weight, QuantizedTensor):
        raise QuantizationError(
            "Expected 'weight' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()

    if bias is not None:
        if isinstance(bias, QuantizedTensor):
            bias = bias.dequantize()

    output = torch.nn.functional.conv_transpose1d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:25
def conv_transpose2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: Union[Size, int] = 1,
    padding: Union[Size, int] = 0,
    output_padding: Union[Size, int] = 0,
    dilation: Union[Size, int] = 1,
    groups: int = 1,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    if strict_quantization and not isinstance(weight, QuantizedTensor):
        raise QuantizationError(
            "Expected 'weight' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()

    if bias is not None:
        if isinstance(bias, QuantizedTensor):
            bias = bias.dequantize()

    output = torch.nn.functional.conv_transpose2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:28
def conv_transpose3d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: Union[Size, int] = 1,
    padding: Union[Size, int] = 0,
    output_padding: Union[Size, int] = 0,
    dilation: Union[Size, int] = 1,
    groups: int = 1,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    if strict_quantization and not isinstance(weight, QuantizedTensor):
        raise QuantizationError(
            "Expected 'weight' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()

    if bias is not None:
        if isinstance(bias, QuantizedTensor):
            bias = bias.dequantize()

    output = torch.nn.functional.conv_transpose3d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:31
def avg_pool1d(
    input: torch.Tensor,
    kernel_size: Union[Size, int],
    stride: Union[Size, int],
    padding: Union[Size, int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    output = torch.nn.functional.avg_pool1d(
        input=input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
    )
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:34
def avg_pool2d(
    input: torch.Tensor,
    kernel_size: Union[Size, int],
    stride: Union[Size, int],
    padding: Union[Size, int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    output = torch.nn.functional.avg_pool2d(
        input=input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
    )
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:37
def avg_pool3d(
    input: torch.Tensor,
    kernel_size: Union[Size, int],
    stride: Union[Size, int],
    padding: Union[Size, int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    output = torch.nn.functional.avg_pool3d(
        input=input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
    )
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:40
def embedding(
    input: torch.Tensor,
    weight: torch.Tensor,
    padding_idx: int | None = None,
    max_norm: float | None = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(weight, QuantizedTensor):
        raise QuantizationError(
            "Expected 'weight' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()

    output = torch.nn.functional.embedding(
        input=input,
        weight=weight,
        padding_idx=padding_idx,
        max_norm=max_norm,
        norm_type=norm_type,
        scale_grad_by_freq=scale_grad_by_freq,
        sparse=sparse,
    )
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:43
def layer_norm(
    input: torch.Tensor,
    normalized_shape: tuple[int, ...],
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    if weight is not None:
        if strict_quantization and not isinstance(weight, QuantizedTensor):
            raise QuantizationError(
                "Expected 'weight' to be an instance of 'QuantizedTensor' because strict_quantization=True."
            )

        if isinstance(weight, QuantizedTensor):
            weight = weight.dequantize()

    if bias is not None:
        if isinstance(bias, QuantizedTensor):
            bias = bias.dequantize()

    output = torch.nn.functional.layer_norm(
        input=input, normalized_shape=normalized_shape, weight=weight, bias=bias, eps=eps
    )
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:46
def matmul(
    input: torch.Tensor,
    other: torch.Tensor,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    if strict_quantization and not isinstance(other, QuantizedTensor):
        raise QuantizationError(
            "Expected 'other' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(other, QuantizedTensor):
        other = other.dequantize()

    output = torch.matmul(input=input, other=other)
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:49
def mm(
    input: torch.Tensor,
    mat2: torch.Tensor,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    if strict_quantization and not isinstance(mat2, QuantizedTensor):
        raise QuantizationError(
            "Expected 'mat2' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(mat2, QuantizedTensor):
        mat2 = mat2.dequantize()

    output = torch.mm(input=input, mat2=mat2)
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:52
def bmm(
    input: torch.Tensor,
    mat2: torch.Tensor,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    if strict_quantization and not isinstance(mat2, QuantizedTensor):
        raise QuantizationError(
            "Expected 'mat2' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(mat2, QuantizedTensor):
        mat2 = mat2.dequantize()

    output = torch.bmm(input=input, mat2=mat2)
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:55
def add(
    input: torch.Tensor,
    other: Union[float, torch.Tensor],
    alpha: float = 1.0,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    if (
        strict_quantization
        and isinstance(other, torch.Tensor)
        and not isinstance(other, QuantizedTensor)
    ):
        raise QuantizationError(
            "Expected 'other' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(other, QuantizedTensor):
        other = other.dequantize()

    output = torch.add(input=input, other=other, alpha=alpha)
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:58
def sub(
    input: torch.Tensor,
    other: Union[float, torch.Tensor],
    alpha: float = 1.0,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    if (
        strict_quantization
        and isinstance(other, torch.Tensor)
        and not isinstance(other, QuantizedTensor)
    ):
        raise QuantizationError(
            "Expected 'other' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(other, QuantizedTensor):
        other = other.dequantize()

    output = torch.sub(input=input, other=other, alpha=alpha)
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:61
def mul(
    input: torch.Tensor,
    other: Union[float, torch.Tensor],
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    if (
        strict_quantization
        and isinstance(other, torch.Tensor)
        and not isinstance(other, QuantizedTensor)
    ):
        raise QuantizationError(
            "Expected 'other' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(other, QuantizedTensor):
        other = other.dequantize()

    output = torch.mul(input=input, other=other)
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:64
def div(
    input: torch.Tensor,
    other: Union[float, torch.Tensor],
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    if (
        strict_quantization
        and isinstance(other, torch.Tensor)
        and not isinstance(other, QuantizedTensor)
    ):
        raise QuantizationError(
            "Expected 'other' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(other, QuantizedTensor):
        other = other.dequantize()

    output = torch.div(input=input, other=other)
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:67
def pow(
    input: torch.Tensor,
    exponent: Union[float, torch.Tensor],
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    if (
        strict_quantization
        and isinstance(exponent, torch.Tensor)
        and not isinstance(exponent, QuantizedTensor)
    ):
        raise QuantizationError(
            "Expected 'exponent' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(exponent, QuantizedTensor):
        exponent = exponent.dequantize()

    output = torch.pow(input=input, exponent=exponent)
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:70
def sum(
    input: torch.Tensor,
    dim: int | None = None,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    output = torch.sum(input=input, dim=dim)
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:73
def bitwise_not(
    input: torch.Tensor,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    output = torch.bitwise_not(input=input)
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:76
def negative(
    input: torch.Tensor,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    output = torch.negative(input=input)
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:79
def positive(
    input: torch.Tensor,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    output = torch.positive(input=input)
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:82
def bitwise_and(
    input: torch.Tensor,
    other: torch.Tensor,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    if strict_quantization and not isinstance(other, QuantizedTensor):
        raise QuantizationError(
            "Expected 'other' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(other, QuantizedTensor):
        other = other.dequantize()

    output = torch.bitwise_and(input=input, other=other)
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:85
def bitwise_or(
    input: torch.Tensor,
    other: torch.Tensor,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    if strict_quantization and not isinstance(other, QuantizedTensor):
        raise QuantizationError(
            "Expected 'other' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(other, QuantizedTensor):
        other = other.dequantize()

    output = torch.bitwise_or(input=input, other=other)
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:88
def bitwise_xor(
    input: torch.Tensor,
    other: torch.Tensor,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    if strict_quantization and not isinstance(other, QuantizedTensor):
        raise QuantizationError(
            "Expected 'other' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(other, QuantizedTensor):
        other = other.dequantize()

    output = torch.bitwise_xor(input=input, other=other)
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:91
def floor_divide(
    input: torch.Tensor,
    other: Union[float, torch.Tensor],
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    if (
        strict_quantization
        and isinstance(other, torch.Tensor)
        and not isinstance(other, QuantizedTensor)
    ):
        raise QuantizationError(
            "Expected 'other' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(other, QuantizedTensor):
        other = other.dequantize()

    output = torch.floor_divide(input=input, other=other)
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:94
def bitwise_left_shift(
    input: torch.Tensor,
    other: Union[float, torch.Tensor],
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    if (
        strict_quantization
        and isinstance(other, torch.Tensor)
        and not isinstance(other, QuantizedTensor)
    ):
        raise QuantizationError(
            "Expected 'other' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(other, QuantizedTensor):
        other = other.dequantize()

    output = torch.bitwise_left_shift(input=input, other=other)
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:97
def bitwise_right_shift(
    input: torch.Tensor,
    other: Union[float, torch.Tensor],
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    if (
        strict_quantization
        and isinstance(other, torch.Tensor)
        and not isinstance(other, QuantizedTensor)
    ):
        raise QuantizationError(
            "Expected 'other' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(other, QuantizedTensor):
        other = other.dequantize()

    output = torch.bitwise_right_shift(input=input, other=other)
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:100
def remainder(
    input: torch.Tensor,
    other: Union[float, torch.Tensor],
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    if (
        strict_quantization
        and isinstance(other, torch.Tensor)
        and not isinstance(other, QuantizedTensor)
    ):
        raise QuantizationError(
            "Expected 'other' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(other, QuantizedTensor):
        other = other.dequantize()

    output = torch.remainder(input=input, other=other)
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:103
def silu(
    input: torch.Tensor,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    output = torch.nn.functional.silu(input=input)
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:106
def gelu(
    input: torch.Tensor,
    approximate: str = "none",
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    output = torch.nn.functional.gelu(input=input, approximate=approximate)
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:109
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(query, QuantizedTensor):
        raise QuantizationError(
            "Expected 'query' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(query, QuantizedTensor):
        query = query.dequantize()

    if strict_quantization and not isinstance(key, QuantizedTensor):
        raise QuantizationError(
            "Expected 'key' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(key, QuantizedTensor):
        key = key.dequantize()

    if strict_quantization and not isinstance(value, QuantizedTensor):
        raise QuantizationError(
            "Expected 'value' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(value, QuantizedTensor):
        value = value.dequantize()

    output = torch.nn.functional.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:112
def dropout(
    input: torch.Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    output = torch.nn.functional.dropout(input=input, p=p, training=training, inplace=inplace)
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:115
def permute(
    input: torch.Tensor,
    dims: tuple[int, ...],
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    output = torch.permute(input=input, dims=dims)
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:118
def cat(
    tensors: Sequence[torch.Tensor],
    dim: int = 0,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    elems__: list[torch.Tensor] = []
    for elem__ in tensors:
        if strict_quantization and not isinstance(elem__, QuantizedTensor):
            raise QuantizationError(
                "Expected 'elem__' to be an instance of 'QuantizedTensor' because strict_quantization=True."
            )

        if isinstance(elem__, QuantizedTensor):
            elem__ = elem__.dequantize()
        elems__.append(elem__)
    tensors = elems__

    output = torch.cat(tensors=tensors, dim=dim)
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:121
def index_add(
    input: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    source: torch.Tensor,
    alpha: float = 1,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    if strict_quantization and not isinstance(source, QuantizedTensor):
        raise QuantizationError(
            "Expected 'source' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(source, QuantizedTensor):
        source = source.dequantize()

    output = torch.index_add(input=input, dim=dim, index=index, source=source, alpha=alpha)
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:124
def cumsum(
    input: torch.Tensor,
    dim: int,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool = True,
) -> torch.Tensor:
    if strict_quantization and output_quantizer is None:
        raise QuantizationError("'output_quantizer' must be provided if strict_quantization=True")

    if strict_quantization and not isinstance(input, QuantizedTensor):
        raise QuantizationError(
            "Expected 'input' to be an instance of 'QuantizedTensor' because strict_quantization=True."
        )

    if isinstance(input, QuantizedTensor):
        input = input.dequantize()

    output = torch.cumsum(input=input, dim=dim)
    if output_quantizer is not None:
        output = output_quantizer(output)
    return output
