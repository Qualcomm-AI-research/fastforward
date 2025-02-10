# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

#
# Warning: you should not make changes to this file directly.
# This file is generated based on 'src/fastforward/_quantops/quantized_operators.yaml'.
#

from typing import Optional, Sequence, TypeAlias, Union

import torch

import fastforward

from fastforward.dispatcher import dispatch
from fastforward.nn.quantizer import Quantizer

from . import fallback

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
    "mul",
    "div",
    "sum",
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
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = fastforward.get_strict_quantization()

    dispatch_op = dispatch(
        "linear",
        input=input,
        weight=weight,
        bias=bias,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op or fallback.linear
    return selected_op(
        input=input,
        weight=weight,
        bias=bias,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )


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
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = fastforward.get_strict_quantization()

    dispatch_op = dispatch(
        "conv1d",
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op or fallback.conv1d
    return selected_op(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )


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
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = fastforward.get_strict_quantization()

    dispatch_op = dispatch(
        "conv2d",
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op or fallback.conv2d
    return selected_op(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )


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
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = fastforward.get_strict_quantization()

    dispatch_op = dispatch(
        "conv3d",
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op or fallback.conv3d
    return selected_op(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:13
def softmax(
    input: torch.Tensor,
    dim: int,
    dtype: torch.dtype | None = None,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = fastforward.get_strict_quantization()

    dispatch_op = dispatch(
        "softmax",
        input=input,
        dim=dim,
        dtype=dtype,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op or fallback.softmax
    return selected_op(
        input=input,
        dim=dim,
        dtype=dtype,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:16
def relu(
    input: torch.Tensor,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = fastforward.get_strict_quantization()

    dispatch_op = dispatch(
        "relu",
        input=input,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op or fallback.relu
    return selected_op(
        input=input, output_quantizer=output_quantizer, strict_quantization=strict_quantization
    )


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:19
def sigmoid(
    input: torch.Tensor,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = fastforward.get_strict_quantization()

    dispatch_op = dispatch(
        "sigmoid",
        input=input,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op or fallback.sigmoid
    return selected_op(
        input=input, output_quantizer=output_quantizer, strict_quantization=strict_quantization
    )


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
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = fastforward.get_strict_quantization()

    dispatch_op = dispatch(
        "conv_transpose1d",
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op or fallback.conv_transpose1d
    return selected_op(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )


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
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = fastforward.get_strict_quantization()

    dispatch_op = dispatch(
        "conv_transpose2d",
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op or fallback.conv_transpose2d
    return selected_op(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )


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
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = fastforward.get_strict_quantization()

    dispatch_op = dispatch(
        "conv_transpose3d",
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op or fallback.conv_transpose3d
    return selected_op(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )


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
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = fastforward.get_strict_quantization()

    dispatch_op = dispatch(
        "avg_pool1d",
        input=input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op or fallback.avg_pool1d
    return selected_op(
        input=input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )


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
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = fastforward.get_strict_quantization()

    dispatch_op = dispatch(
        "avg_pool2d",
        input=input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op or fallback.avg_pool2d
    return selected_op(
        input=input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )


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
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = fastforward.get_strict_quantization()

    dispatch_op = dispatch(
        "avg_pool3d",
        input=input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op or fallback.avg_pool3d
    return selected_op(
        input=input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )


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
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = fastforward.get_strict_quantization()

    dispatch_op = dispatch(
        "embedding",
        input=input,
        weight=weight,
        padding_idx=padding_idx,
        max_norm=max_norm,
        norm_type=norm_type,
        scale_grad_by_freq=scale_grad_by_freq,
        sparse=sparse,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op or fallback.embedding
    return selected_op(
        input=input,
        weight=weight,
        padding_idx=padding_idx,
        max_norm=max_norm,
        norm_type=norm_type,
        scale_grad_by_freq=scale_grad_by_freq,
        sparse=sparse,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:43
def layer_norm(
    input: torch.Tensor,
    normalized_shape: tuple[int, ...],
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = fastforward.get_strict_quantization()

    dispatch_op = dispatch(
        "layer_norm",
        input=input,
        normalized_shape=normalized_shape,
        weight=weight,
        bias=bias,
        eps=eps,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op or fallback.layer_norm
    return selected_op(
        input=input,
        normalized_shape=normalized_shape,
        weight=weight,
        bias=bias,
        eps=eps,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:46
def matmul(
    input: torch.Tensor,
    other: torch.Tensor,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = fastforward.get_strict_quantization()

    dispatch_op = dispatch(
        "matmul",
        input=input,
        other=other,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op or fallback.matmul
    return selected_op(
        input=input,
        other=other,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:49
def mm(
    input: torch.Tensor,
    mat2: torch.Tensor,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = fastforward.get_strict_quantization()

    dispatch_op = dispatch(
        "mm",
        input=input,
        mat2=mat2,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op or fallback.mm
    return selected_op(
        input=input,
        mat2=mat2,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:52
def bmm(
    input: torch.Tensor,
    mat2: torch.Tensor,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = fastforward.get_strict_quantization()

    dispatch_op = dispatch(
        "bmm",
        input=input,
        mat2=mat2,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op or fallback.bmm
    return selected_op(
        input=input,
        mat2=mat2,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:55
def add(
    input: torch.Tensor,
    other: Union[float, torch.Tensor],
    alpha: float = 1.0,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = fastforward.get_strict_quantization()

    dispatch_op = dispatch(
        "add",
        input=input,
        other=other,
        alpha=alpha,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op or fallback.add
    return selected_op(
        input=input,
        other=other,
        alpha=alpha,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:58
def mul(
    input: torch.Tensor,
    other: Union[float, torch.Tensor],
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = fastforward.get_strict_quantization()

    dispatch_op = dispatch(
        "mul",
        input=input,
        other=other,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op or fallback.mul
    return selected_op(
        input=input,
        other=other,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:61
def div(
    input: torch.Tensor,
    other: Union[float, torch.Tensor],
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = fastforward.get_strict_quantization()

    dispatch_op = dispatch(
        "div",
        input=input,
        other=other,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op or fallback.div
    return selected_op(
        input=input,
        other=other,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:64
def sum(
    input: torch.Tensor,
    dim: int | None = None,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = fastforward.get_strict_quantization()

    dispatch_op = dispatch(
        "sum",
        input=input,
        dim=dim,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op or fallback.sum
    return selected_op(
        input=input,
        dim=dim,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:67
def silu(
    input: torch.Tensor,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = fastforward.get_strict_quantization()

    dispatch_op = dispatch(
        "silu",
        input=input,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op or fallback.silu
    return selected_op(
        input=input, output_quantizer=output_quantizer, strict_quantization=strict_quantization
    )


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:70
def gelu(
    input: torch.Tensor,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = fastforward.get_strict_quantization()

    dispatch_op = dispatch(
        "gelu",
        input=input,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op or fallback.gelu
    return selected_op(
        input=input, output_quantizer=output_quantizer, strict_quantization=strict_quantization
    )


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:73
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
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = fastforward.get_strict_quantization()

    dispatch_op = dispatch(
        "scaled_dot_product_attention",
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op or fallback.scaled_dot_product_attention
    return selected_op(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:76
def dropout(
    input: torch.Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = fastforward.get_strict_quantization()

    dispatch_op = dispatch(
        "dropout",
        input=input,
        p=p,
        training=training,
        inplace=inplace,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op or fallback.dropout
    return selected_op(
        input=input,
        p=p,
        training=training,
        inplace=inplace,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:79
def permute(
    input: torch.Tensor,
    dims: tuple[int, ...],
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = fastforward.get_strict_quantization()

    dispatch_op = dispatch(
        "permute",
        input=input,
        dims=dims,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op or fallback.permute
    return selected_op(
        input=input,
        dims=dims,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:82
def cat(
    tensors: Sequence[torch.Tensor],
    dim: int = 0,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = fastforward.get_strict_quantization()

    dispatch_op = dispatch(
        "cat",
        tensors=tensors,
        dim=dim,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op or fallback.cat
    return selected_op(
        tensors=tensors,
        dim=dim,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:85
def index_add(
    input: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    source: torch.Tensor,
    alpha: float = 1,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = fastforward.get_strict_quantization()

    dispatch_op = dispatch(
        "index_add",
        input=input,
        dim=dim,
        index=index,
        source=source,
        alpha=alpha,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op or fallback.index_add
    return selected_op(
        input=input,
        dim=dim,
        index=index,
        source=source,
        alpha=alpha,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )


# Automatically generated based on src/fastforward/_quantops/quantized_operators.yaml:88
def cumsum(
    input: torch.Tensor,
    dim: int,
    *,
    output_quantizer: Optional["Quantizer"] = None,
    strict_quantization: bool | None = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = fastforward.get_strict_quantization()

    dispatch_op = dispatch(
        "cumsum",
        input=input,
        dim=dim,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op or fallback.cumsum
    return selected_op(
        input=input,
        dim=dim,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
