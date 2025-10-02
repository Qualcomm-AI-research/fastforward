# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import dataclasses
import logging

from typing import Any, Iterable, Literal, Protocol, TypedDict

import torch

from typing_extensions import NotRequired

import fastforward as ff

from fastforward.common import ensure_tensor
from fastforward.export._export_helpers import _strict_cast_to_int
from fastforward.quantization._quantizer_impl import _infer_offset
from fastforward.quantization.affine import integer_minimum, quantization_range
from fastforward.quantization.granularity import granularity_from_sizes

logger = logging.getLogger(__name__)


class QuantParametersDict(TypedDict):
    scale: torch.Tensor | float
    offset: torch.Tensor | float | int | None
    num_bits: float | int
    tile_size: Iterable[int]
    data_shape: Iterable[int]
    output_dtype: NotRequired[torch.dtype]


@dataclasses.dataclass
class ProcessedQuantParams:
    scale: torch.Tensor
    offset: torch.Tensor
    qnn_offset: torch.Tensor
    bitwidth: int
    is_symmetric: bool
    data_shape: torch.Size
    tile_size: torch.Size


def _preprocess_quantization_params(
    encoding_value: QuantParametersDict,
) -> ProcessedQuantParams:
    scale = encoding_value["scale"]
    offset = encoding_value["offset"]
    bitwidth = encoding_value["num_bits"]
    data_shape = torch.Size(encoding_value["data_shape"])
    tile_size = torch.Size(encoding_value["tile_size"])

    scale = ensure_tensor(scale)
    if offset is None:
        offset = _infer_offset(offset, scale)
    offset = ensure_tensor(offset)
    offset = torch.round(offset)

    is_symmetric = (torch.all(offset).item()) == 0
    bitwidth = _strict_cast_to_int(bitwidth, "bitwidth")

    qnn_offset = offset - 2 ** (bitwidth - 1)
    if not isinstance(qnn_offset, torch.Tensor):
        qnn_offset = torch.tensor(qnn_offset)

    return ProcessedQuantParams(
        scale, offset, qnn_offset, bitwidth, is_symmetric, data_shape, tile_size
    )


class EncodingSchemaHandler(Protocol):
    @property
    def version(self) -> str:
        """Return the schema version string."""
        ...

    def build_encodings_dictionary(
        self,
    ) -> dict[str, Any]:
        """Generate encodings dictionary for the schema version."""
        ...

    def add_encoding(self, name: str, encoding: QuantParametersDict, is_param: bool) -> None:
        """Store encoding in hander."""
        ...


class LegacySchemaHandler:
    """Schema handler for legacy QNN encoding format (version 0.6.1).

    This handler generates encodings compatible with legacy QNN versions that expect
    a simple two-category structure separating parameters and activations.

    Schema Structure:
    ================

    Top Level:
    ----------------
    {
        "activation_encodings": {<activation_name>: (<encoding_entry>, ...)},
        "param_encodings": {<parameter_name>: (<encoding_entry>, ...)}
    }

    Encoding Entry Format:
    ---------------------
    {
        "bitwidth": int,           # Quantization bit width (e.g., 8, 16)
        "dtype": str,              # Data type, always "int"
        "is_symmetric": str,       # "True" or "False" as string
        "max": float,              # Maximum quantization range value
        "min": float,              # Minimum quantization range value
        "offset": int,             # Zero-point offset (QNN format: offset - 2^(bitwidth-1))
        "scale": float             # Quantization scale factor
    }

    Note:
        Supports only per tensor and per channel quantization. Per channel is only supported
        for the first axis.
    """

    def __init__(self) -> None:
        self._param_encodings: dict[str, tuple[dict[str, Any], ...]] = {}
        self._activation_encodings: dict[str, tuple[dict[str, Any], ...]] = {}

    @property
    def version(self) -> Literal["0.6.1"]:
        return "0.6.1"

    def add_encoding(self, name: str, encoding: QuantParametersDict, is_param: bool) -> None:
        qparams = _preprocess_quantization_params(encoding)

        granularity = granularity_from_sizes(qparams.data_shape, qparams.tile_size)

        match granularity:
            case ff.PerTensor():
                ...
            case ff.PerChannel():
                if len(granularity.channel_dims) > 1 or granularity.channel_dims[0] != 0:
                    msg = f"Channel quantization dimension for {self.__class__.__name__} can only be 0. "
                    msg += f"Instead received granularity: {granularity}"
                    raise ValueError(msg)
            case ff.PerBlock():
                msg = f"Block quantization is not supported for {self.__class__.__name__}. "
                msg += f"Node: {name} was found to use block quantization"
                raise ValueError(msg)
            case _:
                msg = f"Unsupported granularity type, received: {granularity}."
                raise ValueError(msg)

        int_min = integer_minimum(qparams.bitwidth)

        int_min = _strict_cast_to_int(int_min, "int_min")
        bitwidth = _strict_cast_to_int(qparams.bitwidth, "bitwidth")

        min_range, max_range = quantization_range(qparams.scale, qparams.offset, bitwidth)
        min_range = ensure_tensor(min_range)
        max_range = ensure_tensor(max_range)

        entry = []

        for (
            scale_entry,
            offset_entry,
            min_range_entry,
            max_range_entry,
        ) in zip(qparams.scale, qparams.qnn_offset, min_range, max_range):
            output_entry: dict[str, Any] = {
                "bitwidth": int(bitwidth),
                "dtype": "int",
                "is_symmetric": "True" if qparams.is_symmetric else "False",
                "min": min_range_entry.item(),
                "max": max_range_entry.item(),
                "offset": int(offset_entry),
                "scale": scale_entry.item(),
            }
            entry.append(output_entry)

        if is_param:
            self._param_encodings[name] = tuple(entry)
        else:
            self._activation_encodings[name] = tuple(entry)

    def build_encodings_dictionary(
        self,
    ) -> dict[str, Any]:
        return {
            "version": self.version,
            "param_encodings": self._param_encodings,
            "activation_encodings": self._activation_encodings,
        }


class V1SchemaHandler:
    """Schema handler for QNN encoding format version 1.0.0.

    This handler generates encodings using the first standardized QNN format.
    It is the first format to introduce versioning, and changes the activation
    and parameters structure to use lists instead of dictionaries.

    Schema Structure:
    ================

    Top Level:
    -----------
    {
        "version": "1.0.0", # Required! Otherwise QNN will revert to legacy version.
        "activation_encodings": [<encoding_entry>, ...],
        "param_encodings": [<encoding_entry>, ...]
    }

    Encoding Entry Format:
    ---------------------
    {
        "name": str,                          # Tensor/parameter name
        "enc_type": str,                      # "PER_TENSOR" | "PER_CHANNEL" | "PER_BLOCK" | "LPBQ"
        "dtype": str,                         # "INT" | "FLOAT"
        "bw": int,                            # Bit width (e.g., 8, 16)
        "is_sym": bool,                       # True for symmetric, False for asymmetric
        "scale": [float, ...],                # List of scale values (per-channel if multiple)
        "offset": [int, ...],                 # List of offset values (QNN format)
        "block_size"?: int,                   # Optional: for PER_BLOCK quantization
        "compressed_bw"?: int,                # Optional: for compressed quantization
        "per_block_int_scale"?: [int, ...]    # Optional: for block-wise quantization
    }
    """

    def __init__(self) -> None:
        self._param_encodings: list[dict[str, Any]] = []
        self._activation_encodings: list[dict[str, Any]] = []

    @property
    def version(self) -> Literal["1.0.0"]:
        return "1.0.0"

    def add_encoding(self, name: str, encoding: QuantParametersDict, is_param: bool) -> None:
        qparams = _preprocess_quantization_params(encoding)

        granularity = granularity_from_sizes(qparams.data_shape, qparams.tile_size)

        entry: dict[str, Any] = {
            "name": name,
            "dtype": "INT",
            "is_sym": qparams.is_symmetric,
            "bw": int(qparams.bitwidth),
            "scale": [s.item() for s in qparams.scale],
            "offset": [o.item() for o in qparams.qnn_offset],
        }

        match granularity:
            case ff.PerTensor():
                entry["enc_type"] = "PER_TENSOR"
            case ff.PerChannel():
                if len(granularity.channel_dims) > 1 or granularity.channel_dims[0] != 0:
                    msg = f"Channel quantization dimension for {self.__class__.__name__} can only be 0. "
                    msg += f"Instead received granularity: {granularity}"
                    raise ValueError(msg)
                entry["enc_type"] = "PER_CHANNEL"
            case ff.PerBlock():
                block_dims, block_sizes = granularity.block_dims, granularity.block_sizes

                if len(block_dims) > 1 or len(block_sizes) > 1:
                    msg = f"Multi-dimensional block quantization is not supported with {self.__class__.__name__}. "
                    msg += f"Node: {name} has granularity: {granularity}."
                    raise ValueError(msg)
                entry["enc_type"] = "PER_BLOCK"
                entry["block_size"] = block_sizes[0]
            case _:
                msg = f"Unsupported granularity type, received: {granularity}."
                raise ValueError(msg)

        if is_param:
            self._param_encodings.append(entry)
        else:
            self._activation_encodings.append(entry)

    def build_encodings_dictionary(
        self,
    ) -> dict[str, Any]:
        return {
            "version": self.version,
            "param_encodings": self._param_encodings,
            "activation_encodings": self._activation_encodings,
        }


class V2SchemaHandler:
    """Schema handler for QNN encoding format version 2.0.0.

    This handler generates encodings using the second QNN format version that consolidates
    all encodings into a single list and uses more standardized field names aligned with
    ONNX quantization conventions.

    Schema Structure:
    ================

    Top Level:
    -----------
    {
        "version": "2.0.0", # Required! Otherwise QNN will fall back to the legacy version.
        "encodings": [<encoding_entry>, ...]
    }

    Encoding Entry Format:
    ---------------------
    {
        "name": str,                                                    # Tensor/parameter name
        "output_dtype": str,                                            # "int8", "uint8", "int4", "uint4", etc.
        "y_scale": float | [float, ...] | [[float, ...], ...],          # Scale value(s) - single or per-channel
        "y_zero_point"?: int | [int, ...] | [[int, ...], ...],          # Optional: zero-point(s) for asymmetric
        "axis"?: int,                                                   # Optional: quantization axis for per-channel
        "block_size"?: int,                                             # Optional: for block-wise quantization
        "per_block_int_scale"?: [int, ...],,                            # Optional: integer scales for blocks
        "per_channel_float_scale"?: [float, ...]                        # Optional: float scales for channels
    }
    """

    def __init__(self) -> None:
        self._encodings: list[dict[str, Any]] = []

    @property
    def version(self) -> Literal["2.0.0"]:
        return "2.0.0"

    def add_encoding(self, name: str, encoding: QuantParametersDict, is_param: bool) -> None:
        del is_param
        qparams = _preprocess_quantization_params(encoding)
        granularity = granularity_from_sizes(qparams.data_shape, qparams.tile_size)

        entry: dict[str, Any] = {
            "name": name,
            "output_dtype": ("" if qparams.is_symmetric else "u") + f"int{qparams.bitwidth}",
        }

        match granularity:
            case ff.PerTensor():
                entry["y_scale"] = qparams.scale.item()
                if not qparams.is_symmetric:
                    entry["y_zero_point"] = int(qparams.qnn_offset.item())
            case ff.PerChannel():
                channel_dims = granularity.channel_dims
                if len(channel_dims) > 1:
                    msg = f"{self.__class__.__name__} only supports a single axis, but got {len(channel_dims)}."
                    raise ValueError(msg)

                entry["axis"] = channel_dims[0]
                entry["y_scale"] = [s.item() for s in qparams.scale]

                if not qparams.is_symmetric:
                    entry["y_zero_point"] = [int(o.item()) for o in qparams.qnn_offset]
            case ff.PerBlock():
                block_dims, block_sizes = granularity.block_dims, granularity.block_sizes

                if len(block_dims) > 1 or len(block_sizes) > 1:
                    msg = f"Multi-dimensional block quantization is not supported with {self.__class__.__name__}. "
                    msg += f"Node: {name} has granularity: {granularity}."
                    raise ValueError(msg)

                entry["axis"] = block_dims[0]
                entry["block_size"] = block_sizes[0]
                reconstructed_scale = reconstruct_block_shape(
                    qparams.scale, qparams.data_shape, granularity
                )
                entry["y_scale"] = reconstructed_scale

                if not qparams.is_symmetric:
                    reconstructed_offset = reconstruct_block_shape(
                        qparams.qnn_offset, qparams.data_shape, granularity
                    )
                    entry["y_zero_point"] = reconstructed_offset
            case _:
                msg = f"Unsupported granularity type, received: {granularity}."
                raise ValueError(msg)

        self._encodings.append(entry)

    def build_encodings_dictionary(
        self,
    ) -> dict[str, Any]:
        return {"version": self.version, "encodings": self._encodings}


def reconstruct_block_shape(
    parameter: torch.Tensor, data_shape: torch.Size, granularity: ff.PerBlock
) -> Any:
    """Reconstruct the nested array structure for block quantization parameters.

    This function transforms the FastForward default flat quantization parameters
    (scale/offset) into the nested array structure expected by QAIRT V2.0.0 schema
    for block quantization.

    Args:
        parameter: Flat tensor containing quantization parameters (scale or offset values).
        data_shape: Original shape of the quantized tensor.
        granularity: PerBlock granularity object containing block dimensions, sizes,
            and per-channel dimensions.

    Returns:
        Reconstructed parameter structure:
        - Scalar (float/int): For single-block tensors
        - 1D list: For single-axis block quantization
        - 2D+ nested list: For multi-axis block quantization (if supported)

    Example:
        >>> # Tensor shape: [8, 16], block size: [2, 4] on axes [0, 1]
        >>> # Creates 4x4 = 16 blocks total
        >>> parameter = torch.tensor([0.1, 0.2, 0.3, 0.4, ...])  # 16 values
        >>> result = reconstruct_block_shape(parameter, torch.Size([8, 16]), granularity)
        >>> # Returns: [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], ...]

    Note:
        This function is specifically designed for QAIRT V2.0.0 schema compliance.
        The nested structure allows QAIRT to understand the spatial organization
        of quantization parameters within blocked tensors.
    """
    block_dims = tuple(granularity.block_dims)
    block_sizes = tuple(int(size) for size in granularity.block_sizes)
    per_channel_dims = tuple(granularity.per_channel_dims)

    parameter_shape = []

    for idx, size in enumerate(data_shape):
        if idx in per_channel_dims:
            # Per-channel dimensions: keep original size (one param per channel)
            parameter_shape.append(size)
        elif idx in block_dims:
            # Block dimensions: calculate number of blocks along this axis
            block_idx = block_dims.index(idx)
            block_size = block_sizes[block_idx]
            num_blocks = size // block_size
            parameter_shape.append(num_blocks)

    parameter_array = parameter.detach().cpu().numpy()

    if len(parameter_shape) == 0:
        # Scalar case: single block, return single value
        return parameter_array.item() if len(parameter_array) == 1 else parameter_array.tolist()
    elif len(parameter_shape) == 1:
        # 1D case: single axis blocking, return flat list
        return parameter_array.tolist()
    else:
        # Multi-dimensional case: reshape and return nested structure
        # This creates the nested array structure expected by QAIRT V2.0.0
        reshaped = parameter_array.reshape(parameter_shape)
        return reshaped.tolist()
