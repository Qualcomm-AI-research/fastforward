# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging

from typing import Any, Literal, Protocol, TypedDict

import torch

from typing_extensions import NotRequired

from fastforward.common import ensure_tensor
from fastforward.export._export_helpers import _strict_cast_to_int
from fastforward.quantization._quantizer_impl import _infer_offset
from fastforward.quantization.affine import integer_minimum, quantization_range

logger = logging.getLogger(__name__)


class QuantParametersDict(TypedDict):
    scale: torch.Tensor | float
    offset: torch.Tensor | float | int | None
    num_bits: float | int
    tile_size: NotRequired[tuple[int]]
    output_dtype: NotRequired[torch.dtype]


def _preprocess_quantization_params(
    encoding_value: QuantParametersDict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, bool]:
    scale = encoding_value["scale"]
    offset = encoding_value["offset"]
    bitwidth = encoding_value["num_bits"]

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

    return scale, offset, qnn_offset, bitwidth, is_symmetric


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
        scale, offset, qnn_offset, bitwidth, is_symmetric = _preprocess_quantization_params(
            encoding
        )

        int_min = integer_minimum(bitwidth)

        int_min = _strict_cast_to_int(int_min, "int_min")
        bitwidth = _strict_cast_to_int(bitwidth, "bitwidth")

        min_range, max_range = quantization_range(scale, offset, bitwidth)
        min_range = ensure_tensor(min_range)
        max_range = ensure_tensor(max_range)

        entry = []

        for (
            scale_entry,
            offset_entry,
            original_offset_entry,
            min_range_entry,
            max_range_entry,
        ) in zip(scale, qnn_offset, offset, min_range, max_range):
            output_entry: dict[str, Any] = {
                "bitwidth": int(bitwidth),
                "dtype": "int",
                "is_symmetric": "True" if is_symmetric else "False",
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
        scale, _, qnn_offset, bitwidth, is_symmetric = _preprocess_quantization_params(encoding)

        entry: dict[str, Any] = {
            "name": name,
            "dtype": "INT",
            "enc_type": "PER_TENSOR" if len(scale) == 1 else "PER_CHANNEL",
            "is_sym": is_symmetric,
            "bw": int(bitwidth),
            "scale": [s.item() for s in scale],
            "offset": [o.item() for o in qnn_offset],
        }
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
        "name": str,                                # Tensor/parameter name
        "output_dtype": str,                        # "int8", "uint8", "int4", "uint4", etc.
        "y_scale": float | [float, ...],            # Scale value(s) - single or per-channel
        "y_zero_point"?: int | [int, ...],          # Optional: zero-point(s) for asymmetric
        "axis"?: int,                               # Optional: quantization axis for per-channel
        "block_size"?: int,                         # Optional: for block-wise quantization
        "per_block_int_scale"?: [int, ...],         # Optional: integer scales for blocks
        "per_channel_float_scale"?: [float, ...]    # Optional: float scales for channels
    }
    """

    def __init__(self) -> None:
        self._encodings: list[dict[str, Any]] = []

    @property
    def version(self) -> Literal["2.0.0"]:
        return "2.0.0"

    def add_encoding(self, name: str, encoding: QuantParametersDict, is_param: bool) -> None:
        del is_param

        scale, _, qnn_offset, bitwidth, is_symmetric = _preprocess_quantization_params(encoding)
        tile_size = encoding.get("tile_size")

        entry: dict[str, Any] = {
            "name": name,
            "output_dtype": ("" if is_symmetric else "u") + f"int{bitwidth}",
            "y_scale": [s.item() for s in scale] if len(scale) > 1 else scale.item(),
        }

        if not is_symmetric:
            entry["y_zero_point"] = (
                [int(o.item()) for o in qnn_offset]
                if len(qnn_offset) > 1
                else int(qnn_offset.item())
            )

        if len(scale) > 1 and tile_size is not None:
            channel_axis = next((i for i, value in enumerate(tile_size) if value == 1), None)
            if channel_axis is not None:
                entry["axis"] = channel_axis

        self._encodings.append(entry)

    def build_encodings_dictionary(
        self,
    ) -> dict[str, Any]:
        return {"version": self.version, "encodings": self._encodings}
