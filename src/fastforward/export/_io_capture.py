# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Model I/O capture helpers shared by module- and whole-model export.

This module is intentionally dependency-light: it imports neither
`fastforward.export.export` nor the pipeline package, so it can be imported from
export stages without creating an import cycle.
"""

import pathlib
import pickle

from types import TracebackType
from typing import Any, overload

import optree
import torch
import torch.utils._pytree as pytree

from typing_extensions import override

from fastforward.export._export_types import QuantParametersDict
from fastforward.quantization.affine.function import StaticAffineQuantParams
from fastforward.quantized_tensor import QuantizedTensor


class ModuleIORecorder:
    """Provides functionality for logging inputs/outputs/kwargs."""

    def __init__(self, module: torch.nn.Module, module_name: str):
        self.module = module
        self.module_name = module_name
        self.handle: None | torch.utils.hooks.RemovableHandle = None

        self.input: tuple[torch.Tensor, ...]
        self.output: tuple[torch.Tensor, ...]
        self.kwargs: dict[str, Any]

        self.input_quantizer_settings: tuple[QuantParametersDict | None, ...]
        self.output_quantizer_settings: tuple[QuantParametersDict | None, ...]
        self.kwargs_quantizer_settings: dict[str, Any]

    def __call__(
        self,
        _: torch.nn.Module,
        input_: torch.Tensor | tuple[torch.Tensor],
        kwargs: dict[str, Any],
        output_: torch.Tensor | tuple[torch.Tensor],
    ) -> None:
        """Logs inputs/outputs/kwargs in dictionary.

        This implements PyTorch's forward hook interface (with `with_kwargs=True`),
        so it is invoked as `hook(module, input_, kwargs, output_)`.
        """
        self.input = input_ if isinstance(input_, tuple) else (input_,)
        self.kwargs = kwargs

        self.input, self.input_quantizer_settings = _deep_dequantize(self.input)
        self.kwargs, self.kwargs_quantizer_settings = _deep_dequantize(self.kwargs)

        # Flatten the output into the same leaf order the export path (torch.export
        # / ONNX) produces, so the stored reference IO is a flat tuple of plain
        # tensors that matches the deployed graph's output and carries no
        # framework-specific container types (e.g. HF `ModelOutput`).
        output_leaves = pytree.tree_leaves(output_)
        self.output, self.output_quantizer_settings = _dequantize_leaves(output_leaves)

    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__} for module {self.module_name}"

    def attach(self) -> None:
        """Attach recorder to tracked module."""
        if self.handle is not None:
            msg = f"Handle for {self} is already attached. Cannot attach a new one."
            raise RuntimeError(msg)
        self.handle = self.module.register_forward_hook(hook=self, with_kwargs=True)

    def detach(self) -> None:
        """Remove recorder from tracked module."""
        if self.handle is not None:
            self.handle.remove()

    def store_io_as_dict(self, location: pathlib.Path) -> None:
        """Store inputs/outputs/kwargs as a pickle dictionary."""
        input_output_registry = {
            "input": self.input,
            "output": self.output,
            "kwargs": self.kwargs,
        }

        with open(location, "wb") as fp:
            pickle.dump(input_output_registry, fp)

    def __enter__(self) -> "ModuleIORecorder":
        """Attach the recorder to the module when entering the context."""
        self.attach()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Detach the recorder from the module when exiting the context."""
        self.detach()


def _deep_detach_tensors(pytree: Any) -> Any:
    """Detach all tensors in a pytree to avoid exporting non-leaf tensors."""
    return optree.tree_map(
        lambda value: value.detach() if isinstance(value, torch.Tensor) else value,
        pytree,
    )


def _dequant_and_get_quantparams(
    tensor: QuantizedTensor,
) -> tuple[torch.Tensor, QuantParametersDict]:
    """Dequantize a tensor and extract its quantization parameters."""
    quant_args = tensor.quant_args()
    assert isinstance(quant_args, StaticAffineQuantParams)
    scale, offset, num_bits = quant_args.scale, quant_args.offset, quant_args.num_bits
    raw_tile_size = quant_args.granularity.tile_size(tensor.shape)
    tile_size = tensor.shape if raw_tile_size == "data_shape" else raw_tile_size

    tensor_quant_args: QuantParametersDict = {
        "scale": scale,
        "offset": offset,
        "num_bits": num_bits,
        "data_shape": tensor.shape,
        "tile_size": tile_size,
    }

    return tensor.dequantize(), tensor_quant_args


def _dequantize_leaves(
    leaves: list[Any],
) -> tuple[tuple[Any, ...], tuple[QuantParametersDict | None, ...]]:
    """Dequantize a flat list of leaves, collecting quantization parameters.

    Unlike `_deep_dequantize`, this operates on an already-flattened sequence of
    leaves and returns flat tuples. It is used for module outputs, which are
    flattened to match the deployment graph's tuple-of-tensors output order.

    Args:
        leaves: A flat list of leaves (tensors and/or other values).

    Returns:
        A flat tuple of (maybe) dequantized leaves and a flat tuple of quantizer
        settings aligned with it (None for non-quantized leaves).
    """
    dequantized: list[Any] = []
    quantizer_settings: list[QuantParametersDict | None] = []
    for leaf in leaves:
        if isinstance(leaf, QuantizedTensor):
            dequant_tensor, quant_setting = _dequant_and_get_quantparams(leaf)
            dequantized.append(dequant_tensor)
            quantizer_settings.append(quant_setting)
        else:
            dequantized.append(leaf)
            quantizer_settings.append(None)
    return tuple(dequantized), tuple(quantizer_settings)


_StructedQuantParams = (
    optree.PyTree[QuantParametersDict | None]
    | tuple[QuantParametersDict | None, ...]
    | dict[str, QuantParametersDict | None]
)


@overload
def _deep_dequantize(
    pytree: tuple[Any, ...],
) -> tuple[tuple[Any, ...], tuple[QuantParametersDict | None, ...]]: ...


@overload
def _deep_dequantize(
    pytree: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, QuantParametersDict | None]]: ...


@overload
def _deep_dequantize(
    pytree: optree.PyTree[Any],
) -> tuple[optree.PyTree[Any], optree.PyTree[QuantParametersDict | None]]: ...


def _deep_dequantize(
    pytree: optree.PyTree[Any] | tuple[Any, ...] | dict[str, Any],
) -> tuple[optree.PyTree[Any] | tuple[Any, ...] | dict[str, Any], _StructedQuantParams]:
    """Dequantizes tensors in a PyTree structure.

    The output tensors of quantized modules will usually be returned
    as `QuantizedTensor`s. As these are custom tensors they cannot be
    used in that form for exporting, and need to be dequantized. This
    function performs this dequantization recursively on any PyTree structure
    in the case a `QuantizedTensor` is found, and it also stores
    its quantization settings, so these can be appended to the encodings
    file.

    Args:
        pytree: A PyTree structure that may contain `QuantizedTensor`s
            at any level of nesting.

    Returns:
        The (maybe) dequantized PyTree with the same structure, and the
        quantizer settings PyTree for each element (None for non-quantized tensors).
    """
    quantizer_settings: list[Any] = []
    dequantized_flat_args: list[Any] = []
    flat_args, treespec = optree.tree_flatten(pytree)  # type: ignore[arg-type]

    for arg in flat_args:
        if isinstance(arg, QuantizedTensor):
            dequant_tensor, quantizer_setting = _dequant_and_get_quantparams(arg)
            dequantized_flat_args.append(dequant_tensor)
            quantizer_settings.append(quantizer_setting)
        else:
            dequantized_flat_args.append(arg)
            quantizer_settings.append(None)

    new_pytree = optree.tree_unflatten(treespec, dequantized_flat_args)
    quantizer_pytree = optree.tree_unflatten(treespec, quantizer_settings)

    return new_pytree, quantizer_pytree
