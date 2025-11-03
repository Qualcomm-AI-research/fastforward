# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""
!!! experimental
    Export is an experimental feature and is currently under active development.
    Please expect API changes. We encourage you to file bug reports if you run into any problems.

"""  # noqa: D205, D212

import inspect
import json
import logging
import pathlib
import pickle

from contextlib import ExitStack
from types import TracebackType
from typing import Any, overload

import onnxruntime  # type: ignore[import-untyped]
import optree
import torch

from typing_extensions import override

from fastforward.export import export
from fastforward.export._export_schemas import (
    EncodingSchemaHandler,
    QuantParametersDict,
    V1SchemaHandler,
)
from fastforward.mpath import MPathCollection
from fastforward.overrides import disable_quantization
from fastforward.quantization.affine.function import StaticAffineQuantParams
from fastforward.quantized_tensor import QuantizedTensor

logger = logging.getLogger(__name__)


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
        """Logs inputs/outputs/kwargs in dictionary."""
        self.input = input_ if isinstance(input_, tuple) else (input_,)
        self.kwargs = kwargs
        self.output = output_ if isinstance(output_, tuple) else (output_,)

        self.input, self.input_quantizer_settings = _deep_dequantize(self.input)
        self.output, self.output_quantizer_settings = _deep_dequantize(self.output)
        self.kwargs, self.kwargs_quantizer_settings = _deep_dequantize(self.kwargs)

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


def export_modules(
    model: torch.nn.Module,
    args: None | tuple[torch.Tensor] | tuple[()],
    module_or_module_collection: torch.nn.Module | MPathCollection,
    model_name: str,
    output_path: pathlib.Path,
    kwargs: None | dict[str, Any] = None,
    enable_encodings_propagation: bool = False,
    verbose: bool | None = None,
    encoding_schema_handler: EncodingSchemaHandler = V1SchemaHandler(),
    alter_node_names: bool = False,
    onnx_export_options: dict[str, Any] | None = None,
) -> dict[str, pathlib.Path]:
    """Export a collection of modules from a given model.

    Main function for module level export. Given a model, data, kwargs and a collection
    of modules, the function will apply hooks for capturing the inputs/outputs for the
    modules of interest in the model. It will then use the captured inputs for exporting
    the individual modules (using `ff.export.export` function).

    NB: Because this is a quantized model, it is likely that some of the tensors
    forwarded through the different layers will not require quantization. For example
    considering the following simple model:

    input -> input_quantization -> layer1 -> output_quantization -> layer2 -> output_quantization

    Here the first layer has an input quantizer, which will be captured in the encodings JSON,
    but the second layer does not, because it relies on the first layer's output quantization.
    In order to capture this information and correctly quantize inputs/outputs of all modules
    we extend (when this is necessary) the encodings file of the different modules (if that is
    required). This is possible because the `QuantizedTensor`s contain the necessary quantization
    information.

    Args:
        model: A torch model from which modules will be exported.
        args: The input args to the torch model.
        module_or_module_collection: A mpath collection of modules to be individually exported.
        model_name: The name of the model, the output directory will be named after it.
        kwargs: The kwargs used at inference for the torch model
        output_path: Path to the exported artifacts.
        enable_encodings_propagation: Option to propagate the quantization encodings through as many
            view-type operations as possible for each exported graph.
        verbose: Whether to print verbose messages. If `None`, some messages will be printed.
        encoding_schema_handler: Object for choosing and creating the appropriate QNN encodings
            file schema
        alter_node_names: Whether to alter the node names in a graph. This is due to some versions
            of QNN creating new nodes that might cause a duplicate name issue.
        onnx_export_options: Dictionary for passing setting to the `torch.onnx.export` function.
            WARNING: Certain options (such as setting the `optimize` to True) can cause misalignments between
            the ONNX graph and the quantization encodings file. For example, if your mode contains linear layers,
            the associated weight transposition will be removed from the graph and the weights permanently transposed.
            If this is combined with per channel quantization on the weights then your encodings will be pointing
            to the wrong dimension of the weights.

    Returns:
        paths: A dictionary of module names to exported paths (location where the encodings
            and ONNX files are stored).
    """
    args = args or ()
    kwargs = kwargs or {}
    output_path = output_path / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    if args == () and kwargs == {}:
        msg = "Both args and kwargs cannot be None at the same time"
        raise ValueError(msg)

    if isinstance(module_or_module_collection, MPathCollection):
        modules = {mod.full_name: mod.module for mod in module_or_module_collection}
    else:
        modules = {model_name: module_or_module_collection}

    with ExitStack() as stack:
        recorders = [
            stack.enter_context(ModuleIORecorder(module, module_name))
            for module_name, module in modules.items()
        ]
        with disable_quantization(model):
            model(*args, **kwargs)
        for rec in recorders:
            module_output_path = output_path / rec.module_name
            module_output_path.mkdir(parents=True, exist_ok=True)

            rec.store_io_as_dict(
                module_output_path / f"{rec.module_name}_nonquantized_input_output.pickle"
            )

    module_io_recorders: list[ModuleIORecorder] = []
    for module_name, module in modules.items():
        module_io_recorder = ModuleIORecorder(module, module_name)
        module_io_recorders.append(module_io_recorder)
        module_io_recorder.attach()

    model(*args, **kwargs)

    for module_io_recorder in module_io_recorders:
        module_io_recorder.detach()

    paths: dict[str, pathlib.Path] = {}

    for module_io_recorder in module_io_recorders:
        module = module_io_recorder.module
        module_name = module_io_recorder.module_name
        module_output_path = output_path / module_name

        module_input_data = module_io_recorder.input
        module_input_kwargs = module_io_recorder.kwargs

        onnx_options = onnx_export_options or {}

        export(
            module,
            module_input_data,
            module_output_path,
            module_name,
            model_kwargs=module_input_kwargs,
            enable_encodings_propagation=enable_encodings_propagation,
            verbose=verbose,
            encoding_schema_handler=encoding_schema_handler,
            alter_node_names=alter_node_names,
            onnx_export_options=onnx_options,
        )

        module_input_quantizer_settings = module_io_recorder.input_quantizer_settings
        module_output_quantizer_settings = module_io_recorder.output_quantizer_settings
        module_kwargs_quantizer_settings = module_io_recorder.kwargs_quantizer_settings
        quantizer_settings = {
            "input": module_input_quantizer_settings,
            "output": module_output_quantizer_settings,
            "kwargs": module_kwargs_quantizer_settings,
        }
        maybe_extend_encodings_file(
            module_name, module_output_path, quantizer_settings, encoding_schema_handler
        )

        input_output_location = module_output_path / f"{module_name}_input_output.pickle"
        module_io_recorder.store_io_as_dict(input_output_location)

        paths[module_name] = module_output_path
        # Since we are using the same handler, we need to clear the entries once the
        # process is complete for each modules. Otherwise, there can be duplicate entries
        # and overriding of encodings from one module to the next.
        encoding_schema_handler.clear()

    return paths


def maybe_extend_encodings_file(
    module_name: str,
    path: pathlib.Path,
    quantizer_settings: dict[str, Any],
    encoding_schema_handler: EncodingSchemaHandler,
) -> None:
    """Extends the QNN encodings file.

    As detailed in the `export_modules` function docstring there are
    cases when quantization is implicit (the tensor passed to the
    operation is already quantized) and it was not captured in the
    encodings file. This function takes a dictionary of quantization
    settings and appends them to the encodings file of the given module.

    NB: the function only looks at inputs/outputs of modules, no other
    activations or parameters.

    Args:
        module_name: Name of the module.
        path: Path where the module output directory is stored.
        quantizer_settings: The quantizer settings gathered during inference.
        encoding_schema_handler: Object for choosing and creating the appropriate QNN encodings
            file schema
    """
    encodings_file_location = path / f"{module_name}.encodings"
    onnx_artifact_location = path / f"{module_name}.onnx"

    with open(encodings_file_location) as fp:
        encodings_dictionary = json.load(fp)

    # In order to associate potential inputs with quantizer settings we need
    # to retrieve the graph input names. This can be done through the onnxruntime
    # library.
    ort_session = onnxruntime.InferenceSession(
        onnx_artifact_location, providers=["CPUExecutionProvider"]
    )

    quantizer_input_settings = quantizer_settings["input"]
    quantizer_kwargs_settings = quantizer_settings["kwargs"]
    ort_session_inputs = ort_session.get_inputs()

    # Iterate first through all the graph inputs. In the case we find that the
    # input name exists in the kwargs quantizer settings, then we know that
    # the argument was passed as kwarg, and we assign its stored encoding.
    # If it is not found in the kwargs we consider it to be a positional input
    # instead. We perform the same check (whether the quantizer settings are
    # not None) and we add the settings to the schema handler.

    # NB: We consider that the order of the positional arguments is the same
    # for the torch module and the ONNX graph.
    quantizer_kwargs_keys = list(quantizer_kwargs_settings.keys())
    positional_input_idx = 0
    for ort_input in ort_session_inputs:
        # First check if the input is defined as kwarg and if that is quantized.
        quant_settings = quantizer_kwargs_settings.get(ort_input.name)
        if quant_settings is not None:
            encoding_schema_handler.add_encoding(ort_input.name, quant_settings, False)
        # Then the input was passed as a positional argument, or there are no positional
        # inputs. We check if the input was quantized and iterate over the input quantizer
        # settings.
        elif positional_input_idx < len(quantizer_input_settings):
            quant_settings = quantizer_input_settings[positional_input_idx]
            if quant_settings is not None:
                encoding_schema_handler.add_encoding(ort_input.name, quant_settings, False)
            positional_input_idx += 1
        else:
            if frame := inspect.currentframe():
                function_name = f"{__name__}.{frame.f_code.co_name}"
            else:
                function_name = f"{__name__}.maybe_extend_encodings_file"
            msg = (
                f"[{function_name}] Input mapping not found (this is NOT necessarily an error):\n"
                f"  • Input: '{ort_input.name}' (module: '{module_name}')\n"
                f"  • Available kwargs: {quantizer_kwargs_keys}\n"
                f"  • Args exhausted: range({len(quantizer_input_settings)})\n"
                f"  • Causes: missing quantizer | pruned export | graph change\n"
                f"  • Action: skipping encodings entry\n"
            )
            logger.warning(msg)

    # Then iterate through the graph outputs and in the case an output is assigned
    # has quantizer settings, then add them to the schema handler.
    quantizer_output_settings = quantizer_settings["output"]
    ort_session_outputs = ort_session.get_outputs()

    for ort_output, output_quantizer_settings in zip(
        ort_session_outputs, quantizer_output_settings
    ):
        if output_quantizer_settings is not None:
            encoding_schema_handler.add_encoding(ort_output.name, output_quantizer_settings, False)

    encodings_dictionary = encoding_schema_handler.build_encodings_dictionary()

    with open(encodings_file_location, "w") as fp:
        json.dump(encodings_dictionary, fp, indent=4)


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
