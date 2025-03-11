# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import json
import pathlib
import pickle

from typing import Any

import onnxruntime  # type: ignore[import-untyped]
import torch

from typing_extensions import override

from fastforward.export import export
from fastforward.export._export_helpers import QuantParametersDict, create_qnn_encoding_entry
from fastforward.mpath import MPathCollection
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

        self.input_quantizer_settings: tuple[QuantParametersDict, ...]
        self.output_quantizer_settings: tuple[QuantParametersDict, ...]

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

        self.input, self.input_quantizer_settings = maybe_dequantize_tensors(self.input)
        self.output, self.output_quantizer_settings = maybe_dequantize_tensors(self.output)

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


def export_modules(
    model: torch.nn.Module,
    args: None | tuple[torch.Tensor] | tuple[()],
    module_or_module_collection: torch.nn.Module | MPathCollection,
    model_name: str,
    output_path: pathlib.Path,
    kwargs: None | dict[str, Any] = None,
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

    Returns:
        paths: A dictionary of module names to exported paths (location where the encodings
            and ONNX files are stored).
    """
    args = args or ()
    kwargs = kwargs or {}
    output_path = output_path / model_name

    if args == () and kwargs == {}:
        msg = "Both args and kwargs cannot be None at the same time"
        raise ValueError(msg)

    if isinstance(module_or_module_collection, MPathCollection):
        modules = {mod.full_name: mod.module for mod in module_or_module_collection}
    else:
        modules = {model_name: module_or_module_collection}

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

        module_input_data = module_io_recorder.input
        module_input_kwargs = module_io_recorder.kwargs

        exported_path = export(
            module,
            module_input_data,
            output_path,
            module_name,
            model_kwargs=module_input_kwargs,
        )

        module_input_quantizer_settings = module_io_recorder.input_quantizer_settings
        module_output_quantizer_settings = module_io_recorder.output_quantizer_settings
        quantizer_settings = {
            "input": module_input_quantizer_settings,
            "output": module_output_quantizer_settings,
        }
        maybe_extend_encodings_file(module_name, exported_path, quantizer_settings)

        input_output_location = exported_path / f"{module_name}_input_output.pickle"
        module_io_recorder.store_io_as_dict(input_output_location)

        paths[module_name] = exported_path

    return paths


def maybe_extend_encodings_file(
    module_name: str,
    path: pathlib.Path,
    quantizer_settings: dict[str, tuple[QuantParametersDict, ...]],
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
    """
    encodings_file_location = path / f"{module_name}.encodings"
    onnx_artifact_location = path / f"{module_name}.onnx"

    with open(encodings_file_location) as fp:
        encodings_dictionary = json.load(fp)

    # The existing encodings dictionary (only its `activation_encodings` section)
    # might need to be overwritten.
    activation_encodings_dictionary = encodings_dictionary["activation_encodings"]

    # In order to associate potential inputs with quantizer settings we need
    # to retrieve the graph input names. This can be done through the onnxruntime
    # library.
    ort_session = onnxruntime.InferenceSession(
        onnx_artifact_location, providers=["CPUExecutionProvider"]
    )

    quantizer_input_settings = quantizer_settings["input"]
    ort_session_inputs = ort_session.get_inputs()

    # Here we iterate through the quantizer input settings and in the case that
    # an input does not appear on the original encodings file then we create
    # a new entry for it.
    # NB: We cannot retrieve the input's name from the QuantizedTensor, because
    # we are operating at the module and not graph level. However, we can consider
    # that the inputs will appear in the same sequence in both graph and module.
    for input_idx, quant_settings in enumerate(quantizer_input_settings):
        ort_input_name = ort_session_inputs[input_idx].name

        if ort_input_name not in activation_encodings_dictionary:
            qnn_encoding_entry = create_qnn_encoding_entry(quant_settings)
            activation_encodings_dictionary[ort_input_name] = qnn_encoding_entry

    # The same process detailed for appending input encodings to the encodings
    # dictionary is mirrored for the output encodings.
    quantizer_output_settings = quantizer_settings["output"]
    ort_session_outputs = ort_session.get_outputs()

    for output_idx, quant_settings in enumerate(quantizer_output_settings):
        ort_output_name = ort_session_outputs[output_idx].name

        if ort_output_name not in activation_encodings_dictionary:
            qnn_encoding_entry = create_qnn_encoding_entry(quant_settings)
            activation_encodings_dictionary[ort_output_name] = qnn_encoding_entry

    # We overwrite the `activation_encodings` of the original encodings dictionary
    # so it includes any non-explicitly quantized inputs and store the updated version.
    encodings_dictionary["activation_encodings"] = activation_encodings_dictionary

    with open(encodings_file_location, "w") as fp:
        json.dump(encodings_dictionary, fp, indent=4)


def maybe_dequantize_tensors(
    tensors: tuple[torch.Tensor],
) -> tuple[tuple[torch.Tensor, ...], tuple[QuantParametersDict, ...]]:
    """Dequantizes tensors.

    The output tensors of quantized modules will usually be returned
    as `QuantizedTensor`s. As these are custom tensors they cannot be
    used in that form for exporting, and need to be dequantized. This
    function performs this dequantization in the case a `QuantizedTensor`
    is found on the input/output module capture, and it also stores
    its quantization settings, so these can be appended to the encodings
    file.

    Args:
        tensors: A tuple of tensors, where some of these may be
            `QuantizedTensor`s.

    Returns:
        The (maybe) dequantized tensors, and the quantizer settings for
        each of those.
    """
    output_tensors: list[torch.Tensor] = []
    quantizer_settings: list[QuantParametersDict] = []

    for tensor in tensors:
        if isinstance(tensor, QuantizedTensor):
            quant_args = tensor.quant_args()
            assert isinstance(quant_args, StaticAffineQuantParams)
            scale, offset, num_bits = quant_args.scale, quant_args.offset, quant_args.num_bits
            tensor_quant_args: QuantParametersDict = {
                "scale": scale,
                "offset": offset,
                "num_bits": num_bits,
            }

            quantizer_settings.append(tensor_quant_args)
            output_tensors.append(tensor.dequantize())
        else:
            output_tensors.append(tensor)

    return tuple(output_tensors), tuple(quantizer_settings)
