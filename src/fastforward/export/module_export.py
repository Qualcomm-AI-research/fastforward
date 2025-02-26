# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import collections
import json
import pathlib
import pickle

from collections.abc import Sequence
from typing import Any, Callable

import onnxruntime
import torch

from fastforward.export import export
from fastforward.export._export_helpers import QuantParametersDict, create_qnn_encoding_entry
from fastforward.mpath._search import MPathCollection
from fastforward.quantization.affine.function import StaticAffineQuantParams
from fastforward.quantized_tensor import QuantizedTensor


class IOCaptureHook:
    """Provides functionality for logging inputs/outputs/kwargs."""

    def __init__(self, module: torch.nn.Module, module_name: str):
        self.input_output_registry: dict[str, Any] = {}
        self.module = module
        self.module_name = module_name

    def __call__(
        self,
        _: torch.nn.Module,
        input_: torch.Tensor | tuple[torch.Tensor],
        kwargs: dict[str, Any],
        output_: torch.Tensor | tuple[torch.Tensor],
    ) -> None:
        """Logs inputs/outputs/kwargs in dictionary."""
        self.input_output_registry["input"] = input_ if isinstance(input_, tuple) else (input_,)
        self.input_output_registry["kwargs"] = kwargs
        self.input_output_registry["output"] = output_ if isinstance(output_, tuple) else (output_,)


def export_modules(
    model: torch.nn.Module,
    data: None | torch.Tensor,
    module_or_module_collection: torch.nn.Module | MPathCollection,
    model_name: str,
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
        data: The input data to the torch model.
        module_or_module_collection: A mpath collection of modules to be individually exported.
        model_name: The name of the model, the output directory will be named after it.
        kwargs: The kwargs used at inference for the torch model
    Returns:
        paths: A dictionary of module names to exported paths (location where the encodings
            and ONNX files are stored).
    """
    if data is None and (kwargs is None or kwargs == {}):
        msg = "Both data and kwargs cannot be None at the same time"
        raise ValueError(msg)

    if kwargs is None:
        kwargs = {}

    if isinstance(module_or_module_collection, MPathCollection):
        modules = {mod.full_name: mod.module for mod in module_or_module_collection}
    else:
        modules = {model_name: module_or_module_collection}

    hook_instances = {}
    hooks = []
    for module_name, module in modules.items():
        hook = IOCaptureHook(module, module_name)
        hook_instances[module_name] = hook
        hooks.append(module.register_forward_hook(hook=hook, with_kwargs=True))

    if data is not None:
        model(data, **kwargs)
    else:
        model(**kwargs)

    for h in hooks:
        h.remove()

    paths = {}

    for module_name, module in modules.items():
        input_output_registry = hook_instances[module_name].input_output_registry
        quantizer_settings = maybe_dequantize_tensors(input_output_registry)
        module_input_data = input_output_registry["input"]
        module_input_kwargs = input_output_registry["kwargs"]

        exported_path = export(
            module, module_input_data, model_name, module_name, model_kwargs=module_input_kwargs
        )
        paths[module_name] = exported_path
        maybe_extend_encodings_file(module_name, paths[module_name], quantizer_settings)

        input_output_location = exported_path / f"{module_name}_input_output.pickle"
        with open(input_output_location, "wb") as fp:
            pickle.dump(input_output_registry, fp)

    return paths


def maybe_extend_encodings_file(
    module_name: str,
    path: pathlib.Path,
    quantizer_settings: collections.defaultdict[str, list[QuantParametersDict]],
) -> None:
    """Extends the QNN encodings file.

    As detailed in the `export_modules` function docstring there are
    cases when quantization is implicit (the tensor passed to the
    operation is already quantized) and it cannot be captured in the
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

    activation_encodings_dictionary = encodings_dictionary["activation_encodings"]

    ort_session = onnxruntime.InferenceSession(
        onnx_artifact_location, providers=["CPUExecutionProvider"]
    )

    quantizer_input_settings = quantizer_settings["input"]
    ort_session_inputs = ort_session.get_inputs()

    for ort_input_, quant_settings in zip(ort_session_inputs, quantizer_input_settings):
        ort_input_name = ort_input_.name

        if ort_input_name not in activation_encodings_dictionary:
            qnn_encoding_entry = create_qnn_encoding_entry(quant_settings)
            activation_encodings_dictionary[ort_input_name] = qnn_encoding_entry

    quantizer_output_settings = quantizer_settings["output"]
    ort_session_outputs = ort_session.get_outputs()

    for ort_output_, quant_settings in zip(ort_session_outputs, quantizer_output_settings):
        ort_output_name = ort_output_.name

        if ort_output_name not in activation_encodings_dictionary:
            qnn_encoding_entry = create_qnn_encoding_entry(quant_settings)
            activation_encodings_dictionary[ort_output_name] = qnn_encoding_entry

    encodings_dictionary["activation_encodings"] = activation_encodings_dictionary

    with open(encodings_file_location, "w") as fp:
        json.dump(encodings_dictionary, fp, indent=4)


def maybe_dequantize_tensors(
    input_output_registry: dict[str, Any],
) -> collections.defaultdict[str, list[QuantParametersDict]]:
    """Dequantizes input/output tensors.

    The output tensors of quantized modules will usually be returned
    as `QuantizedTensor`s. As these are custom tensors they cannot be
    used in that form for exporting, and need to be dequantized. This
    function performs this dequantization in the case a `QuantizedTensor`
    is found on the input/output module capture, and it also stores
    its quantization settings, so these can be appended to the encodings
    file.

    Args:
        input_output_registry: Dictionary containing the input/kwargs/output
            data capture of a single module.

    Returns:
        quantizer_settings: Dictionary containing the quantizer parameters
            from any `QuantizedTensor`s found in the `input_output_registry`.
    """
    keys_of_interest = ["input", "output"]
    quantizer_settings: collections.defaultdict[str, list[QuantParametersDict]] = (
        collections.defaultdict(list)
    )

    for key in keys_of_interest:
        assert key in input_output_registry
        updated_tensors = []

        for tensor in input_output_registry[key]:
            if isinstance(tensor, QuantizedTensor):
                quant_args: StaticAffineQuantParams = tensor.quant_args()
                tensor_quant_args: QuantParametersDict = {
                    "scale": quant_args.scale,
                    "offset": quant_args.offset,
                    "num_bits": quant_args.num_bits,
                }
                quantizer_settings[key].append(tensor_quant_args)
                updated_tensors.append(tensor.dequantize())
            else:
                updated_tensors.append(tensor)

        input_output_registry[key] = tuple(updated_tensors)

    return quantizer_settings
