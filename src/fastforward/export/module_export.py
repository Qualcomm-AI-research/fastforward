# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import collections
import json
import pathlib

from typing import Any, Callable, Sequence

import onnxruntime
import torch

from fastforward.export import export
from fastforward.export._export_helpers import create_qnn_encoding_entry, QuantParametersDict
from fastforward.mpath._search import MPathCollection
from fastforward.quantization.affine.function import StaticAffineQuantParams
from fastforward.quantized_tensor import QuantizedTensor

input_output_registry: dict[str | torch.nn.Module, dict[str, Any]]


def export_modules(
    model: torch.nn.Module,
    data: torch.Tensor,
    module_collection: MPathCollection,
    model_name: str,
    kwargs: None | dict[str, Any] = None,
) -> dict[str, pathlib.Path]:
    """
    Export a collection of modules from a given model.

    Main helper function for facilitating module level export. Given
    a model, data, kwargs and a collection of modules, the function will apply
    hooks for capturing the inputs/outputs for the modules of interest in the
    model. It will then use the captured inputs for exporting the individual
    modules (using the FF `export` function).

    NB: Because this is a quantized model, it is likely that some of the tensors
    forwarded through the different layers will not require quantization. For example
    considering the following simple model:

    input -> input_quantization -> layer1 -> output_quantization -> layer2 -> output_quantization

    Here the first layer has an input quantizer, which will be captured in the encodings json,
    but the second layer does not, because it relies on the first layer's output quantization.
    In order to capture this information and correctly quantize inputs/outputs of all modules
    we extend (when this is necessary) the encodings file of the different modules (if that is
    required). This is possible because the `QuantizedTensor`s contain the necessary quantization
    information.

    Args:
        model: a torch model from which modules will be exported.
        data: the input data to the torch model.
        module_collection: a mpath collection of modules to be individually exported.
        model_name: the name of the model, the output directory will be named after it.
        kwargs: the kwargs used at inference for the torch model
    Returns:
        paths: a dictionary of module names to exported paths (location where the encodings
            and ONNX files are stored).
    """
    # The dictionary is set to global in order to store the inputs/kwargs/outputs from
    # the torch hooks.
    global input_output_registry
    input_output_registry = {}
    modules = {module.full_name: module.module for module in module_collection}
    paths = {}

    if not kwargs:
        kwargs = {}

    _populate_input_output_data_registry(
        model, data, list(modules.values()), log_input_output_hook, kwargs
    )

    for module_name, module in modules.items():
        quantizer_settings = maybe_dequantize_tensors(input_output_registry[module_name])
        module_input_data = input_output_registry[module_name]["input"]
        module_input_kwargs = input_output_registry[module_name]["kwargs"]

        exported_path = export(
            module, module_input_data, model_name, module_name, model_kwargs=module_input_kwargs
        )
        paths[module_name] = exported_path
        maybe_extend_encodings_file(module_name, paths[module_name], quantizer_settings)

    return paths


def maybe_extend_encodings_file(
    module_name: str,
    path: pathlib.Path,
    quantizer_settings: collections.defaultdict[str, list[QuantParametersDict]],
) -> None:
    """
    Conditional extension of the QNN encodings file.

    As detailed in the `export_modules` function docstring there are
    cases when quantization is implicit (the tensor traveling in the
    operation is already quantized) and it cannot be captured in the
    encodings file. This function performs a takes a dictionary of
    quantization settings and appends them to the encodings file of
    the given module.

    NB: the function only looks at inputs/outputs of modules, no other
    activations or parameters.

    Args:
        module_name: Name of the module.
        path: path where the module output directory is stored.
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
    """
    Conditional dequantization of tensors.

    The output tensors of quantized modules will usually be returned
    as `QuantizedTensor`s. As these are custom tensors they cannot be
    used in that form for exporting, and need to be dequantized. This
    function performs this dequantization in the case a `QuantizedTensor`
    is found on the input/output module capture, and it also stores
    its quantization settings, so these can be appended to the encodings
    file.

    Args:
        input_output_registry: dictionary containing the input/kwargs/output
            data capture of a single module.
    Returns:
        quantizer_settings: dictionary containing the quantizer parameters
            from any `QuantizedTensor`s found in the `input_output_registry`.
    """
    keys_of_interest = ["input", "output"]
    quantizer_settings = collections.defaultdict(list)

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


def log_input_output_hook(
    module: torch.nn.Module,
    input_: torch.Tensor | tuple[torch.Tensor],
    kwargs: dict[str, Any],
    output_: torch.Tensor | tuple[torch.Tensor],
) -> None:
    """
    Simple hook for logging inputs/kwargs/outputs of modules.

    Args:
        module: the module to which the hook is applied.
        input_: input tensors to the module.
        kwargs: any kwargs passed to the module.
        output_: output tensors from the module.
    """
    input_output_registry[module] = {}
    input_output_registry[module]["input"] = input_ if isinstance(input_, tuple) else (input_,)
    input_output_registry[module]["kwargs"] = kwargs
    input_output_registry[module]["output"] = output_ if isinstance(output_, tuple) else (output_,)


def _change_keys_to_result_registry(
    model: torch.nn.Module, input_output_registry: dict[torch.nn.Module | str, dict[str, Any]]
) -> None:
    """
    Change the key from module object, to be the module object's name.

    The population of the dictionary is happening in the hook mechanism, while
    we have access to the module itself, we do not have access to the module's name.
    Therefore we need to iterate through the model's modules and replace the key
    from the module object to the module name.

    Args:
        model: the torch model for which modules are exported.
        input_output_registry: dictionary containing the inputs/kwargs/outputs
            for all modules of interest.
    """
    for name, module in model.named_modules():
        if module in input_output_registry:
            result = input_output_registry.pop(module)
            input_output_registry[name] = result


def _add_hooks(
    modules: Sequence[torch.nn.modules.Module], hook: Callable[[Any], None],
) -> list[torch.utils.hooks.RemovableHandle]:
    """
    Helper function for registering hooks.

    Args:
        modules: List of modules to which the hook will be applied.
        hook: The hook to be applied to the modules of interest.
    Returns:
        hooks: List of hooks applied to the modules of interest.
    """
    hooks = []
    for module in modules:
        hooks.append(module.register_forward_hook(hook=hook, with_kwargs=True))
    return hooks


def _populate_input_output_data_registry(
    model: torch.nn.Module,
    input_data: torch.Tensor,
    modules: Sequence[torch.nn.Module],
    hook: Callable[[Any], None],
    kwargs: None | dict[str, Any],
) -> None:
    """
    Population of the dictionary with data gathered from torch hooks.

    Function for adding hooks, running model inference and cleaning up.

    Args:
        model: torch model to run inference
        input_data: data to infer on
        modules: collection of modules for which the inputs/kwargs/outputs will be captured.
        hook: torch hook to apply to modules of interest.
        kwargs: any additional kwargs required for inference.
    """

    if kwargs is None:
        kwargs = {}

    hooks = _add_hooks(modules=modules, hook=hook)
    model(input_data, **kwargs)

    _change_keys_to_result_registry(model, input_output_registry)
    for h in hooks:
        h.remove()
