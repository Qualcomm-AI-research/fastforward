# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
from typing import Any, Callable

import optree
import torch
import torch._C as _C

import fastforward as ff

from fastforward.nn.quantizer import Quantizer, Tag


class TraceTensor(torch.Tensor):
    """A `torch.Tensor` subclass that tracks the preceding operation and quantizer.

    `TraceTensor` instances store a reference to the `previous_function` (the operation that produced this tensor, set via `__torch_function__`) and
    the `previous_quantizer` (the quantizer layer that consumed or produced this tensor,
    set via `quantizer_override`). This tracking facilitates the annotation of `Quantizer`
    layers with `Tag(before/after:func_name)` metadata, allowing the user to search specific layers in the graph.
    """

    def __new__(cls, data: torch.Tensor) -> "TraceTensor":  # noqa: D102
        return cls._make_subclass(cls, data, require_grad=data.requires_grad)

    def __init__(self, data: Any) -> None:  # noqa: ARG002
        super().__init__()
        self.previous_function: Any = None
        self.previous_quantizer: ff.nn.Quantizer | None = None

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        """Intercepts PyTorch operations on `TraceTensor` instances.

        This method enables the tracking of computational graph flow. It annotates the `previous_quantizer` of input
        `TraceTensor`s with a `Tag("before") / func.__name__` before the operation is performed. It also records the `func`
        (the PyTorch operation) as the `_previous_function` on the resulting `TraceTensor`.
        """
        kwargs = kwargs or {}

        def _annotate_input(arg: Any) -> Any:
            if isinstance(arg, TraceTensor) and isinstance(arg.previous_quantizer, Quantizer):
                if arg.previous_quantizer.quant_metadata is not None:
                    arg.previous_quantizer.quant_metadata.add_tag(Tag("before") / func.__name__)
            return arg

        def _annotate_output(result: Any) -> Any:
            if isinstance(result, torch.Tensor) and not isinstance(result, TraceTensor):
                result = TraceTensor(result)
            if isinstance(result, TraceTensor):
                result.previous_function = func
            return result

        kwargs = kwargs or {}
        args, kwargs = optree.tree_map(_annotate_input, (args, kwargs))  # type: ignore[arg-type]

        with _C.DisableTorchFunctionSubclass():
            output = func(*args, **kwargs)
        (output,) = optree.tree_map(_annotate_output, (output,))  # type: ignore[arg-type]
        return output


def quantizer_override(
    quantizer: ff.nn.Quantizer,
    callback: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    """Register the operations that precede and follow this quantizer forward pass as Tags.

    Combined with `TraceTensor`, this function will add both the preceeding and following operators
    to each quantizer layer as Tag objects: (`Tag("before") / func.__name__`)` and `Tag("after") / func.__name__`).

    Args:
        quantizer: A quantizer layer.
        callback: The quantizer layer forward call.
        args: The arguments passed to the quantizer layer forward call.
        kwargs: Any key-word arguments passed to the quantizer layer forward call.
    """

    def _data(data: torch.Tensor, *_args: Any, **_kwargs: Any) -> torch.Tensor:
        return data

    data = _data(*args, **kwargs)

    if (
        isinstance(data, TraceTensor)
        and data.previous_function is not None
        and quantizer.quant_metadata is not None
    ):
        quantizer.quant_metadata.add_tag(Tag("after") / data.previous_function.__name__)
    quantized = callback(*args, **kwargs)
    if not isinstance(quantized, TraceTensor):
        quantized = TraceTensor(quantized)
    quantized.previous_quantizer = quantizer
    return quantized


def annotate_operator_metadata(model: torch.nn.Module, sample_input: torch.Tensor) -> None:
    """Trace model execution and annotate Quantizer metadata with preceding and succeeding operators.

    This function performs a lightweight tracing of the model's forward pass using a special
    tracing tensor (`TraceTensor`) and quantizer overrides. It identifies the operators that immediately
    precede and follow each `ff.nn.Quantizer` instance in the execution flow and adds
    this information as Tags to the metadata (`quant_metadata`) within the Quantizer objects.

    The annotation happens in-place on the `Quantizer` objects within the provided model.
    We assume that `ff.quantize` has been called on the model to ensure that the model
    contains `Quantizer` instances that need to be annotated.

    Args:
        model: The `torch.nn.Module` instance to be traced and annotated. It is assumed that
               `ff.quantize` has been called on this model.
        sample_input: A sample input tensor to the model. This is used to drive
                      the forward pass for tracing purposes. If the module has conditional logic based on
                      input data, ensure that `sample_input` covers all possible branches or that you run
                      this function with multiple sample inputs to cover all branches.
    """
    trace_tensor = TraceTensor(sample_input)
    handles = []
    try:
        for _, quantizer in model.named_quantizers(skip_stubs=False):
            handles.append(quantizer.register_override(quantizer_override))

        with torch.no_grad(), ff.disable_quantization(model):
            model(trace_tensor)
    finally:
        for handle in handles:
            handle.remove()
