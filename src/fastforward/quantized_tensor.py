# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import contextlib
import copy
import functools
import warnings

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Optional,
    ParamSpec,
    Sequence,
    TypeVar,
    cast,
    overload,
)

import optree
import torch
import torch.overrides

from torch import DisableTorchFunction, Tensor
from torch._C import DisableTorchFunctionSubclass
from torch._C._nn import _parse_to as parse_to_args
from typing_extensions import Never, Self

import fastforward

from fastforward.dispatcher import DispatcherPriority, Predicate, dispatch, register
from fastforward.exceptions import QuantizationError

if TYPE_CHECKING:
    from fastforward.quantization.function import (
        QuantizationContext,
        QuantizationFunction,
        QuantizationParameters,
    )

_TensorT = TypeVar("_TensorT", bound="QuantizedTensor")
_P = ParamSpec("_P")
_T = TypeVar("_T")
_U = TypeVar("_U")


def _to_dtype(dtype: torch.dtype, qtensor: "QuantizedTensor") -> torch.Tensor:
    """
    Dequantize quantized tensor and convert to `dtype`.
    """
    return qtensor.dequantize().to(dtype)


def _torch_function_to_op_name(func: Callable[..., Any]) -> str:
    """
    Convert a pointer to a torch Tensor function to simple name.

    E.g. `torch.add -> "add"`, `torch.Tensor.add -> "add"`. The function returns a primitive name, unlike
    `torch.overrides.resolve_name` (`torch.add -> "torch.add"`, `torch.Tensor.add -> "torch.Tensor.add"`).
    """
    return func.__name__


# Register casting functions
register("double", None, functools.partial(_to_dtype, torch.double))
register("float", None, functools.partial(_to_dtype, torch.float))
register("half", None, functools.partial(_to_dtype, torch.half))
register("bfloat16", None, functools.partial(_to_dtype, torch.bfloat16))
register("long", None, functools.partial(_to_dtype, torch.int64))
register("int", None, functools.partial(_to_dtype, torch.int32))
register("short", None, functools.partial(_to_dtype, torch.int16))
register("char", None, functools.partial(_to_dtype, torch.int8))
register("cdouble", None, functools.partial(_to_dtype, torch.complex128))
register("cfloat", None, functools.partial(_to_dtype, torch.complex64))
register("chalf", None, functools.partial(_to_dtype, torch.complex32))
register("bool", None, functools.partial(_to_dtype, torch.bool))
register("byte", None, functools.partial(_to_dtype, torch.uint8))


def _remove_default_implementation(
    func: Callable[_P, Any], func_name: Optional[str] = None, msg: Optional[str] = None
) -> None:
    """
    Remove default implementation from torch for quantized tensors.

    For some functions that are implemented on torch.Tensor, there is no single way
    to implement it for a quantized tensor as it may depend on the quantized representation.

    These implementations can be provided through the dispatcher, but if none is provided,
    a NotImplementedError should be raised. This function effectively removes the default
    implementation for func and raises a NotImplementedError instead.
    """
    func_name = func_name or _torch_function_to_op_name(func)
    if msg is None:
        msg = (
            f"{func_name} is not implemented for QuantizedTensor. This can happen even when "
            "torch.Tensor does have an implementation as it may not generalize to the quantized "
            f"representation. A user implementation of {func_name} can be registered through the "
            "QuantizedTensor dispatcher system. See the documentation of "
            "`fastforward.dispatcher.register` for more details."
        )

    def not_implemented_impl(*args: _P.args, **kwargs: _P.kwargs) -> Never:
        raise NotImplementedError(msg)

    not_implemented_impl.__name__ = f"{func.__name__}_not_implemented"

    register(func_name, None, not_implemented_impl, DispatcherPriority.NOT_IMPLEMENTED_FALLBACK)


_remove_default_implementation(torch.Tensor.__getitem__)
_remove_default_implementation(torch.Tensor.__reversed__)
_remove_default_implementation(torch.Tensor.__setitem__)
_remove_default_implementation(torch.Tensor._autocast_to_full_precision)
_remove_default_implementation(torch.Tensor._autocast_to_reduced_precision)


# Remove all in-place operations on QuantizedTensor. A QuantizedTenosr specific implementation
# can still be registered through the dispatcher.
for attr in dir(torch.Tensor):
    if attr.endswith("_") and not attr.endswith("__"):
        _msg = (
            f"The in-place operation '{attr}' is not implemented for QuantizedTensor. A user "
            f"implementation of {attr} can be registered through the QuantizedTensor "
            "dispatcher system. See the documentation of `fastforward.dispatcher.register` for "
            "more details."
        )
        _remove_default_implementation(getattr(torch.Tensor, attr), msg=_msg)


_EXCLUDE_FROM_DISPATCH_OR_FALLBACK: set[Callable[..., Any]] = set()


def _set_no_dispatch(attr_name: str, silent: bool = False) -> None:
    """
    Register `attr_name` as 'no dispatch' attribute on QuantizedTensor.

    This means that the default torch.Tensor implementation is used. This
    ensures that no default dequantization happens.

    attr_name is provided as a string instead of a function reference for forward
    compatibility.
    """
    try:
        attr = getattr(torch.Tensor, attr_name)
    except AttributeError:
        if not silent:
            warnings.warn(
                f"Tried to set the atribute '{attr_name}' as a no dispatch attribute for "
                f"QuantizedTensor, but '{attr_name}' is not an attribute of torch.Tensor"
            )
        return

    get_set_descriptor_type = type(torch.Tensor.grad)
    if isinstance(attr, get_set_descriptor_type):
        _EXCLUDE_FROM_DISPATCH_OR_FALLBACK.add(attr.__get__)
        _EXCLUDE_FROM_DISPATCH_OR_FALLBACK.add(attr.__set__)
    else:
        _EXCLUDE_FROM_DISPATCH_OR_FALLBACK.add(attr)


_set_no_dispatch("__cuda_array_interface__")
_set_no_dispatch("__repr__")
_set_no_dispatch("__setstate__")
_set_no_dispatch("_backward_hooks")
_set_no_dispatch("_base")
_set_no_dispatch("_cdata")
_set_no_dispatch("_grad")
_set_no_dispatch("_grad_fn")
_set_no_dispatch("_indices")
_set_no_dispatch("_is_view")
_set_no_dispatch("_nested_tensor_size")
_set_no_dispatch("_nested_tensor_strides")
_set_no_dispatch("_version")
_set_no_dispatch("as_subclass")
_set_no_dispatch("backward")
_set_no_dispatch("data_ptr")
_set_no_dispatch("dim")
_set_no_dispatch("ndim")
_set_no_dispatch("dtype")
_set_no_dispatch("get_device")
_set_no_dispatch("device")
_set_no_dispatch("grad")
_set_no_dispatch("grad_fn")
_set_no_dispatch("has_names")
_set_no_dispatch("indices")
_set_no_dispatch("indices")
_set_no_dispatch("layout")
_set_no_dispatch("name")
_set_no_dispatch("names")
_set_no_dispatch("ndimension")
_set_no_dispatch("nelement")
_set_no_dispatch("numel")
_set_no_dispatch("output_nr")
_set_no_dispatch("pin_memory")
_set_no_dispatch("record_stream")
_set_no_dispatch("refine_names")
_set_no_dispatch("register_hook")
_set_no_dispatch("requires_grad")
_set_no_dispatch("requires_grad_")
_set_no_dispatch("retain_grad")
_set_no_dispatch("shape")
_set_no_dispatch("size")
_set_no_dispatch("sparse_dim")
_set_no_dispatch("sparse_mask")
_set_no_dispatch("storage")
_set_no_dispatch("storage_offset")
_set_no_dispatch("storage_type")
_set_no_dispatch("__dir__")


# fmt: off
_no_dispatch_attributes = [
    "is_coalesced", "is_complex", "is_conj", "is_contiguous", "is_cpu", "is_cuda",
    "is_distributed", "is_floating_point", "is_inference", "is_ipu", "is_leaf", "is_meta",
    "is_mkldnn", "is_mps", "is_neg", "is_nested", "is_nonzero", "is_ort", "is_pinned",
    "is_same_size", "is_set_to", "is_shared", "is_signed", "is_sparse", "is_sparse_csr",
    "is_vulkan", "is_xpu",
]
# fmt: on
for attr in _no_dispatch_attributes:
    _set_no_dispatch(attr)


@overload
def apply_and_reattach(
    func: Callable[[torch.Tensor], torch.Tensor], quantized: None = None
) -> Callable[["QuantizedTensor"], "QuantizedTensor"]: ...


@overload
def apply_and_reattach(
    func: Callable[[torch.Tensor], torch.Tensor], quantized: "QuantizedTensor"
) -> "QuantizedTensor": ...


def apply_and_reattach(
    func: Callable[[torch.Tensor], torch.Tensor], quantized: Optional["QuantizedTensor"] = None
) -> Callable[["QuantizedTensor"], "QuantizedTensor"] | "QuantizedTensor":
    """
    Apply func to quantized as if it is a normal tensor.

    Rewrap the result as quantized tensor using the same quantization metadata.
    If `quantized` is not provided, a callable that accepts a quantized_tensor
    and applies func is returned. This way, apply_and_reattach can also be used
    as a decorator.

    Args:
        func: Function that takes a tensor and produces a tensor
        quantized: The tensor to apply func to

    Returns:
        `QuantizedTensor` Or Callable: If quantized is provided, return the
            result of applying func to quantized.raw_data as a quantized tensor.
            If not, return a function that takes a function and applies func
            when called.
    """
    if quantized is not None:
        return quantized._quantization_context.attach(func(quantized.raw_data))

    @functools.wraps(func)
    def wrapper(quantized: "QuantizedTensor") -> "QuantizedTensor":
        return apply_and_reattach(func, quantized)

    return wrapper


def _rebuild_quantized_tensor(
    data: torch.Tensor, quantization_context: "QuantizationContext[QuantizationParameters]"
) -> "QuantizedTensor":
    return QuantizedTensor(data, quantization_context)


@contextlib.contextmanager
def _ignore_quantized_tensor_warnings() -> Generator[None, None, None]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module=__name__)
        yield


class QuantizedTensor(torch.Tensor):
    """
    Tensor that holds quantized data that can be dequantized.

    In general, a quantized tensor has an integer representation and associated
    parameters. Operations can either dequantize to obtain a 'real-valued'
    representation of the data or act on the integer data directly.

    Args:
        data: Raw quantized data, e.g., the integer repsentation obtained
            from `quantization_context.quantization_fn.quantize`.
        quantization_context: The quantization context used to
            produce data.
    """

    def __new__(cls, data: torch.Tensor, *args: Any, **kwargs: Any) -> "QuantizedTensor":
        """
        Create a new quantized tensor.
        """
        return data.as_subclass(cls)

    def __init__(
        self,
        data: torch.Tensor,
        quantization_context: "QuantizationContext[QuantizationParameters]",
    ):
        del data
        super().__init__()
        self._quantization_context = quantization_context

    # pylint: disable=invalid-name, line-too-long
    # fmt: off
    @overload # type: ignore[override]
    def to(self, dtype: torch.dtype, non_blocking: bool = False, copy: bool = False) -> Tensor: ...
    @overload
    def to( self: _TensorT, device: Optional[torch.device | int | str] = None, dtype: None = None, non_blocking: bool = False, copy: bool = False,) -> _TensorT: ...
    @overload
    def to( self, device: Optional[torch.device | int | str] = None, dtype: Optional[torch.dtype] = None, non_blocking: bool = False, copy: bool = False,) -> torch.Tensor: ...
    @overload
    def to(self, other: torch.Tensor, non_blocking: bool = False, copy: bool = False) -> Tensor: ...
    # fmt: on
    # pylint: disable=invalid-name
    # pylint: enable=invalid-name, line-too-long
    def to(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Perform tensor dtype and/or device conversions.

        When `QuantizedTensor` is moved to a different device, the associated
        quantization parameter tensors are moved to the same device. This
        results in an extra copy of the quantization parameters.

        See PyTorch's documentation for arguments.
        """
        if (len(args) > 0 and isinstance(args[0], torch.Tensor)) or "other" in kwargs:
            raise ValueError(f"{type(self).__name__}.to(other: Tensor, ...) is not supported")

        to_args = parse_to_args(*args, **kwargs)
        device, dtype_, non_blocking, memory_format = to_args
        dtype = cast(torch.dtype | None, dtype_)

        if dtype is not None:
            dequantized = self.dequantize()
            dequantized = dequantized.to(
                device=device, dtype=dtype, non_blocking=non_blocking, memory_format=memory_format
            )
            return dequantized

        with DisableTorchFunctionSubclass():
            moved_qtensor = super().to(
                device=device, non_blocking=non_blocking, memory_format=memory_format
            )

        return type(self)(moved_qtensor, quantization_context=self._quantization_context.to(device))

    def __deepcopy__(self, memo: dict[Any, Any]) -> Self:
        if not self.is_leaf:
            raise RuntimeError(
                "Only Tensors created explicitly by the user "
                "(graph leaves) support the deepcopy protocol at the moment"
            )
        quantization_context = copy.deepcopy(self._quantization_context, memo)
        return type(self)(copy.deepcopy(self.raw_data.detach(), memo), quantization_context)

    def __reduce_ex__(self, proto: int) -> Any:  # type: ignore[override]
        raw_data = self.raw_data.detach()
        return _rebuild_quantized_tensor, (raw_data, self._quantization_context)

    def cuda(  # type: ignore[override]
        self: _TensorT, device: torch.device | int | str | None = None, non_blocking: bool = False
    ) -> _TensorT:
        """Move `QuantizedTensor` and associated parameters to GPU."""
        return self.to(device=device or "cuda", non_blocking=non_blocking)

    def cpu(self: _TensorT) -> _TensorT:  # type: ignore[override]
        """Move `QuantizedTensor` and associated parameters to CPU."""
        return self.to("cpu")

    def dequantize(self) -> torch.Tensor:
        """
        Dequantize and return real-valued torch.Tensor.
        """
        return self._quantization_context.quantization_fn.dequantize(
            self.raw_data, self.quant_args()
        )

    def clone(self) -> "QuantizedTensor":  # type: ignore[override]
        """
        Clone tensor.

        This makes a copy of the tensor and associated quantization
        parameter tensors

        Returns:
            QuantizedTensor: Quantized tensor that holds the same data but copied.

        Note:
            The non-tensor quantization parameters are **not** copied.
        """
        quant_ctx = self._quantization_context.clone_parameters()
        with DisableTorchFunctionSubclass():
            cloned_tensor = super().clone()
        return quant_ctx.attach(cloned_tensor)

    def detach(self) -> "QuantizedTensor":  # type: ignore[override]
        """
        Detach tensor.

        This returns a ternsor that alliases this tensor, but is
        detached from the autograd graph.
        """
        quant_func = self._quantization_context.detach_parameters()
        with DisableTorchFunctionSubclass():
            detached_tensor = super().detach()
        return quant_func.attach(detached_tensor)

    @property
    def raw_data(self) -> torch.Tensor:
        """
        Return raw data representation.

        Returns:
            Torch.Tensor: the raw_data as a normal tensor.
        """
        return self.as_subclass(torch.Tensor)

    def quant_args(self) -> "QuantizationParameters":
        """
        Return quantization arguments.

        Returns:
            QuantArgs: Arguments used to quantize self.

        Note:
            `QuantArgs` can be mutated without it having an effect on self, but
            all reference types on `QuantArgs` are shared and mutation will
            have side effects.
        """
        return self._quantization_context.quantization_params

    @property
    def quantization_context(self) -> "QuantizationContext[QuantizationParameters]":
        return self._quantization_context

    @property
    def quant_func(self) -> type["QuantizationFunction[QuantizationParameters]"]:
        """
        Return the associated quantization function.

        Returns:
            BaseQuantizationFunction: The `QuantizationFunction` implementation used
                to quantize self.

        Note:
            this only returns the implementation and no arguments
            are bound to it. To obtain a bound quantization function use:

               quantized_tensor.quant_func.bind(**quantized_tensor.quant_args())
        """
        return self._quantization_context.quantization_fn

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        kwargs = kwargs or {}
        with DisableTorchFunctionSubclass():
            if func in _EXCLUDE_FROM_DISPATCH_OR_FALLBACK:
                return func(*args, **kwargs)

            # First try to find a kernel through the dispatcher
            op_name = _torch_function_to_op_name(func)

            if op_name:
                kernel = dispatch(op_name, *args, **kwargs)
                if kernel:
                    return kernel(*args, **kwargs)

            # If no kernel is found, use the dequantization fallback
            return _dequantization_fallback(func, *args, **kwargs)

    def __repr__(self, **kwargs: Any) -> str:  # type: ignore[override]
        with DisableTorchFunction():
            with _ignore_quantized_tensor_warnings():
                repr_str = super().__repr__(**kwargs)
        indent = len(type(self).__name__) + 1
        quantizer = self.quant_func.__name__
        return (
            f"{repr_str[:-1]},\n{' '*indent}quantizer={quantizer}, " f"{self.quant_args():short})"
        )

    # fmt: off
    # Remove any implementation of in-place operators. This doesn't prohibit
    # the use of in-place operators, but there is no in-place implementation.
    # As such, a += b is the same as a = a + b and any dequantization will
    # occur. This is required because the result of an operator generally does
    # not adhere to a quantization grid.
    def __iadd__(self, other: Any) -> torch.Tensor: return NotImplemented
    def __isub__(self, other: Any) -> torch.Tensor: return NotImplemented
    def __imul__(self, other: Any) -> torch.Tensor: return NotImplemented
    def __imatmul__(self, other: Any) -> torch.Tensor: return NotImplemented
    def __itruediv__(self, other: Any) -> torch.Tensor: return NotImplemented # type: ignore[override]
    def __ifloordiv__(self, other: Any) -> torch.Tensor: return NotImplemented # type: ignore[misc]
    def __imod__(self, other: Any) -> torch.Tensor: return NotImplemented
    def __ilshift__(self, other: Any) -> torch.Tensor: return NotImplemented # type: ignore[misc]
    def __irshift__(self, other: Any) -> torch.Tensor: return NotImplemented # type: ignore[misc]
    def __iand__(self, other: Any) -> torch.Tensor: return NotImplemented # type: ignore[misc]
    def __ixor__(self, other: Any) -> torch.Tensor: return NotImplemented # type: ignore[misc]
    def __ior__(self, other: Any) -> torch.Tensor: return NotImplemented # type: ignore[misc]
    def __ipow__(self, other: Any, modulo: Optional[int]=None) -> torch.Tensor: # type: ignore[misc]
        return NotImplemented
    # fmt: on

    def int_repr(self) -> torch.Tensor:
        """
        Return the integer (or quantized) data representation underlying this quantized tensor.
        """
        return self.raw_data

    def contiguous(self, memory_format: Any = torch.contiguous_format) -> "QuantizedTensor":
        """Return a tensor with contiguous data, this includes quantization parameters."""
        raw_data = self.raw_data.contiguous(memory_format=memory_format)
        quantization_context = self._quantization_context.contiguous_parameters()
        if raw_data is self.raw_data and quantization_context is self._quantization_context:
            return self
        return quantization_context.attach(raw_data)

    @property
    def is_quantized(self) -> bool:
        """
        Always returns False for a `QuantizedTensor` and should not be used to identify QuantizedTensors.

        This property evaluates to False for QuantizedTensors as it is otherwise
        identified as a PyTorch native Quantized tensor. The recommended
        approach to test for QuantizedTensors is using isinstance.
        """
        warnings.warn(
            "QuantizedTensor.is_quantized is used. This property evaluates to False for "
            "QuantizedTensors as it is otherwise identified as a PyTorch native Quantized "
            "tensor. The recommended approach to test for QuantizedTensor's is using isinstance."
        )
        return False

    @is_quantized.setter
    def is_quantized(self, value: bool) -> None:
        raise AttributeError("AttributeError: can't set attribute 'is_quantized'")


def _dequantize_qtensor(_object: _T) -> _T | torch.Tensor:
    if isinstance(_object, QuantizedTensor):
        return _object.dequantize()
    else:
        return _object


def _dequantization_fallback(func: Callable[..., _U], *args: Any, **kwargs: Any) -> _U:
    if fastforward.get_strict_quantization():
        raise QuantizationError(
            f"{func} was called while `fastforward.get_strict_quantization() == True`. "
            f"Because of this, implicit dequantization is not allowed. "
            f"Implicit dequantization occurs when a non-quantized operator is applied to one or more quantized tensors. "
            f"This error can be resolved by changing the global config, "
            f"performing the error in a temporary config context, "
            f"by explicitly dequantizing the quantized tensors before the operations, "
            f"or by registering a quantized operator that handles this specific case."
        )
    with DisableTorchFunctionSubclass():
        args = optree.tree_map(_dequantize_qtensor, args)  # type: ignore[arg-type, assignment]
        kwargs = optree.tree_map(_dequantize_qtensor, kwargs or {})  # type: ignore[arg-type, assignment]
    return func(*args, **kwargs)


# Define overloads for view and view_as for the 'same shape' case. These may get called
# by torch.autograd.Function if any of the return tensors aliases one of the input
# tensors. This implementation works for any quantizer.
@Predicate
def _view_predicate(self: QuantizedTensor, *shape: int | Sequence[int] | torch.Size) -> bool:
    if len(shape) == 1 and (isinstance(shape[0], torch.Size) or isinstance(shape[0], Sequence)):
        shape = torch.Size(shape[0])
    else:
        shape = torch.Size(cast(tuple[int], shape))
    return shape == self.shape


@Predicate
def _view_as_predicate(self: QuantizedTensor, other: torch.Tensor) -> bool:
    if not isinstance(other, torch.Tensor):
        return False  # type: ignore[unreachable]
    return other.shape == self.shape


@register("view", _view_predicate, priority=DispatcherPriority.FALLBACK)
def _same_shape_view(
    self: QuantizedTensor, *shape: int | Sequence[int] | torch.Size
) -> torch.Tensor:
    if len(shape) == 1 and (isinstance(shape[0], torch.Size) or isinstance(shape[0], Sequence)):
        shape = torch.Size(shape[0])
    else:
        shape = torch.Size(cast(tuple[int], shape))
    return apply_and_reattach(lambda x: x.view(*shape), self)


@register("view_as", _view_as_predicate, priority=DispatcherPriority.FALLBACK)
def _same_shape_view_as(self: QuantizedTensor, other: torch.Tensor) -> torch.Tensor:
    return apply_and_reattach(lambda x: x.view_as(other), self)
