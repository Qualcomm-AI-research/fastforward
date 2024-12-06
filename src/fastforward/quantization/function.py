# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

"""
`QuantizationFunction` implement the logic for quantization and dequantization.

In neural network quantization, the quantization operation is often decomposed
into a `quantize` and `dequantize` operation. The first maps a real valued input
to an integer, whereas the latter is a mapping from integers back to the reals.
When combined (i.e., dequantize o quantize) we obtain an R -> R quantization
mapping.

In FastForward, a quantized tensor associates integer data obtained from the
`quantize` operation with the quantization parameters that specify the
quantization function. As such, from a representation point of view, a
quantized tensor represents the same real-valued data as a dequantized tensor.
In other words, the `dequantize` operation on a quantized tensor is only a
change in representation, not in value.

The assumption in the previous paragraph has two consequences for autograd:

    1. The creation of a `QuantizedTensor` is an implicit dequantize. Hence,
       the backward pass mus take this into account.
    2. Dequantize is an identity operation through the lens of autograd.

This module offers abstractions to implement the quantize/dequantize operation
in a way that follows the assumptions. There are two options:

    1. `QuantizationFunction`: implement `quantize` and `dequantize` operations
       seperately. The resulting quantization function will perform the
       `quantize` function during the forward pass. During the backward pass
       both the forward and backward for `dequantize` is performed to ensure the
       correct gradient is computed. This may lead to some extra overhead when
       the tensor was already dequantized during the forward pass. The benefit
       of this option is that only `quantize` and `dequantize` have to be
       implemented, the backward pass is implicit through standard autograd.

    2. `QuantizationAutogradFunction`: next to `quantize` and `dequantize`, a
       `quant_dequant_backward` must be implemented. This latter implements the
       backward pass for `quantize` and `dequantize` jointly. Note that, since
       only `quantize` is guaranteed to have been called in the forward pass,
       any tensor that must be saved for the backward pass, must be saved during
       the `quantize` operation. The `quantize` and `quant_dequant_backward`
       implementation are used to create a standard `torch.autograd.Function`
       subclass. As such, next to the quantization arguments, they must accept
       a context argument. The main benefit of this option is that, depending
       on the implementation, it may have some performance benefits over the
       implicit gradient of the other option.

For an example implementation, see `fastforward.quantization.affine`.

"""

import functools
import inspect

from typing import Any, Callable, ClassVar, Iterator, Mapping, TypeVar

import torch

from typing_extensions import Self, TypeVarTuple, Unpack

import fastforward

from fastforward.quantization import ste
from fastforward.quantized_tensor import QuantizedTensor

T = TypeVar("T")
Ts = TypeVarTuple("Ts")


def _try_apply_tensor(
    func: Callable[[torch.Tensor], torch.Tensor], *maybe_tensors: Unpack[Ts]
) -> tuple[Unpack[Ts]]:
    return tuple(func(v) if isinstance(v, torch.Tensor) else v for v in maybe_tensors)  # type: ignore[return-value]


class _ImplicitDequantize(torch.autograd.Function):
    """
    Implicitly call dequantize to rewrite autograd graph.

    A Quantized tensor represent a dequantized tensor. However, it does not
    explcitly compute a dequantized value. _ImplicitDequantize is used to
    'record' the dequantize operation in the forward pass and execute the
    backward pass when gradient are computed. Since the forward pass may store
    data that is required for the backward pass, during the backward pass the
    dequantize function is executed.

    _ImplicitDequantize is an autograd function that takes the same arguments as
    quantize/dequantize together with a reference to the dequantize function.
    """

    apply: Callable[..., torch.Tensor]

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        dequantize_fn: Callable[..., torch.Tensor],
        *args: Any,
    ) -> torch.Tensor:
        """
        Return first element of arg and registered dequantize_fn as part of graph.

        Args:
            ctx: Torch autograd context
            dequantize_fn: A function that accepts a tensor as input together with
                all further arguments passed to this function
        """
        # Collect all tensor arguments and store together with the indexes
        # in the args tuple.
        tensor_idxs, tensors = zip(
            *[arg for arg in enumerate(args) if isinstance(arg[1], torch.Tensor)]
        )
        ctx.tensor_idxs = tensor_idxs  # type: ignore[attr-defined]
        ctx.save_for_backward(*tensors)

        # Store all non-tensor arguments and the dequantize function on
        # the FunctionCtx to use during the backward pass.
        ctx.other_args = [arg for arg in args if not isinstance(arg, torch.Tensor)]  # type: ignore[attr-defined]
        ctx.num_args = len(args) + 1  # type: ignore[attr-defined]
        ctx.dequantize_fn = dequantize_fn  # type: ignore[attr-defined]

        # Return first argument, which is expected to be the 'data tensor'.
        # I.e., the forward pass is a no-op.
        return args[0]  # type: ignore[no-any-return]

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx, grad: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        tensors = ctx.saved_tensors  # type: ignore[attr-defined]
        args = ctx.other_args[:]  # type: ignore[attr-defined]

        # Reconstruct the args tuple and collect all tensors for which
        # gradients must be computed.
        compute_grad_inputs: list[torch.Tensor] = []
        grad_idxs: list[int] = []
        for idx, tensor in zip(ctx.tensor_idxs, tensors):  # type: ignore[attr-defined]
            requires_grad = tensor.requires_grad
            tensor = tensor.detach()
            if requires_grad:
                tensor.requires_grad_()
                compute_grad_inputs.append(tensor)
                grad_idxs.append(idx)
            args.insert(idx, tensor)

        grads = [None] * ctx.num_args  # type: ignore[attr-defined]

        # If no inputs require gradient, return early without computing
        # gradients.
        if len(compute_grad_inputs) == 0:
            return tuple(grads)

        # Execute dequantize function and immediately compute its
        # backward pass for selected input tensors.
        with torch.enable_grad():
            output = ctx.dequantize_fn(*args)  # type: ignore[attr-defined]

        # Create gradient output tuple
        tensor_grads = torch.autograd.grad(output, compute_grad_inputs, grad)
        for idx, grad in zip(grad_idxs, tensor_grads):
            grads[idx] = grad
        return (None,) + tuple(grads)


class BaseQuantizationFunction:
    """
    Base class for Quantization functions.

    Users should not directly subclass BaseQuantizationFunction but subclass
    QuantizationFunction or QuantizationAutogradFunction instead.
    """

    def __init__(self) -> None:
        raise

    @staticmethod
    def dequantize(data: torch.Tensor, *args: Any) -> torch.Tensor:
        """
        Dequantize data using args.

        Data represents quantized data. The representation of this data (i.e., datatype) depends
        on the the implementation of `_quantize_forward`. All arguments that are provided
        the the quantize operations are provided to the `dequantize` operation.

        Args:
            data: Raw quantized data
            args: All quantization arguments
        """
        raise NotImplementedError(
            "dequantize must be overriden in BaseQuantizationFunction subclass"
        )

    @classmethod
    def _quantize_forward(
        cls, data: torch.Tensor, *args: Any
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        raise NotImplementedError(
            "_quantize_forward must be overriden in BaseQuantizationFunction subclass"
        )

    @classmethod
    def apply(cls, data: torch.Tensor, *args: Any, **kwargs: Any) -> "QuantizedTensor":
        """
        Apply Quantization function to data with given arguments.

        Args:
            data: Data to quantize
            args: Arguments used in quantize/dequantize function.
            kwargs: Keyword arguments used in quantize/dequantize function.

        Returns:
            QuantizedTensor: Quantized data
        """
        return cls.bind(*args, **kwargs).quantize(data)

    @classmethod
    def bind(cls, *args: Any, **kwargs: Any) -> "BoundQuantizationFunction":
        """
        Construct a `BoundQuantizationFunction` for this QuantizationFuncion.
        """
        try:
            args = cls._destructure_args(*args, **kwargs)
        except TypeError as e:
            # Do some gymnastics to make it seem like error was raised here
            # instead of in destructure_args.
            raise TypeError(*e.args) from None
        return BoundQuantizationFunction(cls, *args)

    @classmethod
    def attach(cls, data: torch.Tensor, *args: Any, **kwargs: Any) -> "QuantizedTensor":
        """
        Create a `QuantizedTensor` that is associated with this `QuantizationFunction` and given data.
        """
        return cls.bind(*args, **kwargs).attach(data)

    @classmethod
    def _restructure_args(cls, *args: Any) -> dict[str, Any]:
        """
        Convert argument list to key-value mapping.

        Convert argument list to a dictionary with argument name keys and
        corresponding values.

        This function is used to provide a structured argument representation
        on the quantized tensor.

        """
        signature = inspect.signature(cls.dequantize)
        kwargs = signature.bind(None, *args).arguments
        data_arg = next(iter(signature.parameters.keys()))
        del kwargs[data_arg]
        return kwargs

    @classmethod
    def _destructure_args(cls, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        """
        Convert arguments to a sequence.

        Convert argument and keyword arguments to a single argument list
        that corresponds to the order of arguments in the quantize/dequantize
        functions.
        """
        signature = inspect.signature(cls.dequantize)
        # This can be unified to use the signature.bind once issue #165 is resolved.
        if fastforward.get_export_mode():
            kwarg_to_arg = [
                kwargs[parameter] for parameter in signature.parameters if parameter in kwargs
            ]

            return tuple(list(args) + kwarg_to_arg)
        else:
            args = signature.bind(None, *args, **kwargs).args
            return args[1:]


class QuantizationFunction(BaseQuantizationFunction):
    """
    Implementation of BaseQuantizationFunction.

    The backward pass for quantize/dequantize is implicit.

    Note: The arguments of the quantize and dequantize implementation *must* be
          the same. This is not enforced, but will violating this contract will
          result in undefined behaviour.
    """

    @staticmethod
    def quantize(data: torch.Tensor, *args: Any) -> torch.Tensor:
        """
        Quantize data using given arguments.

        The output of this function represents the data of a quantized tensor.
        The arguments provided to this function are associated with the
        `QuantizedTensor` separately.

        Returns:
            torch.Tensor: quantized data
        """
        raise NotImplementedError("quantize must be overridden in QuantizationFunction subclass")

    @staticmethod
    def dequantize(data: torch.Tensor, *args: Any) -> torch.Tensor:
        """
        Dequantize data using given arguments.

        Args:
            data: Data to dequantize. The value of this tensor is
                equal to the output of quantize.

        Returns:
            torch.Tensor: dequantized data
        """
        raise NotImplementedError("dequantize must be overriden in QuantizationFunction subclass")

    @classmethod
    def _quantize_forward(cls, data: torch.Tensor, *args: Any) -> torch.Tensor:
        quantized_data = cls.quantize(data, *args)
        return _ImplicitDequantize.apply(cls.dequantize, quantized_data, *args)


class QuantizationAutogradFunction(BaseQuantizationFunction):
    """
    QuantizationFunction with an explicit autograd implementation.

    The autograd 'step' is implemented for the quantize and dequantize function
    jointly. A torch autograd function is created from the quantize and
    quant_dequant_backward functions. Dequantize is only used if the resulting
    tesnor is explcitly dequantized.

    Note: Dequantize must accept the exact same arguments as quantize, *except*
          ctx. This is not enforced, but will violating this contract will
          result in undefined behaviour.
    """

    _autograd_func: ClassVar[type[torch.autograd.Function]]

    @staticmethod
    def quantize(
        ctx: torch.autograd.function.FunctionCtx, data: torch.Tensor, *args: Any
    ) -> torch.Tensor:
        """
        Quantize data using given arguments.

        The output of this function represents the data of a quantized tensor.
        The argumnets provided to this function are associated with the
        `QuantizedTensor` seperarely.

        Returns:
            ctx: Torch autograd context
            torch.Tensor: quantized data
        """
        raise NotImplementedError("quantize must be overriden in QuantizationFunction subclass")

    @staticmethod
    def dequantize(data: torch.Tensor, *args: Any) -> torch.Tensor:
        """
        Dequantize data using given arguments.

        Args:
            data: Data to dequantize. The value of this tensor is
                equal to the output of quantize.

        Returns:
            torch.Tensor: dequantized data
        """
        raise NotImplementedError("dequantize must be overriden in QuantizationFunction subclass")

    @staticmethod
    def quant_dequant_backward(
        ctx: torch.autograd.function.FunctionCtx, grad: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        """
        Implementation of the backward pass for both quantize and dequantize jointly.

        Accepts exactly one gradient arguments, and returns one
        gradient tensor or None for each argument to quantize.
        """
        raise NotImplementedError(
            "quant_dequant_backward must be overriden in QuantizationFunction subclass"
        )

    def __init_subclass__(cls, *args: Any, **kwargs: Any) -> None:
        super().__init_subclass__(*args, **kwargs)

        # Create an autograd function using quantize and quant_dequant_backward.
        # This function is applied during the forward pass.
        cls._autograd_func = type(
            f"{cls.__name__}Function",
            (torch.autograd.Function,),
            {"forward": cls.quantize, "backward": cls.quant_dequant_backward},
        )

    @classmethod
    def _quantize_forward(cls, data: torch.Tensor, *args: Any) -> torch.Tensor:
        return cls._autograd_func.apply(data, *args)  # type: ignore[no-any-return]


class QuantArgs(Mapping[str, Any]):
    """
    QuantArgs is a str to value mapping for quantization arguments.

    The mapping is mutable and will not affect the quantized tensor or
    BoundQuantizationFunction from which QuantArgs was obtained. However,
    values may be shared with `BoundQuantizationFunction`s or
    `QuantizedTensor`s, hence, changes values inplace may have an effect
    outside of the QuantArgs instance.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__["_kwargs"] = kwargs

    def __iter__(self) -> Iterator[str]:
        yield from self._kwargs.keys()

    def __getattr__(self, name: str) -> Any:
        try:
            return self._kwargs[name]
        except KeyError:
            raise AttributeError(f"'QuantArgs' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in self._kwargs:
            raise AttributeError(f"{type(self).__name__} has not attribute '{name}'")
        self._kwargs[name] = value

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def __len__(self) -> int:
        return len(self._kwargs)

    def __format__(self, format_spec: str) -> str:
        match format_spec:
            case "short":
                return self._arglist_repr()
            case _:
                return repr(self)

    def _arglist_repr(self) -> str:
        return ", ".join([f"{k}={v}" for k, v in self._kwargs.items()])

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._arglist_repr()})"


class BoundQuantizationFunction:
    """
    Container that associated a QuantizationFunction with arguments.

    Args:
        quant_func: The quantization function to which
            arguments are bound
        args: Arguments that will be associated with quant_func
    """

    quant_func: type[BaseQuantizationFunction]
    args: tuple[Any, ...]

    def __init__(self, quant_func: type[BaseQuantizationFunction], *args: Any):
        self.quant_func = quant_func
        self.args = args

    def attach(self, data: torch.Tensor, *dynamic_args: Any) -> "QuantizedTensor":
        """
        Create a quantized tensor that associates this `BoundQuantizationFunction` with data.

        Args:
            data: The raw data for the quantized tensor
        """
        if fastforward.get_export_mode():
            # The QuantizedTensor can be used once issue #166 is resolved (waiting for changes
            # on the torch side).
            return self.quant_func.dequantize(data, *self.args)  # type: ignore[return-value]
        else:
            if dynamic_args:
                return QuantizedTensor(data, type(self)(self.quant_func, *dynamic_args))
            else:
                return QuantizedTensor(data, self)

    def rebind(self, **kwargs: Any) -> Self:
        """
        Create a new `BoundQuantizationFunction` and overwrite the newly provided arguments.

        All unprovided arguments are left unchanged. An error is raised if an
        argument is provided that is unknown to the QuantizationFunction.

        Args:
            kwargs: Any new quantization argument

        Returns:
            BoundQuantizationFunction: The new bound function
        """
        current_kwargs = self.quant_func._restructure_args(*self.args)
        for k, v in kwargs.items():
            if k not in current_kwargs:
                raise TypeError(f"{type(self).__name__}.rebind() got an unexpected keyword '{k}'")
            current_kwargs[k] = v
        BoundFuncType = type(self)
        new_args = self.quant_func._destructure_args(**current_kwargs)
        return BoundFuncType(self.quant_func, *new_args)

    def rebind_and_attach(
        self, quantized_tensor: "QuantizedTensor", **kwargs: Any
    ) -> "QuantizedTensor":
        """
        Rebind quantization function and immediatly attach to an existing quantized tensor.

        This will raise an error if `quantized_tensor` is not already quantized
        by the same QuantizationFunction.

        Args:
            quantized_tensor: Existing quantized tensor that is
                reassoacited with an updated QuantizationFunction.
            kwargs: Any new quantization argument

        Returns:
            QuantizedTensor: The newly created quantized tensor that is quantized according
                to the newly created BoundQuantizationFunction.

        Note:
            This only changes the quantization parameters and not the quantized
            data. Moreover, the data of the output quantized tensor is shared
            with the input.
        """
        if quantized_tensor.quant_func != self.quant_func:
            raise ValueError(
                f"Cannot rebind to a quantized tensor that is not originally quantized by "
                f"'{self.quant_func.__name__}'. Found a tensor that is quantized by "
                f"'{quantized_tensor.quant_func}.'"
            )
        new_state = self.rebind(**kwargs)
        return new_state.attach(quantized_tensor.raw_data)

    def clone(self) -> Self:
        """
        Create a new `BoundQuantizationFunction` in which all tensor arguments are cloned.

        Returns:
            BoundQuantizationFunction: Bound function that is equivalent to self, but
            all tensor arguments are cloned.

        Note:
            All non-Tensor arguments are left unchanged and uncopied. Hence, any
            argument that references an object is shared between the output and self.
        """
        cloned_args = _try_apply_tensor(torch.clone, *self.args)
        return type(self)(self.quant_func, *cloned_args)

    def contiguous(self) -> Self:
        """
        Convert any non-contiguous tensor parameter into a contiguous parameter.

        This may result in a memory copy and the creation of new
        BoundQuantizationFunction. If all tensor parameters are already
        contiguous, do nothing and return self.
        """
        args_contiguous = True
        for arg in self.args:
            if isinstance(arg, torch.Tensor) and not arg.is_contiguous():
                args_contiguous = False
                break

        if args_contiguous:
            return self
        else:
            contiguous_args = _try_apply_tensor(lambda arg: arg.contiguous(), *self.args)
            return type(self)(self.quant_func, *contiguous_args)

    def detach_arguments(self) -> Self:
        """
        Create a new `BoundQuantizationFunction` in which all tensor arguments are detached.

        Returns:
            BoundQuantizationFunction: Bound function that is equivalent to self, but
            all tensor arguments are detached.
        """
        detached_args = _try_apply_tensor(torch.detach, *self.args)
        return type(self)(self.quant_func, *detached_args)

    def to(self, device: torch.device | str) -> Self:
        """
        Create a new `BoundQuantizationFunction` in which all tensor arguments are moved to `device`.

        Returns:
            quantization function: bound function that is equivalent to self, but
                all tensor arguments are moved to `device`.
        """
        moved_args = _try_apply_tensor(
            functools.partial(torch.Tensor.to, device=device), *self.args
        )
        return type(self)(self.quant_func, *moved_args)

    def arguments(self) -> QuantArgs:
        """
        Returns the arguments bound to self.quant_func.

        Returns:
            QuantArgs: Arguments bound to this function.

        Note:
            `QuantArgs` can be mutated without it having an effect on self, but
            all reference types on `QuantArgs` are shared and mutation will
            have side effects.
        """
        return QuantArgs(**self.quant_func._restructure_args(*self.args))

    def dequantize(self, quantized: "QuantizedTensor") -> torch.Tensor:
        """
        Helper function to apply dequantize implentation of the associated QuantizationFunction.

        Note: A `QuantizedTensor` and its dequantized floating point value
            represent the same real-valued data. As such, from the autograd
            perspective, the Jocabion of the dequantize operation is an identity
            matrix. This does not mean that dequantize is not ignored during the
            backward pass, but its backward pass is executed as part of the
            'quantize' backward pass. See the docstring in
            `fastforward.quantization.function` for more information.
        """
        return ste.ste(self.quant_func.dequantize)(quantized.raw_data, *self.args)

    def quantize(self, data: torch.Tensor) -> "QuantizedTensor":
        """
        Quantize data and return a QuantizedTensor.

        Args:
            data: Data to quantize

        Returns:
            QuantizedTensor
        """
        quantized_data = self.quant_func._quantize_forward(data, *self.args)
        if isinstance(quantized_data, tuple):
            quant_data, *dynamic_args = quantized_data
        else:
            quant_data, dynamic_args = quantized_data, []
        return self.attach(quant_data, *dynamic_args)

    def __call__(self, data: torch.Tensor) -> "QuantizedTensor":
        """
        Quantize data and return a QuantizedTensor.

        Args:
            data: Data to quantize

        Returns:
            QuantizedTensor
        """
        return self.quantize(data)

    def __format__(self, format_spec: str) -> str:
        match format_spec:
            case "short":
                return self._arglist_repr()
            case _:
                return repr(self)

    def _arglist_repr(self) -> str:
        return f"quantizer={self.quant_func.__name__}, {self.arguments():short}"

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._arglist_repr()})"
