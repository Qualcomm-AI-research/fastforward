# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import collections

from types import SimpleNamespace
from typing import Any, Callable, Iterator

import torch

from typing_extensions import Self, deprecated
from typing_extensions import override as typing_override

from fastforward import forward_override as override


class Tag:
    """Tags are symbol-like objects used to communicate features of a quantizer.

    A tag can be hierarchical using '/' to separate the tag levels. A
    hierarchical tag can also be constructed using `tag / tag`, which produces
    a new tag.
    """

    _tags: dict[str, "Tag"] = {}
    _symbol: str

    def __new__(cls, symbol: str | Self) -> "Tag":
        """Create a new Tag instance or return an existing one if it already exists.

        Args:
            symbol: The symbol representing the tag.

        Returns:
            Tag: The Tag instance.
        """
        if isinstance(symbol, Tag):
            return symbol
        if symbol not in cls._tags:
            tag = super().__new__(cls)
            tag._symbol = symbol
            cls._tags[symbol] = tag
        return cls._tags[symbol]

    def __deepcopy__(self, memo: dict[Any, Any]) -> Self:
        # Since there can only exist a single tag with the same name, we do not
        # deepcopy here.
        return self

    def __copy__(self) -> Self:
        # Since there can only exist a single tag with the same name, we do not
        # copy here.
        return self

    @typing_override
    def __str__(self) -> str:
        return f"#{self._symbol}"

    @typing_override
    def __repr__(self) -> str:
        return f"{self._symbol}"

    def hierarchy(self) -> Iterator[Self]:
        """Return all tags in the hierarchy.

        For example, if the tag is
        `first/second/third`, this function will yield a tag for `first`,
        `first/second`, and `first/second/third`.

        Returns:
            An iterator over the tags in the hierarchy.
        """
        tag_fragments = self._symbol.split("/")
        for i in range(len(tag_fragments), 0, -1):
            yield type(self)("/".join(tag_fragments[0:i]))

    def __truediv__(self, rhs: str | Self) -> Self:
        if isinstance(rhs, Tag):
            rhs = rhs._symbol
        if isinstance(rhs, str):
            return type(self)(f"{self._symbol}/{rhs}")
        return NotImplemented  # type: ignore[unreachable]

    def __rtruediv__(self, lhs: str) -> Self:
        if isinstance(lhs, str):
            return type(self)(lhs) / self
        return NotImplemented  # type: ignore[unreachable]


# Define a set of default tags
_parameter_quantizer = Tag("parameter")
_activation_quantizer = Tag("activation")
_weight_quantizer = _parameter_quantizer / "weight"
_bias_quantizer = _parameter_quantizer / "bias"
_input_quantizer = _activation_quantizer / "input"
_output_quantizer = _activation_quantizer / "output"

# Collect default tags in namespace for easy imports
default_tags = SimpleNamespace(
    parameter_quantizer=_parameter_quantizer,
    activation_quantizer=_activation_quantizer,
    weight_quantizer=_weight_quantizer,
    bias_quantizer=_bias_quantizer,
    input_quantizer=_input_quantizer,
    output_quantizer=_output_quantizer,
)


class _TagAttribute:
    """Descriptor class for tag attributes in QuantizerMetadata."""

    def __init__(self, tag: str | Tag) -> None:
        """Initialize the _TagAttribute with a tag.

        Args:
            tag: The tag associated with this attribute.
        """
        self._tag = Tag(tag)

    def __get__(self, instance: "QuantizerMetadata", owner: Any = None) -> bool:
        """Get the value of the attribute for an instance of QuantizerMetadata.

        Args:
            instance: The instance of QuantizerMetadata.
            owner: The owner class.

        Returns:
            True if the tag part of the metadata, False otherwise.
        """
        return self._tag in instance


class QuantizerMetadata:
    """Metadata class for quantizers, holding tags and additional attributes.

    Args:
        tags: Tags to be added to the metadata.
        weight_quantizer: Whether to add the weight quantizer tag.
        bias_quantizer: Whether to add the bias quantizer tag.
        input_quantizer: Whether to add the input quantizer tag.
        output_quantizer: Whether to add the output quantizer tag.
        shape: The shape attribute.
        kwargs: Additional attributes.
    """

    parameter_quantizer = _TagAttribute(_parameter_quantizer)
    weight_quantizer = _TagAttribute(_weight_quantizer)
    bias_quantizer = _TagAttribute(_bias_quantizer)
    input_quantizer = _TagAttribute(_input_quantizer)
    activation_quantizer = _TagAttribute(_activation_quantizer)
    output_quantizer = _TagAttribute(_output_quantizer)

    def __init__(
        self,
        *tags: str | Tag,
        weight_quantizer: bool = False,
        bias_quantizer: bool = False,
        input_quantizer: bool = False,
        output_quantizer: bool = False,
        shape: tuple[int, ...] | torch.Size | None = None,
        **kwargs: Any,
    ) -> None:
        self._tags: set[Tag] = set()
        self._kwargs = kwargs
        self._kwargs["shape"] = shape

        for tag in tags:
            self.add_tag(tag)

        if weight_quantizer:
            self.add_tag(default_tags.weight_quantizer)
        if bias_quantizer:
            self.add_tag(default_tags.bias_quantizer)
        if input_quantizer:
            self.add_tag(default_tags.input_quantizer)
        if output_quantizer:
            self.add_tag(default_tags.output_quantizer)

    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def add_tag(self, tag: Tag | str) -> None:
        """Add a tag to the metadata.

        Args:
            tag: The tag to be added.
        """
        for t in Tag(tag).hierarchy():
            self._tags.add(t)

    def __repr__(self) -> str:
        kwargs = ", ".join(f"{k}={v}" for k, v in self._kwargs.items())
        return f"{type(self).__name__}(tags={self._tags}, {kwargs})"

    def __contains__(self, tag: str | Tag) -> bool:
        return Tag(tag) in self._tags

    def __getattr__(self, key: str) -> Any:
        if key in self._kwargs:
            return self._kwargs[key]
        return super().__getattribute__(key)

    @property
    def shape(self) -> tuple[int, ...] | torch.Size | None:
        """Return the shape metadata, if set. Otherwise return None.

        Returns:
            tuple[int, ...] | torch.Size | None: The shape attribute.
        """
        return self._kwargs.get("shape")

    def is_extension(self, other: Self) -> bool:
        """Returns True if other is an extension of self, False otherwise.

        metadata B is an extension of metadata A if B's tag set is a superset
        of A's tag set and all attributes of A match with the corresponding
        attributes of B. That is, B may have more tags and/or attributes, but
        they can not conflict with those of A.

        Args:
            other: Metadata to check if it is an extension of self

        Returns:
            Boolean indicating if other is an extension of self
        """
        if not self._tags.issubset(other._tags):
            return False

        for key, value in self._kwargs.items():
            if key == "shape" and value is None:
                continue
            if other._kwargs[key] != value:
                return False

        return True

    @deprecated(
        "to_stub will be removed in a future release. "
        "Use QuantizerStub and pass metadata arguments directly instead"
    )
    def to_stub(self) -> "QuantizerStub":
        """Deprecated."""
        return QuantizerStub(_metadata=self)


class Quantizer(torch.nn.Module):
    """Base class for Quantizers."""

    _quantizer_overrides: dict[int, override.OverrideFn[torch.Tensor]]
    quant_metadata: QuantizerMetadata | None

    def __init__(self) -> None:
        """Initialize the Quantizer.

        This sets up the quantizer overrides as an ordered dictionary and initializes
        the quantizer metadata to None.
        """
        super().__init__()
        # Use OrderedDict even when dict would suffice as dict uses __slots__
        # which prevents making weakreferences as uses in OverrideHandle
        super(torch.nn.Module, self).__setattr__("_quantizer_overrides", collections.OrderedDict())
        self.quant_metadata = None

    def quantize(self, data: torch.Tensor) -> torch.Tensor:
        """Quantize the input data.

        This method should be overridden by subclasses to implement the actual
        quantization logic.

        Args:
            data: The input tensor to be quantized.

        Returns:
            torch.Tensor: The quantized tensor.
        """
        raise NotImplementedError

    def register_override(
        self, override_fn: override.OverrideFn[torch.Tensor]
    ) -> override.OverrideHandle:
        """Push a quantizer override on the module.

        This override will be called instead of `self.quantizer`. The quantizer
        and self.quantizer is passed into the overwrite and can be called. If
        multiple overrides are registered, calling the overwritten function from
        an override will trigger the 'next' override.

        Args:
            override_fn: The function override for quantize.

        Returns:
            (override.OverrideHandle) a handle that can be used to remove the
            pushed override.
        """
        handle = override.OverrideHandle(self._quantizer_overrides)
        self._quantizer_overrides[handle.handle_id] = override_fn
        return handle

    @typing_override
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        wrapped_quantize = override.apply_overrides(self, self.quantize, self._quantizer_overrides)
        return wrapped_quantize(data)

    def extra_repr(self) -> str:
        """Extra representation of the quantizer.

        This includes information about the registered overrides.

        Returns:
            str: The extra representation string.
        """
        extra_str = super().extra_repr()
        extra_str += "\n(overrides): \n" if self._quantizer_overrides else ""
        for i, (x, override_) in enumerate(self._quantizer_overrides.items()):
            extra_str += f"  ({i}): {override_}\n"
        return extra_str

    __call__: Callable[..., torch.Tensor]

    def is_stub(self) -> bool:
        """Returns: False, indicating this is not a stub."""
        return False


class QuantizerStub(Quantizer):
    """Stub class for Quantizers.

    Used for Quantizer/Quantized network initialization.

    Args:
        tags: Tags to be added to the metadata.
        weight_quantizer: Whether to add the weight quantizer tag.
        bias_quantizer: Whether to add the bias quantizer tag.
        input_quantizer: Whether to add the input quantizer tag.
        output_quantizer: Whether to add the output quantizer tag.
        shape: The shape attribute.
        __metadata: If provided, use as metadata and ignore all other arguments.
        kwargs: Additional attributes.
    """

    quant_metadata: QuantizerMetadata

    def __init__(
        self,
        *tags: str | Tag,
        weight_quantizer: bool = False,
        bias_quantizer: bool = False,
        input_quantizer: bool = False,
        output_quantizer: bool = False,
        shape: tuple[int, ...] | torch.Size | None = None,
        _metadata: QuantizerMetadata | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if _metadata is not None:
            self.quant_metadata = _metadata
        else:
            self.quant_metadata = QuantizerMetadata(
                *tags,
                weight_quantizer=weight_quantizer,
                bias_quantizer=bias_quantizer,
                input_quantizer=input_quantizer,
                output_quantizer=output_quantizer,
                shape=shape,
                **kwargs,
            )

    def quantize(self, data: torch.Tensor) -> torch.Tensor:
        """Stub quantize method that returns the input data unchanged.

        Args:
            data: The input tensor.

        Returns:
            torch.Tensor: The same input tensor, unchanged.
        """
        return data

    def is_stub(self) -> bool:
        """Check if this quantizer is a stub.

        Returns:
            bool: True, indicating this is a stub.
        """
        return True
