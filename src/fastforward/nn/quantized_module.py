# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging
import textwrap
import warnings

from typing import Any, Iterator, TypeAlias, Union, cast

import torch

import fastforward as ff

from fastforward.exceptions import QuantizationError
from fastforward.nn import Quantizer, QuantizerMetadata, QuantizerStub

ModuleType: TypeAlias = type[torch.nn.Module]
QuantizedModuleType: TypeAlias = type["QuantizedModule"]
ModuleConversionDict: TypeAlias = dict[ModuleType, Union[QuantizedModuleType, "SkipQuantization"]]

logger = logging.getLogger(__name__)
logger.addFilter(ff.logging.DuplicateLogFilter(levels=(logging.INFO, logging.WARNING)))


class _QuantizedModuleMeta(type):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        instance = super().__call__(*args, **kwargs)
        instance.__init_quantization__()
        return instance


def named_quantizers(
    module: torch.nn.Module,
    prefix: str = "",
    recurse: bool = True,
    remove_duplicate: bool = True,
    skip_stubs: bool = True,
) -> Iterator[tuple[str, Quantizer]]:
    """`Iterator` over quantizers with names.

    Return an iterator over `QuantizedModule`s in module, yielding both
    the name of the quantizer as well ass the quantizer itself. Yields only
    direct children if `recurse` is False.

    Args:
        module: The module to yield the quantizers for.
        prefix: Str that prefixed to the name of the module.
        recurse: Only yield direct children if True, yield all quantizers in
            submodules otherwise.
        remove_duplicate: Flag indicating whether duplicated should be
            yielded once or multiple times.
        skip_stubs: Do not yield instances of `QuantizerStub` if True.

    Yields:
        (str, QuantizerModule): `Tuple` of name and quantizer
    """
    if recurse:
        module_iter = module.named_modules(prefix="", remove_duplicate=remove_duplicate)
    else:
        module_iter = module.named_children()
    for name, module in module_iter:
        if isinstance(module, Quantizer):
            if skip_stubs and isinstance(module, QuantizerStub):
                continue
            prefixed_name = prefix + ("." if prefix else "") + name
            yield prefixed_name, module


_QUANTIZED_MODULE_MAP: dict[type[torch.nn.Module], list[type["QuantizedModule"]]] = {}


def _record_quantized_module(cls: type["QuantizedModule"]) -> None:
    module_bases: list[type[torch.nn.Module]] = []
    for basecls in cls.__bases__:
        if not issubclass(basecls, QuantizedModule) and issubclass(basecls, torch.nn.Module):
            module_bases.append(basecls)

    if len(module_bases) == 0:
        # We are subclassing a QuantizedModule. If there is exactly 1 Module in
        # the MRO, we can associate the new quantized module with that module,
        # otherwise the user is responsible for including it manually.
        for basecls in cls.__mro__[1:]:
            if issubclass(basecls, QuantizedModule):
                continue
            if issubclass(basecls, torch.nn.Module) and basecls is not torch.nn.Module:
                module_bases.append(basecls)

    if len(module_bases) == 0:
        # Nothing to record in module map because no non-quantized baseclass is
        # present
        return

    if len(module_bases) > 1:
        # In the unlikely case of multiple module bases, we cannot automatically
        # include it in the module map. In this case, the end user is responsible
        # for including it manually.
        return

    module = module_bases[0]
    _QUANTIZED_MODULE_MAP.setdefault(module, []).append(cls)


class QuantizedModule(torch.nn.Module, metaclass=_QuantizedModuleMeta):  # pylint: disable=abstract-method
    """Base class for quantized neural network models/modules.

    # Extending existing (non-quantized) modules
    In many cases, a quantized module will be a specialization of a non-quantized module.
    `QuantizedModule` follows an extension principle, i.e., QuantizedModule extends the
    existing classes by implementing all the quantization related initialization logic in the
    `__init_quantization__()` method. I.e., no quantization logic should be implemented
    in `__init__()`

    When a quantized module is initialized, first the `__init__()` of the super classes is called.
    Once these conclude, `__init_quantization__()` is executed. Alternatively, one can create
    an instance of the super class. Update the `__class__` attribute to the quantized class
    and call `__init_quantization__()`. By adhering to this separation of initialization, we can
    easily convert submodules of a non-quantized neural network to their quantized counterparts.

    Quantized classes are automatically detected through the subclass hierarchy. For example,
    a class that subclasses `QuantizedModule` and `torch.nn.Linear` is considered a quantized
    implementation of `torch.nn.Linear` and will be used in `quantized_module`. To opt-out from this
    automatic discovery mechanism, pass `include_in_module_map=False` as class argument:

    ```python
    class MyQuantizedModule(torch.nn.QuantizedModule, include_in_module_map=False):
        ...
    ```

    Note:
        Future versions of this library will include support to automatically generate quantized
        counterparts. However, until that feature lands, quantized counterparts need to be
        implemented manually. Quantized implementations for a subset of modules available in
        PyTorch can be found in the `fastforward.nn` module.

    # `QuantizedStub`s
    A QuantizedModule does not define specific quantizers during initialization. Instead, one
    or more `QuantizerStub`s are defined. These may be replaced by explicit quantizers at a later
    time. This makes `QuantizedModule`s general over particular quantizers.
    """

    _quantizer_metadata: dict[str, QuantizerMetadata]

    def __init_quantization__(self) -> None:
        """Quantization specific initializer.

        This method is automatically called when a `QuantizedModule` is
        initialized directly, after `__init__()`.
        """
        super(torch.nn.Module, self).__setattr__("_quantizer_metadata", {})

    def __init_subclass__(cls, include_in_module_map: bool = True) -> None:
        """Record `QuantizedModule` subclasses in module map for model conversion.

        Args:
            include_in_module_map: If True, the newly created `QuantizedModule` is recorded
                in the global module conversion map. This means that the module
                will be automatically discovered and used for module replacements
                in `ff.quantize_model`. If False, this registration is skipped. This
                can be useful for one-off modules or modules for which you want
                to provide the mapping explicitly.
        """
        if not include_in_module_map:
            return
        _record_quantized_module(cls)

    def quantize_children(
        self: torch.nn.Module,
        extra_conversion: ModuleConversionDict | None = None,
        skip_quantized_modules: bool = False,
    ) -> None:
        """Quantize children.

        Converts the children modules to their `QuantizedModule` counterparts defined in
        `fastforward.nn.quantized_module.quantized_module_map` and `extra_conversion`.
        This may be used as part of `__init_quantization__` to perform recursive quantization
        initialization.

        Args:
            extra_conversion: A dict that maps `torch.nn.Module` to `QuantizedModule` subclasses.
              For any conversion, this dict is first checked. If there is no match, the general
              mapping as given by `quantized_module_map` is used.
            skip_quantized_modules: If `True` do not try to requantize already
                quantized modules.

        Warning:
            If this method is used on a module that has submodules for which no
            mapping is available in both `extra_conversion` and
            `quantized_module_map`, an error is thrown and the conversions
            fails.
        """
        for _, child in self.named_children():
            if not isinstance(child, Quantizer):
                quantize_model(
                    child,
                    extra_conversion=extra_conversion,
                    skip_quantized_modules=skip_quantized_modules,
                )

    def register_quantizer(
        self, name: str, quantizer: Quantizer | None, *, _register_module: bool = True
    ) -> None:
        """Register new quantizer on module.

        Register a quantizer with under `name` on this module. This method is similar to
        register_module, but specific to quantizers. If an attribute is set to a quantizer
        on a QuantizerModule, this method is called automatically.

        Args:
            name: The name of the quantizer. It will be exposed under this name as an attribute
                on the QuantizedModule
            quantizer: The quantizer to register.
        """
        if not isinstance(quantizer, Quantizer) and quantizer is not None:
            raise TypeError(f"{quantizer} is not a Quantizer subclass")

        metadata = self.__dict__.get("_quantizer_metadata")
        if metadata is None:
            raise AttributeError(
                f"Cannot assign quantizer before {type(self).__name__}.__init_quantization__() call"
            )

        if _register_module:
            self.register_module(name, quantizer)

        if quantizer is None:
            return

        # Either this module or the set quantizer may store metadata.
        # Here we ensure that the module/quantizer's metadata is consistent.
        # This may result in overwriting the quantizer's metadata, which in turn
        # may result in an inconsistency with earlier setup. In this case
        # a warning is raised.
        module_has_metadata = name in self._quantizer_metadata
        if quantizer.quant_metadata is not None:
            if not module_has_metadata:
                # quantizer has metadata, but module has not
                self._quantizer_metadata[name] = quantizer.quant_metadata
            else:
                # both module and quantizer have metadata, check for consistency
                if not self._quantizer_metadata[name].is_extension(quantizer.quant_metadata):
                    warnings.warn(
                        f"Quantizer metadata for {name} is not a consistent "
                        f"extension with stored quantization metadata for {name}. "
                        "The quantizer metadata is updated to match the module. "
                        "Because of this, the quantization state may become inconsistent, "
                        "for example, when the same quantizer is shared.",
                        RuntimeWarning,
                    )
        elif not module_has_metadata:
            # No metadata on quantizer or module, use default
            self._quantizer_metadata[name] = QuantizerMetadata()

        quantizer.quant_metadata = self._quantizer_metadata[name]

    def __setattr__(self, name: str, value: torch.Tensor | torch.nn.Module) -> None:
        super().__setattr__(name, value)
        if isinstance(value, Quantizer):
            # Module was already registered through Module.__setattr__(...)
            # Only register as quantizer
            self.register_quantizer(name, value, _register_module=False)

    def named_quantizers(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
        skip_stubs: bool = True,
    ) -> Iterator[tuple[str, Quantizer]]:
        """`Iterator` over quantizers and their names.

        Return an iterator over `QuantizedModule`s in the network, yielding both
        the name of the quantizer as well ass the quantizer itself. Yields only
        direct children if `recurse` is False.

        Args:
            prefix: Str that prefixed to the name of the module.
            recurse: Only yield direct children if True, yield all quantizers in
                submodules otherwise.
            remove_duplicate: Flag indicating whether duplicated should be
                yielded once or multiple times.
            skip_stubs: Do not yield instances of `QuantizerStub` if True.

        Yields:
            (str, QuantizerModule): `Tuple` of name and quantizer
        """
        yield from named_quantizers(self, prefix, recurse, remove_duplicate, skip_stubs=skip_stubs)

    def quantizers(self, recurse: bool = True, skip_stubs: bool = True) -> Iterator[Quantizer]:
        """Iterator over quantizers.

        Return an iterator over `QuantizedModule`s in the network. Yields only
        direct children if `recurse` is False.

        Args:
            recurse: Only yield direct children if True, yield all quantizers in
                submodules otherwise.
            skip_stubs: Do not yield instances of `QuantizerStub` if True.
        """
        for _, quantizer in self.named_quantizers(recurse=recurse, skip_stubs=skip_stubs):
            yield quantizer


class SkipQuantization:
    """Marker class to signify that a module must not be quantized."""

    def __repr__(self) -> str:
        return "<skip quantization>"


SKIP_QUANTIZATION = SkipQuantization()


def _missing_modules(
    model: torch.nn.Module,
    module_map: dict[ModuleType, QuantizedModuleType | SkipQuantization],
    skip_quantized_modules: bool = False,
) -> list[type[torch.nn.Module]]:
    """Find unquantizable modules.

    Returns a list of Module types that appear in `model` but for which no
    quantized counterpart is available.
    """
    submodule_types = {type(module) for module in model.modules()} | {type(model)}
    missing_modules: list[type[torch.nn.Module]] = []
    for module_type in submodule_types:
        if module_type not in module_map:
            if module_type is torch.nn.Module:
                continue
            if issubclass(module_type, Quantizer):
                continue
            if skip_quantized_modules and issubclass(module_type, QuantizedModule):
                continue
            missing_modules.append(module_type)
    return missing_modules


def _check_quantizable(
    model: torch.nn.Module,
    module_map: dict[ModuleType, QuantizedModuleType | SkipQuantization],
    skip_quantized_modules: bool = False,
) -> None:
    missing_modules = _missing_modules(model, module_map, skip_quantized_modules)
    if not missing_modules:
        return

    module_list = "\n".join(
        f"      - {module.__module__}.{module.__qualname__}" for module in missing_modules
    )
    msg = f"""
    Cannot quantize model because no quantized version of the following modules is known:
{module_list}
    It is possible that quantized definitions of one or more of these models
    exists, but have not been imported."
    """

    raise QuantizationError(textwrap.dedent(msg).strip())


def surrogate_quantized_modules(model: torch.nn.Module) -> ModuleConversionDict:
    """Create surrogate quantization modules for prototyping.

    Construct a `ModuleConversionDict` that contains surrogate quantized modules
    of all submodules in `model` for which no quantized counterpart exists.
    In this context, a surrogate is a `QuantizedModule` without any changes to
    it's forward pass. For example, a surrogate for `MyModule` looks like:

        class QuantizedMyModule(QuantizedModule, MyModule, include_in_module_map=False):
            pass

    These surrogates act as quantized implementations, but perform no real
    quantization. This is useful to temporarily create placeholder quantized
    implementations or for quantized implementations of container Modules,
    i.e., modules which only call submodules.

    When the result of this function is used as `extra_conversion` in
    `quantize_model` the quantization conversion process will always succeed,
    however, the resulting model may not be correctly quantized. For example,
    given our `MyModule` example, `MyModule.forward` may perform an operation
    that should be quantized. Since the forward method is untouched, the
    resulting model may not be correctly quantizable.

    Note that when using this function, the created classes are not
    automatically discovered as with normal `QuantizedModule` creation.
    Hence, future calls of `quantize_model` will not incorrectly discover
    classes that where created through this function.

    Warning:
        Caution is advised when using this function. Inspect the result of
        `quantize_model` and validate if the created model matches your
        expectations. All quantized modules introduces by using this function
        have a type name that ends in 'Surrogate'.

    Args:
        model: `Model` for which to create placeholder `QuantizedModule` implementations

    Returns:
        a `ModuleConversionDict` that can be used as `extra_conversion` in `quantize_model`
    """
    module_map: ModuleConversionDict = {}
    default_module_map: ModuleConversionDict = cast(ModuleConversionDict, quantized_module_map())
    for module in _missing_modules(
        model, module_map=default_module_map, skip_quantized_modules=True
    ):
        quantized_module = type(
            f"Quantized{module.__name__}Surrogate",
            (QuantizedModule, module),
            {},
            include_in_module_map=False,
        )
        module_map[module] = cast(type[QuantizedModule], quantized_module)
    return module_map


def quantize_model(
    model: torch.nn.Module,
    recursive: bool = True,
    extra_conversion: ModuleConversionDict | None = None,
    skip_quantized_modules: bool = False,
) -> torch.nn.Module:
    """Convert modules and submodules to quantized counterparts.

    Converts a `torch.nn.Module` and it's children to their `QuantizedModule`
    counterparts defined in
    `fastforward.nn.quantized_module.quantized_module_map` and `extra_conversion`.

    Args:
      model: Model to quantize
      recursive: If True, recursively quantize all submodules
      extra_conversion: A mapping from `torch.nn.Module` subclasses to `QuantizedModule`
        subclasses that extends the default mapping returned by `quantized_module_map`.
        All (sub)modules in model will be replaced according to this mapping. A module
        can be marked as `do not replace` by adding an entry that maps to
        `fastforward.nn.quantized_module.SKIP_QUANTIZATION`.
      skip_quantized_modules: If True, do not attempt to replace already quantized modules. If
        False, quantized modules are treated as any other module and may be replaced if an entry
        is available in the conversion map or an error is thrown otherwise.
    """
    module_map = quantized_module_map() | (extra_conversion or {})
    _check_quantizable(model, module_map, skip_quantized_modules=skip_quantized_modules)

    if skip_quantized_modules and isinstance(model, QuantizedModule):
        logger.info(
            "Skipping requantization of '%s' because skip_quantized_modules=True", type(model)
        )
        return model

    try:
        quantized_class = module_map[type(model)]
    except KeyError as e:
        raise QuantizationError(
            f"Quantization is not supported for '{type(model)}'. \n"
            f"Supported types can be found in `fastforward.nn.quantized_module.quantized_module_map`"
        ) from e

    if isinstance(quantized_class, SkipQuantization):
        logger.info(
            "Skipping quantization of '%s' because the conversion is set to "
            "fastforward.nn.quantized_module.SKIP_QUANTIZATION",
            type(model).__name__,
        )
        return model

    original_class = model.__class__
    model.__class__ = quantized_class
    model = cast(QuantizedModule, model)
    model.__init_quantization__()
    logger.debug("Converting '%s' to '%s'", original_class.__name__, quantized_class.__name__)

    if recursive:
        model.quantize_children(
            extra_conversion=extra_conversion, skip_quantized_modules=skip_quantized_modules
        )

    return model


def quantized_module_map() -> dict[ModuleType, QuantizedModuleType]:
    """Returns a dictionary that maps `torch.nn.Module`s to their `ff.nn.QuantizedModule` counterparts.

    Warning:
        This finds all known subclasses of `ff.nn.QuantizedModule`, hence, in
        order for a particular `QuantizedModule` subclass to be included it
        needs to be created.
    """
    mapping: dict[ModuleType, QuantizedModuleType] = {}
    for module_type, quantized_module_types in _QUANTIZED_MODULE_MAP.items():
        if not quantized_module_types:
            continue
        if len(quantized_module_types) > 1:
            qtype = quantized_module_types[-1]
            logger.warn(
                "Multiple quantized versions of '%s.%s' exists. "
                "Defaulting to '%s.%s' which was created last",
                module_type.__module__,
                module_type.__qualname__,
                qtype.__module__,
                qtype.__qualname__,
            )
        mapping[module_type] = quantized_module_types[-1]

    return mapping
