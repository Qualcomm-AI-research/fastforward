# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging
import textwrap
import warnings

from collections import defaultdict
from operator import attrgetter
from pathlib import Path
from typing import Any, Iterator, Literal, TypeAlias, Union, cast

import torch
import yaml

from safetensors import safe_open
from safetensors.torch import save_file

import fastforward as ff

from fastforward.cache import get_assets_path
from fastforward.exceptions import QuantizationError
from fastforward.nn import Quantizer, QuantizerMetadata, QuantizerStub

ModuleType: TypeAlias = type[torch.nn.Module]
QuantizedModuleType: TypeAlias = type["QuantizedModule"]
ModuleConversionDict: TypeAlias = dict[ModuleType, Union[QuantizedModuleType, "SkipQuantization"]]

logger = logging.getLogger(__name__)
logger.addFilter(ff.logging_utils.DuplicateLogFilter(levels=(logging.INFO, logging.WARNING)))


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
    the name of the quantizer as well as the quantizer itself. Yields only
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


def quantizer_state_dict(module: torch.nn.Module) -> dict[str, Any]:
    """State dict for all (and only) quantizers in `module`.

    state dict can be used using `torch.nn.Module.load_state_dict(state_dict,
    strict=False)` to load previously stored quantizer state.

    Args:
        module: The module for which to obtain a quantizer state dict.

    Returns:
        Dictionary that is a state dict for all (initialized) quantizers in
        `module`.
    """
    state_dict: dict[str, torch.Tensor] = {}
    for name, quantizer in ff.nn.quantized_module.named_quantizers(module):
        prefix = name + ("." if name else "")
        quantizer.state_dict(destination=state_dict, prefix=prefix)
    return state_dict


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


_OverwriteOptions: TypeAlias = Literal["overwrite"] | Literal["skip"] | Literal["error"]


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
            msg = f"{quantizer} is not a Quantizer subclass"  # type: ignore[unreachable]
            raise TypeError(msg)

        metadata = self.__dict__.get("_quantizer_metadata")
        if metadata is None:
            msg = (
                f"Cannot assign quantizer before {type(self).__name__}.__init_quantization__() call"
            )
            raise AttributeError(msg)

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

    def quantizer_state_dict(self) -> dict[str, Any]:
        """State dict for all (and only) quantizers in `module`.

        state dict can be used using `torch.nn.Module.load_state_dict(state_dict,
        strict=False)` to load previously stored quantizer state.

        Returns:
            Dictionary that is a state dict for all (initialized) quantizers in
            module.
        """
        return quantizer_state_dict(self)

    def save_quantization_state(
        self,
        *,
        tag: str = "main",
        name_or_path: str | Path | None = None,
        cache_dir: Path | None = None,
    ) -> Path:
        """Save quantization state to disk for later restoration.

        Saves the quantization state of all quantizers in this module to disk,
        including quantizer parameters, metadata, and configuration information.
        The state is saved as a SafeTensors file with accompanying YAML configuration.

        Args:
            tag: Tag to identify this particular save. Used to organize multiple
                saves of the same model. Defaults to "main".
            name_or_path: Model identifier or path. If None, attempts to extract
                from the model's config.name_or_path attribute. Used to determine
                the save location and validate consistency during loading.
            cache_dir: Directory where quantization state should be cached. If None,
                uses the default cache directory from get_assets_path().

        Returns:
            Path to the saved configuration file (config.yaml).

        Raises:
            RuntimeError: If the model identifier cannot be determined.
            ValueError: If the cache directory cannot be created due to an
                existing file with the same name.

        Note:
            The quantization state is saved in a directory structure:
            {cache_dir}/quantization-state/{name_or_path}/{tag}/
            containing 'model.safetensors' and 'config.yaml' files.
        """
        if name_or_path is None:
            name_or_path = getattr(getattr(self, "config", None), "name_or_path", None)
        if name_or_path is None:
            raise RuntimeError(
                "Unable to detect the model identifier. Please provide it manually "
                "if there is no `config.name_or_path` property in the model"
            )
        assets_path = get_assets_path(
            f"quantization-state/{name_or_path}", tag, cache_dir=cache_dir
        )
        try:
            assets_path.mkdir(exist_ok=True, parents=True, mode=0o775)
        except (FileExistsError, NotADirectoryError):
            msg = f"Cannot create directory {assets_path} because of an existing file."
            raise ValueError(msg)
        transformers_version = getattr(getattr(self, "config", None), "transformers_version", None)
        fastforward_version = ff.__version__
        quantizers = defaultdict(list)
        for name, quantizer in self.named_quantizers(remove_duplicate=False):
            quantizers[quantizer].append(name)
        # State dictionary containing quantizer parameters keyed by
        # "first_quantizer_name.param_name". For shared quantizers (same quantizer instance used
        # in multiple locations), parameters are stored only once using the lexicographically first
        # quantizer name to avoid duplication.
        # Example: {
        #     "layer1.weight_quantizer.scale": tensor([1.0]),
        #     "layer1.weight_quantizer.offset": tensor([0])
        # }
        state = {}
        # Metadata mapping each quantizer name to its parameter keys in the format
        # "param=tensor_key". This enables reconstruction of individual quantizer state_dicts during
        # loading by mapping parameter names to their corresponding tensor keys in the SafeTensors
        # file.
        # Example: {
        #     "layer1.weight_quantizer":
        #         "scale=layer1.weight_quantizer.scale,offset=layer1.weight_quantizer.offset",
        #     "layer2.weight_quantizer":
        #         "scale=layer1.weight_quantizer.scale,offset=layer1.weight_quantizer.offset",
        # }
        metadata = {}
        for quantizer, names in quantizers.items():
            first_name = min(names)
            for key, value in quantizer.state_dict().items():
                state[f"{first_name}.{key}"] = value
            for name in names:
                metadata[name] = ",".join(
                    f"{key}={first_name}.{key}" for key in quantizer.state_dict().keys()
                )
        config = {
            "version": "1.0",
            "name_or_path": str(name_or_path),
            "transformers_version": str(transformers_version),
            "fastforward_version": str(fastforward_version),
            "quantizers": {
                name: quantizer for quantizer, names in quantizers.items() for name in names
            },
        }
        save_file(state, assets_path / "model.safetensors", metadata=metadata)
        config_path = assets_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, sort_keys=False)
        return config_path

    def load_quantization_state(
        self,
        *,
        tag: str = "main",
        name_or_path: str | Path | None = None,
        cache_dir: Path | None = None,
        overwrite_policy: _OverwriteOptions = "error",
    ) -> None:
        """Load quantization state from saved files.

        Args:
            tag: Tag used when saving the quantization state. Defaults to "main".
            name_or_path: Model identifier used when saving. If None, attempts to get from config.
            cache_dir: Directory where the quantization state was cached. If None, uses
                default cache.
            overwrite_policy: The policy to use when a loader quantizer is already present
                in the model. Options are 'skip', 'overwrite' and 'error'.

        Raises:
            RuntimeError: If the model identifier cannot be determined.
            FileNotFoundError: If the quantization state files are not found.
            ValueError: If the loaded configuration is incompatible.
        """
        name: str | None = getattr(getattr(self, "config", None), "name_or_path", None)
        if name_or_path is not None and not Path(name_or_path).exists():
            name = str(name_or_path)
        if name is None:
            raise RuntimeError(
                "Unable to detect the model identifier. Please provide it manually "
                "if there is no `config.name_or_path` property in the model"
            )
        if name_or_path is not None and Path(name_or_path).exists():
            config_path = Path(name_or_path)
        else:
            config_path = (
                get_assets_path(f"quantization-state/{name}", tag, cache_dir=cache_dir)
                / "config.yaml"
            )
        model_path = config_path.parent / "model.safetensors"

        # Check if files exist
        if not config_path.exists():
            msg = f"Quantization state config not found at {config_path}"
            raise FileNotFoundError(msg)
        if not model_path.exists():
            msg = f"Quantization state model not found at {model_path}"
            raise FileNotFoundError(msg)

        # Load configuration
        with open(config_path, "r") as f:
            config = yaml.load(f, yaml.Loader)

        # Validate configuration
        if config.get("version") != "1.0":
            msg = f"Unsupported quantization state version: {config.get('version')}"
            raise ValueError(msg)

        # if user provides a fill path to the config, we assume he knows what he is doing
        if str(config.get("name_or_path")) != str(name):
            msg = (
                f"Model identifier mismatch: expected '{name_or_path}', "
                f"found '{config.get('name_or_path')}' in saved state"
            )
            logger.error(msg)
            raise RuntimeError(msg)

        quantizers: dict[str, Quantizer] = config.get("quantizers", {})
        # Reconstruct quantizer state_dict by parsing metadata to map parameter names to tensor keys.
        # The metadata format "param=tensor_key" allows to load the correct tensors for each
        # parameter. For shared quantizers, multiple quantizer names may reference the same tensor
        # keys.
        with safe_open(model_path, framework="pt") as f:
            metadata = f.metadata()
            for name, quantizer in quantizers.items():
                missing_keys, unexpected_keys = quantizer.load_state_dict({
                    state_key: f.get_tensor(tensor_key)
                    for key in metadata[f"{name}"].split(",")
                    for state_key, tensor_key in (key.split("="),)
                })
                if missing_keys or unexpected_keys:
                    msg = (
                        f"There are some missing ({missing_keys}) or unexpected "
                        f"({unexpected_keys}) keys during loading state_dict"
                    )
                    logger.error(msg)
                    raise RuntimeError(msg)

        for name, quantizer in quantizers.items():
            parts = name.rsplit(".", 1)
            parent = self if len(parts) == 1 else attrgetter(parts[0])(self)
            parent_attribute = parts[-1]
            current_quantizer = getattr(parent, parent_attribute, None)

            is_quantizer = isinstance(current_quantizer, Quantizer)
            is_quantizer_stub = isinstance(current_quantizer, QuantizerStub)
            if is_quantizer and not is_quantizer_stub:
                if overwrite_policy == "error":
                    msg = (
                        f"'{name}' is a quantizer, but is already initialized. If "
                        + 'you want to overwrite the existing quantizer, use overwrite_policy="overwrite" '
                        + "or if you want to skip loading existing quantizers use "
                        + 'overwrite_policy="skip"'
                    )
                    raise QuantizationError()
                elif overwrite_policy == "skip":
                    continue
                elif overwrite_policy != "overwrite":
                    msg = (  # type: ignore[unreachable]
                        "Encountered a quantizer that was already initialized. Since "
                        + f"overwrite_policy={overwrite_policy} is illegal cannot resolve conflict."
                        + "please use 'error', 'skip', or 'overwrite"
                    )
                    raise QuantizationError(msg)
            if not is_quantizer:
                msg = f"'{name}' is not a quantizer or was overwritten by a non-quantizer object"
                raise ValueError(msg)

            setattr(parent, parent_attribute, quantizer)


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
    else:
        _quantize_module(model, module_map)

    if isinstance(model, QuantizedModule) and recursive:
        model.quantize_children(
            extra_conversion=extra_conversion, skip_quantized_modules=skip_quantized_modules
        )

    return model


def _quantize_module(module: torch.nn.Module, module_map: ModuleConversionDict) -> None:
    try:
        quantized_class = module_map[type(module)]
    except KeyError as e:
        msg = (
            f"Quantization is not supported for '{type(module)}'. \n"
            f"Supported types can be found in `fastforward.nn.quantized_module.quantized_module_map`"
        )
        raise QuantizationError(msg) from e

    if isinstance(quantized_class, SkipQuantization):
        logger.info(
            "Skipping quantization of '%s' because the conversion is set to "
            "fastforward.nn.quantized_module.SKIP_QUANTIZATION",
            type(module).__name__,
        )
        return

    original_class = module.__class__
    module.__class__ = quantized_class
    module = cast(QuantizedModule, module)
    module.__init_quantization__()
    logger.debug("Converting '%s' to '%s'", original_class.__name__, quantized_class.__name__)


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
            logger.warning(
                "Multiple quantized versions of '%s.%s' exists. "
                "Defaulting to '%s.%s' which was created last",
                module_type.__module__,
                module_type.__qualname__,
                qtype.__module__,
                qtype.__qualname__,
            )
        mapping[module_type] = quantized_module_types[-1]

    return mapping
