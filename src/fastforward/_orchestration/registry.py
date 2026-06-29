# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Maps algorithms to the torch modules they target.

We provide some default mappings at the end of this file.
"""

from __future__ import annotations

import abc
import contextlib
import dataclasses

from collections.abc import Callable, Generator
from typing import Any, Iterable, Iterator, Mapping, TypeAlias, cast

import torch

import fastforward as ff

from fastforward import mpath
from fastforward._orchestration.graph_module import Region, SubgraphSpec

Algorithm: TypeAlias = Callable[..., Any]

TargetType: TypeAlias = (
    type[torch.nn.Module]
    | torch.nn.Module
    | mpath.selector.BaseSelector
    | Iterable["type[torch.nn.Module] | torch.nn.Module | mpath.selector.BaseSelector | Selector"]
)


class Selector(abc.ABC):
    """Abstract base class for target selectors.

    Subclass this and implement :meth:`resolve` to create a custom selector.
    """

    @abc.abstractmethod
    def resolve(self, model: torch.nn.Module) -> list[Region]:
        """Return the regions in *model* that this selector targets."""
        ...


@dataclasses.dataclass(frozen=True)
class ModuleTypeSelector(Selector):
    """Selects all modules in a model that are instances of the given types."""

    types: tuple[type[torch.nn.Module], ...]

    def resolve(self, model: torch.nn.Module) -> list[Region]:  # noqa: D102
        return [m for m in model.modules() if isinstance(m, self.types)]


@dataclasses.dataclass(frozen=True)
class ModuleInstanceSelector(Selector):
    """Selects an explicit set of module instances, bypassing any search.

    This selector requries **all** modules to be present on the model.
    """

    modules: frozenset[torch.nn.Module]

    def resolve(self, model: torch.nn.Module) -> list[Region]:  # noqa: D102
        if not self.modules <= (model_modules := set(model.modules())):
            missing = self.modules - model_modules
            msg = f"Modules {[type(m).__name__ for m in missing]} not found on model {type(model).__name__}."
            raise ValueError(msg)

        return list(self.modules)


@dataclasses.dataclass(frozen=True)
class MPathSelector(Selector):
    """Selects modules in a model by a (pre-parsed) mpath query."""

    query: mpath.selector.BaseSelector

    def resolve(self, model: torch.nn.Module) -> list[Region]:  # noqa: D102
        return list(mpath.search(self.query, model).modules())


@dataclasses.dataclass(frozen=True)
class CompositeSelector(Selector):
    """Unions the results of multiple selectors, preserving order and deduplicating."""

    selectors: tuple[Selector, ...]

    def resolve(self, model: torch.nn.Module) -> list[Region]:  # noqa: D102
        seen: set[int] = set()
        result: list[Region] = []
        for selector in self.selectors:
            for m in selector.resolve(model):
                if id(m) not in seen:
                    seen.add(id(m))
                    result.append(m)
        return result


def normalize(target: TargetType | Selector) -> Selector:
    """Normalize / check possible target types to a standard object.

    Targets as provided by users can be a variety of inputs. We expect:
    - (sequence of) Module types : I want to target all torch.nn.Linear layers.
    - (sequence of) instantiated modules : I want to target these specific layers.
    - A parsed mpath query : I want to target the modules this query matches.

    Args:
        target: A user provided target that should be normalized.

    Returns:
         A selector that can resolve the target against a model.
    """
    match target:
        case Selector():
            return target
        case type() if issubclass(target, torch.nn.Module):
            return ModuleTypeSelector(types=(target,))
        case mpath.selector.BaseSelector():
            return MPathSelector(query=target)
        case torch.nn.Module():
            return ModuleInstanceSelector(modules=frozenset([target]))
        case target if isinstance(target, Iterable) and not isinstance(target, str):  # type: ignore[redundant-expr, unreachable]
            if not (items := list(target)):
                raise TypeError("Empty target sequence")

            if all(isinstance(t, type) and issubclass(t, torch.nn.Module) for t in items):
                return ModuleTypeSelector(types=tuple(cast(list[type[torch.nn.Module]], items)))

            return CompositeSelector(selectors=tuple(normalize(item) for item in items))
        case type():
            msg = f"Expected a torch.nn.Module subclass, got {target!r}."  # type: ignore[unreachable]
            raise TypeError(msg)
        case _:
            msg = f"Invalid target: expected a Module type, Module instance, parsed mpath query, or sequence thereof; got {target!r}."  # type: ignore[unreachable]
            raise TypeError(msg)


class NoTargetsFound(Exception):
    """Raised when no selector is registered for the given algorithm."""


@dataclasses.dataclass(frozen=True)
class AlgorithmSpec:
    """The configuration registered for an algorithm."""

    target: Selector


class _AlgorithmRegistry(Mapping[Algorithm, AlgorithmSpec]):
    """A mapping of algorithms to their registered configuration.

    Targets are validated and normalized into a selector upon registration.
    Each algorithm has a single spec; re-registering overwrites the previous
    one if it existed.
    """

    def __init__(self) -> None:
        self._specs: dict[Algorithm, AlgorithmSpec] = {}

    def __getitem__(self, algorithm: Algorithm) -> AlgorithmSpec:
        return self._specs[algorithm]

    def __iter__(self) -> Iterator[Algorithm]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def register(self, algorithm: Algorithm, target: TargetType | Selector) -> None:
        """Register an algorithm, target pair.

        Accepts a target specification (see `TargetType`) or a Selector, validates it
        and stores it as a `Selector`, which can be used at resolve time.

        If the algorithm was previously registered, the existing spec is **overwritten**.

        Args:
            algorithm: The algorithm that runs on 'target'.
            target: A Selector, Module type, Module instance, parsed mpath query, or sequence thereof.
        """
        self._specs[algorithm] = AlgorithmSpec(target=normalize(target))

    def resolve(self, model: torch.nn.Module, algorithm: Algorithm) -> list[SubgraphSpec]:
        """Return specs for each region in a model that the given algorithm targets.

        Raises:
            NoTargetsFound: If no target is registered for the algorithm, or if the
                registered target matched no modules on the model.
        """
        if algorithm not in self._specs:
            msg = f"No target registered for algorithm {algorithm.__name__!r}."
            raise NoTargetsFound(msg)

        if not (regions := self._specs[algorithm].target.resolve(model)):
            msg = f"Target for {algorithm.__name__!r} matched no modules on {type(model).__name__}."
            raise NoTargetsFound(msg)

        return [SubgraphSpec(region=region, fn=algorithm) for region in regions]


_registry = _AlgorithmRegistry()


def register(algorithm: Algorithm, targets: TargetType | Selector) -> None:
    """Declare which modules an algorithm should target.

    Re-registering the same algorithm overwrites the previous target.

    Args:
        algorithm: The algorithm to register targets for.
        targets: A Selector, module type, tuple of types, parsed mpath query, or list of module instances.
    """
    _registry.register(algorithm, targets)


def resolve(model: torch.nn.Module, algorithm: Algorithm) -> list[SubgraphSpec]:
    """Resolve which regions in a model an algorithm targets.

    Args:
        model: The model to resolve targets against.
        algorithm: The algorithm whose targets to resolve.

    Raises:
        NoTargetsFound: If no target is registered for the algorithm, or if the
            registered target matched no modules on the model.
    """
    return _registry.resolve(model, algorithm)


@contextlib.contextmanager
def override(
    algorithm: Algorithm, targets: TargetType | Selector | None
) -> Generator[None, None, None]:
    """Temporarily override the registered target for an algorithm.

    If *targets* is None the existing registration is used unchanged.

    Args:
        algorithm: The algorithm whose targets to override.
        targets: The temporary target specification, or None to keep the current one.
    """
    if targets is None:
        yield
        return

    previous = _registry._specs.get(algorithm)
    _registry.register(algorithm, targets)
    try:
        yield
    finally:
        if previous is None:
            _registry._specs.pop(algorithm, None)
        else:
            _registry._specs[algorithm] = previous


# We pre-register baseline methods with expected Algorithm-Target pairs.
register(ff.quantization.gptq, (ff.nn.QuantizedLinear, ff.nn.QuantizedConv2d))
