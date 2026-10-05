# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Declarative data requirements for layer-wise optimization.

An algorithm that optimizes a layer needs data to work with. GPTQ needs the
activations that arrive at the layer. AdaRound needs those inputs paired with
the fp outputs of the same layer, both from the original model.

A `DataFlow` declares *what* data an algorithm needs, not *how* to produce it.
The declaration is written once, at registration time, when no concrete graph
exists yet. The scheduler later works out which modules to run, what to cache,
and in which order.

## Example

AdaRound calculates || Wx - W'x ||^2 with W' the quantized weights.
We could specify the data requirements in several ways:

```python
# (1): our function requires fp x, we do Wx and W'x.
flows = [InputActivations("original")]  # x

# (2): our function requires fp x, and fp Wx, we do W'x.
flows = [
    InputActivations("original"),  # x
    OutputActivations("original")  # Wx
]

# (3): We want to adjust for calibration (e.g. GPTQ in practice):
flows = [InputActivations("quantized")]  # x'
```


An algorithm can also ask for the gradient of a scalar objective at the output of
its region. YAQA, for example, needs the gradients of the original model:

```python
fisher = Gradient(empirical_nll, logits=ModelOutputs("original"), labels=loader)
flows = [InputActivations("original"), fisher]
# The algorithm is called as fn(module, inputs, output_grads).
```
The function is called once per run, iterates the data itself, and yields
one real scalar loss per batch:

```python
def empirical_nll(logits, labels):
    for ((out,), _), target in zip(logits, labels):
        yield torch.nn.functional.cross_entropy(out, target)
```

All regions that read equal `Gradient` flows share one call of the function and
one backward step per node. The gradients show the weights from before every
mutation, so only a pinned generator is supported, and the model must be in eval
mode.

Kinds of flow today: `InputActivations`, `OutputActivations`, `Gradient`.
"""

import abc

from contextlib import nullcontext
from types import MappingProxyType
from typing import Any, Callable, ContextManager, Iterable, Mapping

import attrs
import torch

import fastforward as ff


@attrs.define(frozen=True, eq=False)
class FlowGenerator:
    """Defines the execution context and scheduling order of a data flow.

    Args:
        key: Unique identifier for this generator.
        context: Factory that produces the context manager for the forward pass.
        order: Scheduling position (lower runs first).
        pinned: Whether the data shows the weights from before every mutation. A
            pinned generator loses its data the moment an optimization changes a
            weight it reads through, so the scheduler must take that data first. A
            generator that is not pinned shows the weights as they are, so its data
            can always be produced again.
    """

    key: str
    context: Callable[[torch.nn.Module | None], ContextManager[None]]
    order: int
    pinned: bool = False


_generators: dict[str, FlowGenerator] = {}


def register_generator(generator: FlowGenerator) -> FlowGenerator:
    """Make a FlowGenerator available by its key for use in flow declarations.

    Once registered, the generator's key can be passed as a shorthand string to
    DataFlow constructors (e.g. `InputActivations("original")`).

    Args:
        generator: The FlowGenerator to register.

    Returns:
        The same generator.
    """
    _generators[generator.key] = generator
    return generator


def _disable_quantization(module: torch.nn.Module | None) -> ContextManager[None]:
    """Disable quantization on `module`, or do nothing without one.

    A `call_function`/`call_method` node holds no module, so there is nothing
    to disable; entering the returned context is then a no-op.
    """
    if module is None:
        return nullcontext()
    return ff.disable_quantization(module)


def _quantized_context(_: torch.nn.Module | None) -> ContextManager[None]:
    return nullcontext()


def _any_context(_: torch.nn.Module | None) -> ContextManager[None]:
    return nullcontext()


# Run with quantization disabled; produces baseline (unquantized) activations.
ORIGINAL = register_generator(
    FlowGenerator("original", _disable_quantization, order=2, pinned=True)
)

# Run with the model as-is; produces activations reflecting all mutations so far.
QUANTIZED = register_generator(FlowGenerator("quantized", _quantized_context, order=5))

# No constraints on the model state; default for a plain forward pass.
ANY = register_generator(FlowGenerator("any", _any_context, order=10))


def _to_generator(value: str | FlowGenerator | ContextManager[None]) -> FlowGenerator:
    if isinstance(value, FlowGenerator):
        return value

    if isinstance(value, str):
        if value not in _generators:
            available = ", ".join(repr(k) for k in _generators)
            msg = f"Unknown flow generator {value!r}; registered: {available}"
            raise KeyError(msg)
        return _generators[value]

    cm: ContextManager[None] = value
    key = type(cm).__qualname__
    if key not in _generators:
        # Add anonymous context manager if not existing yet.
        def _anon_context(
            _: torch.nn.Module | None, _cm: ContextManager[None] = cm
        ) -> ContextManager[None]:
            return _cm

        register_generator(FlowGenerator(key, _anon_context, order=0))
    return _generators[key]


@attrs.define(frozen=True)
class DataFlow(abc.ABC):
    """One data requirement of the layer being optimized.

    Args:
        generator: The flow generator defining execution context for this data.
        cache: Whether the work done to produce this data may be reused. The value
            reaches `CallModule.cache` as a hint for the consumer. It does not move
            a call, and it does not keep a register entry alive: the lifetime pass
            frees every entry after its last reader either way.
    """

    generator: FlowGenerator = attrs.field(converter=_to_generator)
    cache: bool = True

    def __attrs_post_init__(self) -> None:
        """Reject a requirement that no model state can satisfy.

        A pinned generator shows the weights from before every mutation. To produce
        that data more than once, the scheduler would need a copy of the original
        weights, which is not implemented yet. So a pinned flow must cache.

        Raises:
            NotImplementedError: If a pinned flow declares `cache=False`.
        """
        if self.generator.pinned and not self.cache:
            msg = (
                f"The {self.generator.key!r} flow generator is pinned, so cache=False "
                f"requires a copy of the model, which is not implemented yet."
            )
            raise NotImplementedError(msg)


@attrs.define(frozen=True, repr=False)
class InputActivations(DataFlow):
    """The activations arriving at the region's input boundary."""


@attrs.define(frozen=True, repr=False)
class OutputActivations(DataFlow):
    """The activations leaving the region's output boundary."""


def _to_sources(sources: Mapping[str, Any]) -> Mapping[str, Any]:
    """Freeze the source mapping owned by a `Gradient`, because its hash reads it."""
    return MappingProxyType(sources)


def _sources_key(sources: Mapping[str, Any]) -> frozenset[tuple[str, Any]]:
    """Compare `ModelOutputs` sources by value and other sources by identity.

    Other sources can be tensors and data loaders, which have no reliable equality.
    """
    return frozenset(
        (name, value if isinstance(value, ModelOutputs) else id(value))
        for name, value in sources.items()
    )


@attrs.define(frozen=True)
class ModelOutputs:
    """The graph outputs of the model, under one flow generator.

    Args:
        generator: The flow generator the outputs are produced under.
    """

    generator: FlowGenerator = attrs.field(converter=_to_generator)


@attrs.define(frozen=True, repr=False, init=False)
class Gradient(DataFlow):
    """The gradient of a scalar loss with respect to the output of the region.

    The scheduler calls `f(**sources)` one time per run. `f` iterates over the
    batches and yields one scalar loss per batch. For each batch, the region
    receives dL/dy, with y the output of the region.

    The backward pass goes through the model under the generator of the
    `ModelOutputs` sources. That generator becomes the generator of this flow,
    so all `ModelOutputs` sources must use the same generator.

    Regions that read equal gradients share one call of `f` and one backward pass.
    Two gradients are equal when they have the same `f`, equal `ModelOutputs`
    sources, and the same objects as other sources.

    The backward pass runs each node a second time, so the model must be in eval mode.
    A functional call, like `F.dropout(x, p, training=...)`, keeps the `training` value
    it got when the graph was made. If it fixes `training=True`, the gradients do not
    match the forward pass. This is not supported.

    Args:
        f: The loss function. It yields one real scalar loss per batch.
        **sources: The keyword arguments of `f`. A `ModelOutputs` source becomes
            a series of model outputs, one item per batch. Other values (e.g. a
            label data loader) go to `f` without change.
    """

    f: Callable[..., Iterable[torch.Tensor]] = attrs.field(
        kw_only=True, validator=attrs.validators.is_callable()
    )
    sources: Mapping[str, Any] = attrs.field(kw_only=True, converter=_to_sources, eq=_sources_key)

    def __init__(self, f: Callable[..., Iterable[torch.Tensor]], /, **sources: Any) -> None:
        generators = {
            source.generator for source in sources.values() if isinstance(source, ModelOutputs)
        }
        if len(generators) != 1:
            keys = sorted(generator.key for generator in generators)
            msg = f"Gradient must read ModelOutputs of one generator, but reads {keys}."
            raise ValueError(msg)
        self.__attrs_init__(generators.pop(), f=f, sources=sources)

    def __attrs_post_init__(self) -> None:
        """Reject a gradient that the scheduler cannot produce.

        Raises:
            NotImplementedError: If the generator is not pinned.
        """
        super().__attrs_post_init__()
        if not self.generator.pinned:
            msg = (
                f"Flow of {self.generator.key!r} is not pinned, but gradients are only "
                f"implemented for pinned generators."
            )
            raise NotImplementedError(msg)

    @property
    def outputs(self) -> dict[str, ModelOutputs]:
        """The sources that read model outputs, in declaration order."""
        return {
            name: value for name, value in self.sources.items() if isinstance(value, ModelOutputs)
        }

    @property
    def values(self) -> dict[str, Any]:
        """The sources that reach `f` unchanged."""
        return {
            name: value
            for name, value in self.sources.items()
            if not isinstance(value, ModelOutputs)
        }
