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

Kinds of flow today: `InputActivations`, `OutputActivations`.
"""

import abc

from contextlib import nullcontext
from typing import Callable, ContextManager

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
    context: Callable[[torch.nn.Module], ContextManager[None]]
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


def _disable_quantization(module: torch.nn.Module) -> ContextManager[None]:
    return ff.disable_quantization(module)


def _quantized_context(_: torch.nn.Module) -> ContextManager[None]:
    return nullcontext()


def _any_context(_: torch.nn.Module) -> ContextManager[None]:
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
            _: torch.nn.Module, _cm: ContextManager[None] = cm
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
