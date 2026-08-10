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
from typing import Callable, ContextManager, TypeAlias

import attrs
import torch

import fastforward as ff


@attrs.define(frozen=True, eq=False)
class FlowGenerator:
    """Defines the execution context and scheduling priority of a data flow.

    Args:
        key: Unique identifier for this generator.
        context: Factory that produces the context manager for the forward pass.
        priority: Scheduling order (lower runs first).
    """

    key: str
    context: Callable[[torch.nn.Module], ContextManager[None]]
    priority: int


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


# Run with quantization disabled; produces baseline (unquantized) activations.
ORIGINAL = register_generator(
    FlowGenerator("original", lambda m: _disable_quantization(m), priority=5)
)

# Run with the model as-is; produces activations reflecting all mutations so far.
QUANTIZED = register_generator(FlowGenerator("quantized", lambda _: nullcontext(), priority=10))

# No constraints on the model state; default for a plain forward pass.
ANY = register_generator(FlowGenerator("any", lambda _: nullcontext(), priority=0))


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

        register_generator(FlowGenerator(key, _anon_context, priority=0))
    return _generators[key]


@attrs.define(frozen=True)
class DataFlow(abc.ABC):
    """One data requirement of the layer being optimized.

    Args:
        generator: The flow generator defining execution context for this data.
        cache: Whether the work done to produce this data may be reused.
    """

    generator: FlowGenerator = attrs.field(converter=_to_generator)
    cache: bool = True


@attrs.define(frozen=True, repr=False)
class InputActivations(DataFlow):
    """The activations arriving at the region's input boundary."""


@attrs.define(frozen=True, repr=False)
class OutputActivations(DataFlow):
    """The activations leaving the region's output boundary."""


# Closed union of every concrete flow. Use this instead of `DataFlow` in type
# parameters that need exhaustiveness -- a `match` against `Flow` lets mypy
# flag any missed subclass through `assert_never`.
Flow: TypeAlias = InputActivations | OutputActivations
