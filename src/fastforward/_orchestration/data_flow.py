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

Kinds of flow today: `InputActivations`, `OutputActivations`. Both are
`ActivationsFlow`s
"""

import abc
import enum

from typing import Callable, NoReturn, TypeAlias

import attrs
import torch

from fastforward._orchestration.graph_module import Region


class FlowMode(enum.StrEnum):
    """Which model runs the pass that produces the data.

    The mode is also a signal to the scheduler. Data produced by a model that
    optimization does not touch stays valid once computed. Data produced by the
    model under optimization goes stale and must be produced again.

    Attributes:
        ORIGINAL: The model as it was before optimization started.
        QUANTIZED: The model with all optimization done so far.
    """

    ORIGINAL = "original"
    QUANTIZED = "quantized"

    @classmethod
    def _missing_(cls, value: object) -> NoReturn:
        # Catch spelling mistakes. 'quantised' vs 'quantized', etc. and
        # return the actual enum options (default error does not).
        modes = ", ".join(repr(str(m)) for m in cls)
        msg = f"Invalid flow mode {value!r}; expected one of {modes}."
        raise ValueError(msg)


# A resolver: names the module execution should start at, given the region under
# optimization. Resolved once a real graph exists. `None` means no bound: run
# every ancestor of the region.
FlowSource: TypeAlias = Callable[[Region], torch.nn.Module]


def _checked_source(source: object) -> FlowSource | None:
    """Return `source` as a resolver, or raise if it is not one.

    `source` is typed `object` because this is the boundary that takes input
    which has not been checked. A `torch.nn.Module` is callable, so it satisfies
    `FlowSource` structurally but is not a resolver: passing `model.fc1` where
    `lambda region: region.fc1` was meant fails here, rather than much later
    inside the graph walk.
    """
    if source is not None and (not callable(source) or isinstance(source, torch.nn.Module)):
        msg = (
            f"Invalid flow source {source!r}; expected a resolver "
            "(Callable[[Region], Module]) or None."
        )
        raise TypeError(msg)
    return source


@attrs.define(frozen=True)
class DataFlow(abc.ABC):
    """One data requirement of the layer being optimized.

    Concrete subclasses (`InputActivations`, `OutputActivations`, and future
    kinds like gradients or a derived flow) describe *how* that data is produced.

    Args:
        cache: Whether the work done to produce this data may be reused.
    """

    cache: bool = True


@attrs.define(frozen=True)
class ActivationsFlow(DataFlow):
    """Base for flows that collect forward activations at the region.

    `InputActivations` and `OutputActivations` share the nodes they need and
    their mode; they differ only in *where* on the region they collect.

    Args:
        mode: Which model produces the activations.
        source: A callable naming the module execution should start at. `None`
            (default) means unbounded: every ancestor of the region runs.
        cache: Whether the work done to produce this data may be reused.
    """

    mode: FlowMode = attrs.field(converter=FlowMode)
    source: FlowSource | None = attrs.field(default=None, converter=_checked_source, kw_only=True)
    cache: bool = True

    def __repr__(self) -> str:
        source = None if self.source is None else getattr(self.source, "__name__", "<resolver>")
        return f"{type(self).__name__}(mode={self.mode!r}, source={source}, cache={self.cache!r})"


@attrs.define(frozen=True, repr=False)
class InputActivations(ActivationsFlow):
    """The activations arriving at the region's input boundary."""


@attrs.define(frozen=True, repr=False)
class OutputActivations(ActivationsFlow):
    """The activations leaving the region's output boundary."""


# Closed union of every concrete flow. Use this instead of `DataFlow` in type
# parameters that need exhaustiveness -- a `match` against `Flow` lets mypy
# flag any missed subclass through `assert_never`.
Flow: TypeAlias = InputActivations | OutputActivations
