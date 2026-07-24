# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Fuse QDQ (grid-snapped) values into a model's weights.

This module provides a standalone, no-I/O transform that replaces a model's
weight parameters with their quantize-dequantize ("QDQ", grid-snapped) values
and optionally converts the corresponding weight quantizers into stubs.

The transform is used directly by algorithms and export paths that need
grid-snapped weights without persisting anything to disk (e.g. GPTQ,
orchestration, export). It is also the primitive on top of which
`fastforward.quantization.save_load.save_quantized_model` is built.

Discovery of which quantizer snaps which weight is factored behind the
`WeightQuantizerDiscovery` protocol. The default `ConventionDiscovery` relies on
the FastForward naming/tagging convention (a module exposing a `weight`
parameter alongside a `weight`-tagged `weight_quantizer`). A future
forward-pass-based strategy can be substituted without changing this module's
callers.
"""

from collections import defaultdict
from typing import Iterator, Protocol, runtime_checkable

import torch

import fastforward as ff

from fastforward.exceptions import QuantizationError
from fastforward.nn.quantizer import Quantizer, QuantizerStub

# A single weight-fuse target: the module owning the weight, the attribute name
# of the weight parameter on that module, and the quantizer that snaps it.
WeightQuantizerTarget = tuple[torch.nn.Module, str, Quantizer]


@runtime_checkable
class WeightQuantizerDiscovery(Protocol):
    """Strategy that yields the weight quantizers to fuse in a model.

    Implementations return an iterator of `(module, weight_attr, quantizer)`
    triples, where `getattr(module, weight_attr)` is the weight parameter to be
    replaced by its QDQ values and `quantizer` is the (non-stub) quantizer that
    produces those values.
    """

    def __call__(self, model: torch.nn.Module) -> Iterator[WeightQuantizerTarget]:
        """Yield `(module, weight_attr, quantizer)` targets for `model`."""
        ...


class ConventionDiscovery:
    """Discover weight quantizers by the FastForward naming/tagging convention.

    Uses `find_quantizers` with an mpath tag query to locate initialized
    quantizers, then resolves the owning module and weight parameter via the
    `FilterResult.parent` and a configurable weight attribute name.

    This relies on no forward pass and matches how quantized modules such as
    `QuantizedLinear`, `QuantizedConv{1,2}d` and `QuantizedEmbedding` wire their
    weight quantizers.

    Args:
        weight_attr: Attribute name of the weight parameter on the parent module.
        tag: The quantizer tag to query for (e.g. ``"parameter/weight"`` or
            ``"parameter/bias"``). Defaults to ``"parameter/weight"``.
    """

    def __init__(self, weight_attr: str = "weight", *, tag: str = "parameter/weight") -> None:
        self._weight_attr = weight_attr
        self._tag = tag

    def __call__(self, model: torch.nn.Module) -> Iterator[WeightQuantizerTarget]:
        """Yield weight-quantizer targets discovered in `model`."""
        for result in ff.find_quantizers(model, f"**/[quantizer:{self._tag}]"):
            quantizer = result.module
            assert isinstance(quantizer, Quantizer)

            if quantizer.is_stub():
                continue

            parent = result.parent
            weight = getattr(parent, self._weight_attr, None)
            if not isinstance(weight, torch.nn.Parameter):
                continue

            yield parent, self._weight_attr, quantizer


def _fuse_target(
    module: torch.nn.Module,
    weight_attr: str,
    quantizer: Quantizer,
    *,
    stub_quantizer: bool,
) -> None:
    """Snap a single weight to its QDQ values in-place, optionally stubbing.

    Args:
        module: The module owning the weight parameter.
        weight_attr: Attribute name of the weight parameter on `module`.
        quantizer: The quantizer producing the QDQ values.
        stub_quantizer: If True, replace the weight quantizer with a
            `QuantizerStub` after snapping the weight.
    """
    weight = getattr(module, weight_attr)
    # Disable strict quantization: the quantizer returns a QuantizedTensor which
    # we immediately dequantize to a regular tensor.
    with ff.strict_quantization(False):
        qdq_weight = quantizer(weight).dequantize()

    if qdq_weight is weight:
        # The quantizer was a no-op (e.g. disabled); nothing to fuse or stub.
        return

    with torch.no_grad():
        weight.copy_(qdq_weight)

    if stub_quantizer:
        _stub_target(module, quantizer)


def _stub_target(module: torch.nn.Module, quantizer: Quantizer) -> None:
    """Replace `quantizer` on `module` with an equivalent `QuantizerStub`.

    The stub inherits the original quantizer's metadata, so tags and shape
    information survive the replacement.

    Args:
        module: The module owning the quantizer.
        quantizer: The quantizer instance to replace.
    """
    for name, sibling in module.named_children():
        if sibling is quantizer:
            setattr(module, name, QuantizerStub(_metadata=quantizer.quant_metadata))
            break


def _check_tied_weight_targets(
    model: torch.nn.Module, targets: list[WeightQuantizerTarget]
) -> None:
    """Reject tied weights whose quantizers would snap them to different grids.

    Fusing walks each target independently and writes into the weight in place.
    When two modules share one weight `Parameter` (e.g. a tied
    `lm_head.weight`/`embed_tokens.weight` pair) but own separately configured
    quantizers, the second write silently overwrites the first, leaving the
    weight on whichever grid happened to be fused last. The result is wrong for
    at least one of the two modules, so refuse rather than corrupt the weight.

    Tied weights sharing a single quantizer instance, or quantizers that produce
    identical QDQ values, are fused normally: the repeated write is a no-op.

    Args:
        model: The model being fused, used to report qualified module names.
        targets: The `(module, weight_attr, quantizer)` triples about to be fused.

    Raises:
        QuantizationError: If a tied weight is snapped by quantizers that
            disagree on the grid.
    """
    by_storage: dict[tuple[int, torch.Size], list[WeightQuantizerTarget]] = defaultdict(list)
    for module, weight_attr, quantizer in targets:
        weight = getattr(module, weight_attr)
        by_storage[(weight.data_ptr(), weight.shape)].append((module, weight_attr, quantizer))

    module_names = {id(submodule): name for name, submodule in model.named_modules()}

    for group in by_storage.values():
        if len(group) == 1:
            continue

        # Compare the actual QDQ output rather than the quantizer configuration:
        # differently-configured quantizers may still agree on the grid, and only
        # a disagreement corrupts the weight.
        weight = getattr(group[0][0], group[0][1])
        reference: torch.Tensor | None = None
        for module, weight_attr, quantizer in group:
            with ff.strict_quantization(False):
                qdq_weight = quantizer(weight).dequantize()
            if reference is None:
                reference = qdq_weight
            elif not torch.equal(reference, qdq_weight):
                names = ", ".join(
                    f"{module_names.get(id(module), type(module).__name__)}.{weight_attr}"
                    for module, weight_attr, _ in group
                )
                msg = (
                    f"Cannot fuse QDQ weights: the weight shared by [{names}] is tied "
                    "across modules whose weight quantizers snap it to different grids, "
                    "so fusing would leave it correct for at most one of them. Untie the "
                    "weights (assign each module its own copy of the parameter), or share "
                    "a single quantizer instance between them so both agree on the grid."
                )
                raise QuantizationError(msg)


def fuse_qdq_weights(
    model: torch.nn.Module,
    *,
    stub_quantizers: bool = False,
    discovery: WeightQuantizerDiscovery | None = None,
) -> None:
    """Replace a model's weights with their QDQ (grid-snapped) values in-place.

    For every discovered weight quantizer, the associated weight parameter is
    overwritten in-place with its quantize-dequantize value, so the model runs
    in float but matches its quantized numerics. No files are written; this is a
    pure in-memory transform intended for use by algorithms (e.g. GPTQ),
    orchestration, and export paths, as well as by
    `fastforward.quantization.save_load.save_quantized_model`.

    Args:
        model: The (quantized) model whose weights should be fused.
        stub_quantizers: If True, each fused weight quantizer is replaced by a
            `QuantizerStub`, so subsequent forward passes do not re-quantize the
            already grid-snapped weights. If False, the weight quantizers remain
            active. For idempotent affine quantizers (e.g. `LinearQuantizer`),
            re-quantizing a grid-snapped weight is a no-op as long as the
            quantization parameters are unchanged. Custom quantizers may not
            satisfy this property; use ``stub_quantizers=True`` when in doubt.
        discovery: Strategy used to find the weight quantizers to fuse. Defaults
            to `ConventionDiscovery`, which uses the FastForward naming/tagging
            convention. Provide a custom strategy to change how weight
            quantizers are discovered (e.g. a forward-pass-based approach).

    Note:
        Weights retain their original floating-point dtype but hold grid-snapped
        values. Only weights whose quantizer is discovered are affected;
        activation quantizers are left untouched.

    Raises:
        QuantizationError: If a weight is tied across modules whose quantizers
            would snap it to different grids. See `_check_tied_weight_targets`.
    """
    discovery = ConventionDiscovery() if discovery is None else discovery
    targets = list(discovery(model))
    _check_tied_weight_targets(model, targets)
    for module, weight_attr, quantizer in targets:
        _fuse_target(module, weight_attr, quantizer, stub_quantizer=stub_quantizers)


def find_weight_quantizers(
    model: torch.nn.Module,
    *,
    discovery: WeightQuantizerDiscovery | None = None,
) -> list[WeightQuantizerTarget]:
    """Return the weight-quantizer targets `fuse_qdq_weights` would act on.

    Exposes the discovery step on its own, so callers can inspect which weight
    quantizers a fuse or stub pass would affect without performing it (e.g. to
    apply an overwrite policy before stubbing).

    Args:
        model: The (quantized) model to inspect.
        discovery: Strategy used to find the weight quantizers. Defaults to
            `ConventionDiscovery`.

    Returns:
        The discovered `(module, weight_attr, quantizer)` triples.
    """
    discovery = ConventionDiscovery() if discovery is None else discovery
    return list(discovery(model))


def stub_weight_quantizers(
    model: torch.nn.Module,
    *,
    discovery: WeightQuantizerDiscovery | None = None,
) -> None:
    """Replace a model's weight quantizers with stubs, leaving weights untouched.

    This is the stubbing half of `fuse_qdq_weights` without the fusing: weights
    are not modified, only the discovered weight quantizers are replaced by
    equivalent `QuantizerStub`s (preserving their metadata).

    The primary use is loading an already-fused artifact: when weights were
    persisted as QDQ (grid-snapped) values, the weight quantizers must be
    stubbed so a subsequent forward pass does not re-quantize them.

    Args:
        model: The (quantized) model whose weight quantizers should be stubbed.
        discovery: Strategy used to find the weight quantizers. Defaults to
            `ConventionDiscovery`, which uses the FastForward naming/tagging
            convention.

    Note:
        Only weight quantizers are affected; activation quantizers are left
        untouched. Weights are never read or written by this function.
    """
    discovery = ConventionDiscovery() if discovery is None else discovery
    for module, _, quantizer in list(discovery(model)):
        _stub_target(module, quantizer)
