# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Fuse QDQ (grid-snapped) values into a model's weights.

This module provides a standalone, no-I/O transform that replaces a model's
weight parameters with their quantize-dequantize ("QDQ", grid-snapped) values
and optionally converts the corresponding weight quantizers into stubs.

The transform is used directly by algorithms and export paths that need
grid-snapped weights without persisting anything to disk (e.g. GPTQ,
orchestration, export). It is also the primitive on top of which
`QuantizedModule.save_quantized_model` is built.

Discovery of which quantizer snaps which weight is factored behind the
`WeightQuantizerDiscovery` protocol. The default `ConventionDiscovery` relies on
the FastForward naming/tagging convention (a module exposing a `weight`
parameter alongside a `weight`-tagged `weight_quantizer`). A future
forward-pass-based strategy can be substituted without changing this module's
callers.
"""

from typing import Iterator, Protocol, runtime_checkable

import torch

import fastforward as ff

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
        for name, sibling in module.named_children():
            if sibling is quantizer:
                setattr(module, name, QuantizerStub(_metadata=quantizer.quant_metadata))
                break


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
    `QuantizedModule.save_quantized_model`.

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
    """
    discovery = ConventionDiscovery() if discovery is None else discovery
    for module, weight_attr, quantizer in list(discovery(model)):
        _fuse_target(module, weight_attr, quantizer, stub_quantizer=stub_quantizers)
