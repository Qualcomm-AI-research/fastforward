# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import fastforward as ff
import pytest
import torch

from fastforward._autoquant.superclass import (
    _quantizable_superclass,
    ancestor_chain,
    method_provider_chain,
)


class _Parent(torch.nn.Module):
    """Plain base module without a hand-written quantized counterpart."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def helper(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @classmethod
    def build(cls) -> "_Parent":
        return cls()


class _Child(_Parent):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)


class _MroSkipChild(_Child):
    """Deliberately skips a level of the MRO by anchoring to an ancestor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super(_Child, self).forward(x)  # noqa: UP008


class _MroSkipGrandchild(_MroSkipChild):
    """Deliberately skips two levels of the MRO by anchoring further up."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super(_Child, self).forward(x)  # noqa: UP008


class _Mixin:
    """Non-module mixin that precedes the module base in the MRO."""


class _MixinChild(_Mixin, _Parent):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)


class _ModuleChild(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)  # type: ignore[no-any-return]


class _LinearChild(torch.nn.Linear):
    """Subclasses a module that has a hand-written `QuantizedLinear` counterpart."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)


# Test: _quantizable_superclass


@pytest.mark.parametrize(
    "module_type, expected",
    [
        pytest.param(_Child, _Parent, id="immediate_module_base"),
        pytest.param(_MroSkipChild, _Child, id="nearest_base_not_furthest"),
        pytest.param(_MixinChild, _Parent, id="skips_non_module_mixin"),
        pytest.param(_LinearChild, torch.nn.Linear, id="torch_builtin_base"),
    ],
)
def test_quantizable_superclass_returns_first_module_in_mro(
    module_type: type[torch.nn.Module], expected: type[torch.nn.Module]
) -> None:
    # GIVEN a module type with at least one quantizable module in its MRO

    # WHEN its quantizable superclass is resolved
    superclass = _quantizable_superclass(module_type)

    # THEN the first torch.nn.Module after the type itself is returned, which is
    # what a super() call inside that type resolves against
    assert superclass is expected


@pytest.mark.parametrize(
    "module_type",
    [
        pytest.param(_Parent, id="direct_module_subclass"),
        pytest.param(_ModuleChild, id="super_targets_module_itself"),
        pytest.param(ff.nn.QuantizedLinear, id="already_a_quantized_module"),
    ],
)
def test_quantizable_superclass_returns_none(module_type: type[torch.nn.Module]) -> None:
    # GIVEN a module type whose nearest module superclass is not a meaningful
    # quantization target -- either torch.nn.Module itself or an existing
    # QuantizedModule

    # WHEN its quantizable superclass is resolved
    superclass = _quantizable_superclass(module_type)

    # THEN no superclass is selected
    assert superclass is None


# Test: ancestor_chain


@pytest.mark.parametrize(
    "module_type, anchor_name, expected",
    [
        pytest.param(_MroSkipChild, "_Child", [_Child], id="skips_one_level"),
        pytest.param(_MroSkipGrandchild, "_Child", [_MroSkipChild, _Child], id="skips_two_levels"),
        pytest.param(_Child, "_Parent", [_Parent], id="explicit_immediate_superclass"),
    ],
)
def test_ancestor_chain_returns_segment_up_to_anchor(
    module_type: type[torch.nn.Module],
    anchor_name: str,
    expected: list[type[torch.nn.Module]],
) -> None:
    # GIVEN a module type and the name of an ancestor named explicitly in one of
    # its methods' `super(Ancestor, self)` calls

    # WHEN the MRO segment leading up to and including that ancestor is resolved
    chain = ancestor_chain(module_type, anchor_name)

    # THEN the segment starts at the immediate superclass and ends at the anchor
    assert chain == expected


def test_ancestor_chain_returns_none_for_unknown_anchor() -> None:
    # GIVEN an anchor name that does not name any ancestor in the module's MRO

    # WHEN the MRO segment leading up to it is resolved
    chain = ancestor_chain(_MroSkipChild, "NotAnAncestor")

    # THEN no segment is found
    assert chain is None


# Test: method_provider_chain


@pytest.mark.parametrize(
    "module_type, anchor_type, method_name, expected",
    [
        pytest.param(_MroSkipChild, _Child, "forward", [_Parent], id="provider_is_next_ancestor"),
        pytest.param(
            _MroSkipGrandchild,
            _MroSkipChild,
            "helper",
            [_Child, _Parent],
            id="spans_ancestors_without_own_method",
        ),
    ],
)
def test_method_provider_chain_reaches_the_providing_ancestor(
    module_type: type[torch.nn.Module],
    anchor_type: type[torch.nn.Module],
    method_name: str,
    expected: list[type[torch.nn.Module]],
) -> None:
    # GIVEN a `super(Anchor, self).method()` call, which resolves past `Anchor`
    # itself to whichever later ancestor defines `method`

    # WHEN the ancestors after the anchor up to that provider are resolved
    chain = method_provider_chain(module_type, anchor_type, method_name)

    # THEN the segment ends at the first ancestor defining the method itself,
    # so the generated counterpart chain can reach a quantized implementation
    assert chain == expected


@pytest.mark.parametrize(
    "anchor_type, method_name",
    [
        pytest.param(_Parent, "forward", id="only_torch_module_remains"),
        pytest.param(_Parent, "not_a_method", id="no_ancestor_defines_it"),
    ],
)
def test_method_provider_chain_is_empty_without_quantizable_provider(
    anchor_type: type[torch.nn.Module], method_name: str
) -> None:
    # GIVEN an anchor after which no quantizable ancestor provides the method --
    # the search reaches `torch.nn.Module` first

    # WHEN the provider chain is resolved
    chain = method_provider_chain(_Child, anchor_type, method_name)

    # THEN no segment is reported, so the callsite is left for the caller to
    # handle rather than re-anchored onto a class that cannot provide it
    assert chain == []
