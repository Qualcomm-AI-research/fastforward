# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pytest
import torch

from fastforward import mpath
from fastforward.mpath.selector import MPathQueryError


class _MockFragment(mpath.Fragment):
    def match(self, fragment_name: str, module: torch.nn.Module) -> bool:
        return False


def test_selector_creation() -> None:
    s1, s2, s3 = (mpath.Selector(None, _MockFragment()) for _ in range(3))

    for sel in (s1 / s2 / s3, s1 / (s2 / s3), (s1 / s2) / s3, (s1 / s2).extends(s3)):
        assert isinstance(sel.next, mpath.Selector)
        assert isinstance(sel.next.next, mpath.Selector)
        assert sel.next.next.next is None

        assert sel.fragment is s1.fragment
        assert sel.next.fragment is s2.fragment
        assert sel.next.next.fragment is s3.fragment

    assert len(s1 / s2 / s3) == 3
    assert len(s1 / s2 / s3 / s1) == 4
    assert (s1 / s2 / s3)[0].fragment is s1.fragment  # type: ignore[attr-defined]
    assert (s1 / s2 / s3)[1].fragment is s2.fragment  # type: ignore[attr-defined]
    assert (s1 / s2 / s3)[2].fragment is s3.fragment  # type: ignore[attr-defined]

    observed = mpath.root / "abc" / "[cls:dict]" / "**"
    assert isinstance(observed_0 := observed[0], mpath.Selector)
    assert isinstance(observed_1 := observed[1], mpath.Selector)
    assert isinstance(observed_2 := observed[2], mpath.Selector)
    assert isinstance(observed_3 := observed[3], mpath.Selector)

    assert isinstance(observed_0.fragment, mpath.fragments.WildcardFragment)
    assert isinstance(observed_1.fragment, mpath.fragments.PathFragment)
    assert isinstance(observed_2.fragment, mpath.fragments.ClassFragment)
    assert isinstance(observed_3.fragment, mpath.fragments.WildcardFragment)


def test_selector_combinations() -> None:
    abc = mpath.query("/abc")
    xyz = mpath.query("/xyz")
    qed = mpath.query("/qed")

    assert isinstance(abc, mpath.Selector)
    assert isinstance(xyz, mpath.Selector)
    assert isinstance(qed, mpath.Selector)

    joint_selector = abc & qed
    assert isinstance(joint_selector, mpath.Selector)
    assert isinstance(joint_selector.fragment, mpath.fragments.JointFragment)
    assert joint_selector.fragment._selector_fragments == (abc.fragment, qed.fragment)

    compound1 = abc / qed
    compound2 = xyz / qed
    disjoint_selector = compound1 | compound2
    assert isinstance(disjoint_selector, mpath.selector.MultiSelector)
    assert disjoint_selector.selectors == (compound1, compound2)

    multi_selector = abc / qed
    with pytest.raises(MPathQueryError):
        _ = multi_selector & abc
    with pytest.raises(MPathQueryError):
        _ = abc & multi_selector


def test_aliases() -> None:
    aliases = mpath.aliases(alias1="base/first", alias2="&alias1/second")
    expected = {"alias1": mpath.query("base/first"), "alias2": mpath.query("base/first/second")}

    for k, query in aliases.items():
        query_expected = list(expected[k])
        for part, part_expected in zip(query, query_expected):
            assert isinstance(part, mpath.selector.Selector)
            assert isinstance(part_expected, mpath.selector.Selector)
            match part_expected.fragment:
                case mpath.fragments.PathFragment():
                    assert isinstance(part.fragment, mpath.fragments.PathFragment)
                    assert part.fragment._fragment_str == part_expected.fragment._fragment_str
                case mpath.fragments.WildcardFragment():
                    assert isinstance(part.fragment, mpath.fragments.WildcardFragment)
                    assert part.fragment.match_multiple == part_expected.fragment.match_multiple
                case _:
                    assert False, "Fragments do not match"
