# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import string

import pytest
import torch

from fastforward.mpath import Fragment, Selector, fragments, query, register_mpath_query_extension
from fastforward.mpath import _parser as parser
from fastforward.mpath import selector as selectors

tt = parser.TokenType


def assert_tokens(
    expected_types: list[tt], expected_raw: list[str], expected_positions: list[int], raw_query: str
) -> None:
    observed_tokens = list(parser._tokengen(raw_query))

    assert len(expected_types) == len(expected_raw)
    assert len(expected_types) == len(expected_positions)
    assert len(observed_tokens) - 1 == len(expected_types)  # EOL is not explicitly given
    for etype, raw, pos, etoken in zip(
        expected_types, expected_raw, expected_positions, observed_tokens
    ):
        assert etoken.kind is etype
        assert etoken.raw == raw
        assert etoken.position == pos


def test_tokenizer() -> None:
    ident = tt.IDENTIFIER
    types = [ident, tt.FORWARD_SLASH, ident, tt.FORWARD_SLASH, ident]
    raw_text = ["abc", "/", "def", "/", "qed"]
    positions = [0, 3, 4, 7, 8]
    assert_tokens(types, raw_text, positions, "abc/def/qed")

    types = [tt.ASTERISK, tt.FORWARD_SLASH, tt.ASTERISK, tt.ASTERISK]
    raw_text = ["*", "/", "*", "*"]
    positions = [0, 1, 2, 3]
    assert_tokens(types, raw_text, positions, "*/**")

    ident = tt.IDENTIFIER
    types = [
        ident,
        tt.FORWARD_SLASH,
        tt.TILDE,
        ident,
        tt.FORWARD_SLASH,
        ident,
        tt.FORWARD_SLASH,
        tt.DIGIT,
    ]
    raw_text = ["abc", "/", "~", "def", "/", "qed", "/", "0"]
    positions = [0, 3, 4, 5, 8, 9, 12, 13]
    assert_tokens(types, raw_text, positions, "abc/~def/qed/0")

    types = [
        ident,
        tt.FORWARD_SLASH,
        tt.LEFT_BRACKET,
        ident,
        tt.COLON,
        tt.IDENTIFIER,
        tt.RIGHT_BRACKET,
    ]
    raw_text = ["root", "/", "[", "cls", ":", "__something__", "]"]
    positions = [0, 4, 5, 6, 9, 10, 23]
    assert_tokens(types, raw_text, positions, "root/[cls:__something__]")


def _parse_fragment(raw: str) -> selectors.BaseSelector:
    return parser.parse(raw, context=parser.get_caller_context())


def test_fragment_parser() -> None:
    selector = _parse_fragment("/abc")
    assert isinstance(selector, selectors.Selector)
    assert isinstance(selector.fragment, fragments.PathFragment)
    assert selector.fragment.match("abc", torch.nn.Module())
    assert not selector.fragment.match("not_abc", torch.nn.Module())

    selector = _parse_fragment("abc")
    assert isinstance(selector, selectors.Selector)
    assert isinstance(selector.fragment, fragments.PathFragment)
    assert selector.fragment.match("abc", torch.nn.Module())
    assert not selector.fragment.match("not_abc", torch.nn.Module())

    selector = _parse_fragment("/~abc")
    assert isinstance(selector, selectors.Selector)
    assert isinstance(selector.fragment, fragments.InvertedFragment)
    assert not selector.fragment.match("abc", torch.nn.Module())
    assert selector.fragment.match("not_abc", torch.nn.Module())

    ModuleSubclass = type("ModuleSubclass", (torch.nn.Module,), {})
    selector = _parse_fragment("/[cls:ModuleSubclass]")
    assert isinstance(selector, selectors.Selector)
    assert isinstance(selector.fragment, fragments.ClassFragment)
    assert not selector.fragment.match("some_name", torch.nn.Module())
    assert selector.fragment.match("some_name", ModuleSubclass())

    selector = _parse_fragment("/[re:abc.*]")
    assert isinstance(selector, selectors.Selector)
    assert isinstance(selector.fragment, fragments.RegexPathFragment)
    assert selector.fragment.match("abc", torch.nn.Module())
    assert selector.fragment.match("abc_plus_more", torch.nn.Module())
    assert not selector.fragment.match("not_abc", torch.nn.Module())

    selector = _parse_fragment(r"/[re:abc[1-4\]]")
    assert isinstance(selector, selectors.Selector)
    assert isinstance(selector.fragment, fragments.RegexPathFragment)
    assert selector.fragment.match("abc1", torch.nn.Module())
    assert selector.fragment.match("abc4", torch.nn.Module())
    assert not selector.fragment.match("abc5", torch.nn.Module())

    with pytest.raises(parser.MPathParseError):
        _parse_fragment("/[re:[abc.*]]")


def _create_TestFragment(expected_raw_str: str) -> type[Fragment]:
    class TestFragment(Fragment):
        def match(self, fragment_name: str, module: torch.nn.Module) -> bool:
            """Matches a single fragment of a path on name or module.

            Args:
                fragment_name: The name of the path, corresponds to the attribute
                    name on the 'parent' object.
                module: The module that corresponds to the fragment name.

            Returns: Boolean indicating whether the fragment matches the current position
            """
            raise NotImplementedError(f"match is not implemented on {type(self).__name__}")

        @classmethod
        def from_raw_string(cls, raw_str: str) -> Selector:
            assert raw_str == expected_raw_str
            return Selector(None, cls())

    return TestFragment


def test_query_extension_raw() -> None:
    """Test if the extension text is forwarded exactly to the extension.

    I.e. in the query "[ext:<some text>]", the extension should receive "<some
    text>" exactly for all possible characters.
    """
    # ']' is part of string.printable, but '\' happens to be just before it such that it is
    # 'properly' escaped.
    TestFragment = _create_TestFragment(string.printable)
    with register_mpath_query_extension("testext", TestFragment):  # type: ignore[arg-type]
        # The following line will raise an MPathParseError instead of an AssertionError. The actual
        # assertion is printed as cause.
        query(f"[testext:{string.printable}]")


def test_parse() -> None:
    ModuleSubclass = type("ModuleSubclass", (torch.nn.Module,), {})
    selector = parser.parse("abc/[class:torch.nn.Module]/[cls:ModuleSubclass]/qed/")
    expected_fragments = [
        fragments.PathFragment("abc"),
        fragments.ClassFragment(torch.nn.Module),
        fragments.ClassFragment(ModuleSubclass),
        fragments.PathFragment("qed"),
    ]

    for expected, observed in zip(expected_fragments, selector.fragments()):
        assert expected.__class__ == observed.__class__
        assert expected.__dict__ == observed.__dict__

    selector = parser.parse("/abc/[class:torch.nn.Module]/[cls:ModuleSubclass]/qed")
    expected_fragments = [
        fragments.PathFragment("abc"),
        fragments.ClassFragment(torch.nn.Module),
        fragments.ClassFragment(ModuleSubclass),
        fragments.PathFragment("qed"),
    ]

    for expected, observed in zip(expected_fragments, selector.fragments()):
        assert expected.__class__ == observed.__class__
        assert expected.__dict__ == observed.__dict__

    with pytest.raises(parser.MPathParseError):
        parser.parse("[class:NonexistingClass]")
    with pytest.raises(parser.MPathParseError):
        parser.parse("[class:torch.NonexistingClass]")
