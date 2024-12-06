# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import dataclasses
import enum
import functools
import inspect

from operator import truediv
from typing import (
    Any,
    Callable,
    Generator,
    Iterator,
    Optional,
    Protocol,
    Sequence,
    TypeAlias,
    runtime_checkable,
)

from . import fragments, selector


class TokenType(enum.Enum):
    LEFT_BRACKET = enum.auto()
    RIGHT_BRACKET = enum.auto()
    LEFT_PAREN = enum.auto()
    RIGHT_PAREN = enum.auto()
    LEFT_BRACE = enum.auto()
    RIGHT_BRACE = enum.auto()
    STRING = enum.auto()
    DIGIT = enum.auto()
    IDENTIFIER = enum.auto()
    FORWARD_SLASH = enum.auto()
    BACKWARD_SLASH = enum.auto()
    COLON = enum.auto()
    COMMA = enum.auto()
    PERIOD = enum.auto()
    ELLIPSIS = enum.auto()
    TILDE = enum.auto()
    AMPERSAND = enum.auto()
    ASTERISK = enum.auto()
    CHARACTER = enum.auto()
    EOL = enum.auto()


@dataclasses.dataclass(frozen=True)
class Token:
    position: int
    len: int
    kind: TokenType
    raw: str


class MPathParseError(Exception): ...


def tokengen(src: str) -> Iterator[Token]:
    try:
        yield from _tokengen(src)
    except IndexError:
        raise MPathParseError("Unexpected end of line")


SINGLE_CHAR_TOKENS = {
    "(": TokenType.LEFT_PAREN,
    ")": TokenType.RIGHT_PAREN,
    "[": TokenType.LEFT_BRACKET,
    "]": TokenType.RIGHT_BRACKET,
    "{": TokenType.LEFT_BRACE,
    "}": TokenType.RIGHT_BRACE,
    ":": TokenType.COLON,
    ",": TokenType.COMMA,
    "~": TokenType.TILDE,
    "&": TokenType.AMPERSAND,
    "/": TokenType.FORWARD_SLASH,
    "\\": TokenType.BACKWARD_SLASH,
    "*": TokenType.ASTERISK,
}


def _tokengen(src: str) -> Iterator[Token]:
    pos = 0

    def increment(incr: int = 1) -> None:
        nonlocal pos
        pos += incr

    def token(start: int, end: int, kind: TokenType) -> Token:
        return Token(start, end - start - 1, kind, src[start:end])

    while pos < len(src):
        start = pos
        increment()

        char = src[start]
        if char.isspace():
            continue

        if kind := SINGLE_CHAR_TOKENS.get(char, None):
            yield token(start, pos, kind)
            continue

        if char == ".":
            if src[start : start + 3] == "...":
                yield token(start, start + 3, TokenType.ELLIPSIS)
            else:
                yield token(start, start + 1, TokenType.PERIOD)
            continue

        if char in "0123456789":
            while pos < len(src) and src[start : pos + 1].isdigit():
                increment()
            yield token(start, pos, TokenType.DIGIT)
            continue

        if char == '"':
            while not quote_delimited(src[start:pos]) and pos < len(src):
                increment()
            if not quote_delimited(src[start:pos]):
                # In this case, we did not find a string, return '"' as a
                # single character and reset pos
                pos = start + 1
                yield token(start, pos, TokenType.CHARACTER)

            yield token(start, pos, TokenType.STRING)
            continue

        if char.isidentifier():
            while pos < len(src) and src[start : pos + 1].isidentifier():
                increment()
            yield token(start, pos, TokenType.IDENTIFIER)
            continue

        yield token(start, pos, TokenType.CHARACTER)

    yield token(pos, pos, TokenType.EOL)


def quote_delimited(src: str) -> bool:
    return len(src) > 1 and src[0] == '"' and src[-1] == '"' and src[-2] != "\\"


class Tokenizer:
    def __init__(self, tokgen: Iterator[Token]) -> None:
        self.tokengen = tokgen
        self._peek_token: None | Token = None

    def peek(self) -> Token:
        if not self._peek_token:
            try:
                self._peek_token = next(self.tokengen)
            except StopIteration:
                raise MPathParseError("Encountered end of string before EOL token")
        return self._peek_token

    def next(self) -> Token:
        token = self.peek()
        if token.kind != TokenType.EOL:
            self._peek_token = None
        return token


@runtime_checkable
class MpathQueryExtension(Protocol):
    @classmethod
    def from_raw_string(cls, __str: str) -> selector.Selector:
        raise NotImplementedError()


@runtime_checkable
class MpathQueryContextualExtension(Protocol):
    @classmethod
    def from_raw_string_with_context(
        cls, __str: str, __context: dict[str, Any]
    ) -> selector.Selector:
        raise NotImplementedError()


_QueryExtension: TypeAlias = type[MpathQueryExtension] | type[MpathQueryContextualExtension]


class QueryExtensionContext:
    def __init__(self, tag: str) -> None:
        self._tag = tag

    def __enter__(self) -> Generator[None, None, None]:
        yield

    def __exit__(self, exc_type, exc_value, traceback):  # type: ignore[no-untyped-def]
        del Parser.fragment_extensions[self._tag]


def register_mpath_query_extension(
    tag: str, extension: _QueryExtension, override: bool = False
) -> QueryExtensionContext:
    r"""
    Register an MPath query extension.

    A query extension produces a SelectorFragment from a string. Once
    registered, it can be used in an MPath query string using "[<tag>:
    <input>]". Here input is passed to the extensions and can be any string.
    Note that ']' will need to be escaped using '\]' when used as part of the
    input string.

    Args:
        tag: Name of the query extension and tag it's made available under in
            MPath query strings.
        extension: The extension to register
        override: Boolean indicating wheter an existing extensions by the same
            name may be overridden.
    """
    if tag in Parser.fragment_extensions and not override:
        raise ValueError(
            f"fragment parser with tag '{tag}' already exists. Use override=True to override it"
        )

    if not issubclass(extension, (MpathQueryExtension, MpathQueryContextualExtension)):
        raise TypeError(
            f"Unable to register {extension.__name__} as mpath query extension because it "
            "does not satisfy the MpathQueryExtension protocol."
        )

    Parser.fragment_extensions[tag] = extension
    return QueryExtensionContext(tag=tag)


def mpath_query_extension(
    tag: str, override: bool = False
) -> Callable[[_QueryExtension], _QueryExtension]:
    r"""
    Decorator for creating an MPath query extensions.

    A query extension produces a SelectorFragment from a string. Once
    registered, it can be used in an MPath query string using "[<tag>:
    <input>]". Here input is passed to the extensions and can be any string.
    Note that ']' will need to be escaped using '\]' when used as part of the
    input string.

    Args:
        tag: Name of the query extension and tag it's made available under in
            MPath query strings.
        override: Boolean indicating wheter an existing extensions by the same
            name may be overridden.
    """

    def extension_wrapper(extension: _QueryExtension) -> _QueryExtension:
        register_mpath_query_extension(tag, extension, override)

        return extension

    return extension_wrapper


_QueryExtType: TypeAlias = type[MpathQueryExtension] | type[MpathQueryContextualExtension]


class Parser:
    fragment_extensions: dict[str, _QueryExtType] = {}

    def __init__(
        self, src: str, context: dict[str, Any], aliases: dict[str, selector.BaseSelector]
    ) -> None:
        self._src = src
        self._context = context
        self._aliases = aliases
        self.tokenizer = Tokenizer(tokengen(src))

    def consume(self, kind: TokenType) -> Token | None:
        if self.tokenizer.peek().kind == kind:
            return self.tokenizer.next()
        return None

    def consume_until(self, kind: TokenType, escape_tokens: Sequence[TokenType] = ()) -> Token:
        if kind in escape_tokens:
            raise ValueError(
                f"'{kind}' cannot be an escape token because it is already used as "
                f"sentinel, i.e, 'kind={kind}'"
            )

        escaped = False
        while True:
            token = self.tokenizer.peek()
            if token.kind == kind and not escaped:
                return self.tokenizer.next()
            elif token.kind == TokenType.EOL:
                raise MPathParseError(f"Expected '{kind.name}' but found end of line")
            elif token.kind in escape_tokens and not escaped:
                escaped = True
                self.tokenizer.next()
            else:
                escaped = False
                self.tokenizer.next()
        assert False  # unreachable

    def expect(self, *kinds: TokenType) -> Token:
        token = self.tokenizer.peek()
        if token.kind in kinds:
            return self.tokenizer.next()
        if len(kinds) == 1:
            err = (
                f"Expected {kinds[0].name} but found {token.kind.name} at position {token.position}"
            )
        else:
            expected = ", ".join(kind.name for kind in kinds)
            err = (
                f"Expect one of {expected} but found {token.kind.name} at position {token.position}"
            )
        raise MPathParseError(err)

    def parse(self) -> selector.BaseSelector:
        self.consume(TokenType.FORWARD_SLASH)  # Consume optional forward slash prefix
        selector_ = self._parse_selector()

        if not self.consume(TokenType.EOL):
            raise MPathParseError("Invalid MPath query")

        return selector_

    def _parse_selector(self) -> selector.BaseSelector:
        selector_fragments: list[selector.BaseSelector] = []
        while True:
            selector_fragments.append(self._parse_fragment())
            if not self.consume(TokenType.FORWARD_SLASH):
                break
            # If next char is EOL, then previous '/' was trailing and is
            # ignored
            if self.consume(TokenType.EOL):
                break

        if not selector_fragments:
            raise MPathParseError("Unable to parse MPath query")
        return functools.reduce(truediv, selector_fragments[1:], selector_fragments[0])

    def _parse_fragment(self) -> selector.BaseSelector:
        match (tok := self.tokenizer.peek()).kind:
            case TokenType.LEFT_BRACKET:
                return self._parse_extension()
            case TokenType.DIGIT | TokenType.IDENTIFIER:
                return self._parse_path_fragment()
            case TokenType.ASTERISK:
                self.consume(TokenType.ASTERISK)
                match_multiple = False
                if self.consume(TokenType.ASTERISK):
                    match_multiple = True
                fragment = fragments.WildcardFragment(match_multiple=match_multiple)
                return selector.Selector(None, fragment)
            case TokenType.LEFT_BRACE:
                return self._parse_multi_selector()
            case TokenType.TILDE:
                self.expect(TokenType.TILDE)
                return ~self._parse_fragment()
            case TokenType.AMPERSAND:
                return self._parse_alias()
            case _:
                raise MPathParseError(
                    f"Expected fragment at position {tok.position} but found '{tok.kind}"
                )

    def _parse_path_fragment(self) -> selector.BaseSelector:
        token = self.expect(TokenType.IDENTIFIER, TokenType.DIGIT)
        fragment = fragments.PathFragment(token.raw)
        return selector.Selector(None, fragment)

    def _parse_extension(self) -> selector.BaseSelector:
        self.expect(TokenType.LEFT_BRACKET)
        tag = self.expect(TokenType.IDENTIFIER).raw
        colon = self.expect(TokenType.COLON)
        closing_bracket = self.consume_until(
            TokenType.RIGHT_BRACKET, escape_tokens=[TokenType.BACKWARD_SLASH]
        )

        spec_str = self._src[colon.position + 1 : closing_bracket.position]
        if tag not in self.fragment_extensions:
            raise MPathParseError(f"'{tag}' is not a known mpath extension")

        extension = self.fragment_extensions[tag]
        try:
            if issubclass(extension, MpathQueryContextualExtension):
                selector = extension.from_raw_string_with_context(spec_str, self._context)
            else:
                selector = extension.from_raw_string(spec_str)
        except Exception as e:
            raise MPathParseError(
                f"Unable to instantiate MPATH extension '{extension.__name__}'"
            ) from e
        return selector

    def _parse_multi_selector(self) -> selector.MultiSelector:
        selectors: list[selector.BaseSelector] = []
        self.expect(TokenType.LEFT_BRACE)
        while True:
            selectors.append(self._parse_selector())
            match self.expect(TokenType.COMMA, TokenType.RIGHT_BRACE).kind:
                case TokenType.COMMA:
                    continue
                case TokenType.RIGHT_BRACE:
                    break
        return selector.MultiSelector(None, tuple(selectors))

    def _parse_alias(self) -> selector.BaseSelector:
        self.expect(TokenType.AMPERSAND)
        identifier = self.expect(TokenType.IDENTIFIER).raw
        if identifier not in self._aliases:
            raise MPathParseError(f"'{identifier}' is  not a known alias")
        return self._aliases[identifier]


def get_caller_context(stack_depth: int = 1) -> dict[str, Any]:
    """
    An MPath extension may need access to the caller context. This function
    returns a dictionary of the builtins, globals and locals available in the
    caller context. `stack_depth` determines how deep in the stack we loop for
    the context.
    """
    current_frame = inspect.currentframe()
    context: dict[str, Any] = {}
    try:
        for _ in range(stack_depth + 1):
            if current_frame is not None and current_frame.f_back is not None:
                current_frame = current_frame.f_back

        if current_frame is not None:
            context = {
                **current_frame.f_builtins,
                **current_frame.f_globals,
                **current_frame.f_locals,
            }
    finally:
        del current_frame

    return context


def parse(
    raw: str,
    context: Optional[dict[str, Any]] = None,
    aliases: Optional[dict[str, selector.BaseSelector]] = None,
) -> selector.BaseSelector:
    """
    Parse a raw query and produce a Selector
    """
    aliases = aliases or {}
    context = context or get_caller_context()
    return Parser(raw, context=context, aliases=aliases).parse()
