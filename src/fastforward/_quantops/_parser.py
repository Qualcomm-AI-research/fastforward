# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import dataclasses
import enum
import re

from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    NewType,
    Optional,
    Sequence,
    TypeAlias,
    TypeVar,
)

T = TypeVar("T")


class TokenizationError(Exception):
    pass


class ParseError(Exception):
    pass


class TokenKind(enum.Enum):
    Identifier = enum.auto()
    Digit = enum.auto()

    Pipe = enum.auto()
    Comma = enum.auto()
    Equals = enum.auto()
    Minus = enum.auto()
    Plus = enum.auto()
    ForwardSlash = enum.auto()
    BackSlash = enum.auto()
    Questionmark = enum.auto()
    Period = enum.auto()
    Colon = enum.auto()

    LeftParen = enum.auto()
    RightParen = enum.auto()
    LeftBracket = enum.auto()
    RightBracket = enum.auto()
    LeftAngle = enum.auto()
    RightAngle = enum.auto()
    LeftBrace = enum.auto()
    RightBrace = enum.auto()

    Arrow = enum.auto()
    EOL = enum.auto()


@dataclasses.dataclass(frozen=True)
class Token:
    source: str
    kind: TokenKind
    position: int
    len: int


class _Lexer:
    def __init__(self, src: str) -> None:
        self._src = src
        self._position = 0

        self._puncmap = {
            "|": TokenKind.Pipe,
            ",": TokenKind.Comma,
            ":": TokenKind.Colon,
            "=": TokenKind.Equals,
            "-": TokenKind.Minus,
            "+": TokenKind.Plus,
            "/": TokenKind.ForwardSlash,
            "\\": TokenKind.BackSlash,
            "?": TokenKind.Questionmark,
            ".": TokenKind.Period,
            "(": TokenKind.LeftParen,
            ")": TokenKind.RightParen,
            "[": TokenKind.LeftBracket,
            "]": TokenKind.RightBracket,
            "<": TokenKind.LeftAngle,
            ">": TokenKind.RightAngle,
            "{": TokenKind.LeftBrace,
            "}": TokenKind.RightBrace,
        }

    def _emit_punctiation(self) -> Token:
        position = self._position
        self._position += 1
        char = self._src[position]
        return Token(char, self._puncmap[char], position, 1)

    def _emit_identifier(self) -> Token:
        start = self._position
        char = self._src[self._position]
        while str.isalnum(char) or char == "_":
            self._position += 1
            if self._position == len(self._src):
                break
            char = self._src[self._position]
        return Token(
            self._src[start : self._position], TokenKind.Identifier, start, self._position - start
        )

    def _emit_digit(self) -> Token:
        start = self._position
        matcher = re.compile(r"[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?")
        if match := matcher.match(self._src[self._position :]):
            match_str = match.group()
            self._position += len(match_str)
            return Token(match_str, TokenKind.Digit, start, self._position)
        else:
            raise TokenizationError(f"Expected number at position {start}")

    def _peek_char(self) -> Optional[str]:
        if self._position + 1 < len(self._src):
            return self._src[self._position + 1]
        return None

    def __iter__(self) -> Iterator[Token]:
        while self._position < len(self._src):
            char = self._src[self._position]

            if str.isspace(char):
                self._position += 1
                continue
            if char == "-" and self._peek_char() == ">":
                yield Token("->", TokenKind.Arrow, self._position, 2)
                self._position += 2
                continue
            if char in self._puncmap:
                yield self._emit_punctiation()
                continue
            if char.isidentifier():
                yield self._emit_identifier()
                continue
            if char.isdigit():
                yield self._emit_digit()
                continue

            raise TokenizationError(
                f"Encountered unexpected character '{char}' at position {self._position}"
            )
        yield Token("<EOL>", TokenKind.EOL, len(self._src), 0)


class _Tokenizer:
    Checkpoint = NewType("Checkpoint", int)

    def __init__(self, tokengen: Iterable[Token]) -> None:
        self._tokengen = iter(tokengen)
        self._tokens: list[Token] = []
        self._position = 0

    def peek(self) -> Token:
        # Produce tokens from generator until token at position has been
        # generated. Return the last token in the stream if the total number of
        # tokens is less than self._position
        while self._position >= len(self._tokens):
            try:
                self._tokens.append(next(self._tokengen))
            except StopIteration:
                return self._tokens[-1]
        return self._tokens[self._position]

    def __next__(self) -> Token:
        return self.next()

    def next(self) -> Token:
        token = self.peek()
        self._position += 1
        return token

    def checkpoint(self) -> Checkpoint:
        return self.Checkpoint(self._position)

    def reset(self, checkpoint: Checkpoint) -> None:
        self._position = checkpoint

    def _pos_repr(self) -> str:
        before = " ".join(tok.source for tok in self._tokens[: self._position])
        after = " ".join(tok.source for tok in self._tokens[self._position :])
        return f"{before} ** {after}"


def tokenizer(src: str) -> _Tokenizer:
    return _Tokenizer(_Lexer(src))


def _default_production(*args: T) -> T | list[T]:
    if len(args) == 1:
        return args[0]
    return list(args)


@dataclasses.dataclass(frozen=True)
class ParseRule:
    name: str
    _expressions: dataclasses.InitVar[Sequence["ParseExpression | str"]]
    expressions: tuple["ParseExpression", ...] = dataclasses.field(init=False)
    production: Callable[..., Any] = _default_production
    left_recursive: bool = dataclasses.field(init=False, default=False)

    def __post_init__(self, expressions: Sequence["ParseExpression | str"]) -> None:
        def _make_expression(exp: "ParseExpression | str") -> ParseExpression:
            if not isinstance(exp, ParseExpression):
                return ParseExpression(exp)
            return exp

        object.__setattr__(self, "expressions", tuple(map(_make_expression, expressions)))

    def set_left_recursive(self) -> None:
        object.__setattr__(self, "left_recursive", True)


@dataclasses.dataclass(frozen=True)
class _ParseExp:
    target: str
    optional: bool = False
    remove: bool = False
    token: Optional[TokenKind] = None

    def __post_init__(self) -> None:
        target = self.target
        if target.startswith("[") and target.endswith("]"):
            object.__setattr__(self, "optional", True)
            target = target[1:-1].strip()
        if target.startswith("!"):
            object.__setattr__(self, "remove", True)
            target = target[1:].strip()
        if target.isupper():
            for kind in TokenKind:
                if kind.name.upper() == target:
                    object.__setattr__(self, "token", kind)
                    break
            else:
                raise ParseError(f"Unable to create parser, {target} is not a valid token kind")

        object.__setattr__(self, "target", target)


@dataclasses.dataclass(frozen=True)
class ParseExpression:
    expression: dataclasses.InitVar[str]
    parts: tuple[_ParseExp, ...] = dataclasses.field(init=False)

    def __post_init__(self, expression: str) -> None:
        parts = tuple(_ParseExp(exp) for exp in expression.split())
        object.__setattr__(self, "parts", parts)


class Sentinel:
    _sentinels: dict[str, "Sentinel"] = {}
    _name: str

    def __new__(cls, name: str) -> "Sentinel":
        if name not in cls._sentinels:
            obj = super().__new__(cls)
            obj._name = name
            cls._sentinels[name] = obj
        return cls._sentinels[name]

    def __repr__(self) -> str:
        return f"<{self._name.upper()}>"


MISSING = Sentinel("MISSING")
SKIP = Sentinel("SKIP")


CP: TypeAlias = _Tokenizer.Checkpoint


class Parser:
    """
    Simple PEG-based parser.

    The grammar is expected to be stored in rules. From this, the rules are
    inspecting for left recursion, which is handles accordingly during parsing.

    A rule with the name start is expected and parsing will commence from this
    rule. The parser is initialized using a tokenizer, but will only parse
    by calling the `parse` method. After this, the parser/tokenizer are in a
    consumed state and will not parse the input again. Calling `parse` more than
    once results in undefined behaviour.

    If the parse fails, a ParseError is raised with a description of the error,
    if possible. Otherwise, the production rules of the grammar define the
    return type of the parse function.

    Args:
        tokenizer: Tokenizer to be used during parsing
    """

    rules: tuple[ParseRule, ...] = ()

    _left_recursive_memo: dict[tuple[ParseRule, CP], tuple[CP, Any]]
    _latest_error: tuple[_Tokenizer.Checkpoint, TokenKind | None, Token | None]

    def __init__(self, tokenizer: _Tokenizer) -> None:
        self._tokenizer = tokenizer
        self._rule_table = self._build_rule_table()
        self._left_recursive_memo = {}
        self._latest_error = (_Tokenizer.Checkpoint(0), None, None)

    def _build_rule_table(self) -> dict[str, ParseRule]:
        rule_table: dict[str, ParseRule] = {rule.name: rule for rule in self.rules}

        def _is_left_recursive(
            rule: ParseRule, target: str, _visited: tuple[str, ...] = ()
        ) -> bool:
            for expr in rule.expressions:
                part_target = expr.parts[0].target
                if expr.parts[0].token is not None:
                    continue
                if part_target in _visited:
                    return False
                visisted = _visited + (part_target,)
                if part_target == name or _is_left_recursive(
                    rule_table[part_target], target, visisted
                ):
                    return True
            return False

        for name, rule in rule_table.items():
            if _is_left_recursive(rule, name):
                rule_table[name].set_left_recursive()

        return rule_table

    def _next_token(self) -> Token:
        return next(self._tokenizer)

    def _peek_token(self) -> Token:
        return self._tokenizer.peek()

    def _checkpoint(self) -> _Tokenizer.Checkpoint:
        return self._tokenizer.checkpoint()

    def _reset(self, checkpoint: _Tokenizer.Checkpoint) -> None:
        self._tokenizer.reset(checkpoint)

    def parse(self) -> Any:
        if result := self._parse_rule(self._rule_table["start"]):
            return result
        error_pos, expected, observed = self._latest_error
        if expected is None or observed is None:
            raise ParseError("parsing failed with unknown error")
        else:
            raise ParseError(
                f"Expected '{expected.name}' but found '{observed.kind.name}' "
                f"({observed.source}) at position {error_pos}"
            )

    def _parse_rule(self, rule: ParseRule, _maybe_left_recursive: bool = True) -> Any:
        if _maybe_left_recursive and rule.left_recursive:
            return self._parse_lr_rule(rule)

        for expression in rule.expressions:
            if result := self._parse_expression(expression):
                return rule.production(*result)

        return None

    def _parse_lr_rule(self, rule: ParseRule) -> Any:
        checkpoint = self._checkpoint()
        memo_key = (rule, checkpoint)
        if memo_key in self._left_recursive_memo:
            last_checkpoint, result = self._left_recursive_memo[memo_key]
            self._reset(last_checkpoint)
        else:
            last_checkpoint, result = self._left_recursive_memo[memo_key] = checkpoint, None
            while True:
                self._reset(checkpoint)
                last_result = self._parse_rule(rule, _maybe_left_recursive=False)

                if (cp := self._checkpoint()) <= last_checkpoint:
                    break
                result = last_result
                last_checkpoint = cp
                self._left_recursive_memo[memo_key] = (last_checkpoint, result)

            self._reset(last_checkpoint)

        return result

    def _parse_expression(self, expression: ParseExpression) -> list[Any] | None:
        result: list[Any] = []
        checkpoint = self._checkpoint()
        for exp_part in expression.parts:
            if exp_result := self._parse_expression_part(exp_part):
                if exp_result is not SKIP:
                    result.append(exp_result)
            else:
                self._reset(checkpoint)
                return None
        return result

    def _parse_expression_part(self, exp: _ParseExp) -> Any:
        if exp.token is None:
            alternative_result = MISSING if exp.optional else None
            return self._parse_rule(self._rule_table[exp.target]) or alternative_result
        elif self._peek_token().kind == exp.token:
            token = self._next_token()
            return token if not exp.remove else SKIP
        elif exp.optional:
            return MISSING if not exp.remove else SKIP
        else:
            # This parse expression part failed. We report the latest observed
            # error to the user as this corresponds to the longest failing
            # parse. Update error information if this error occured later in
            # the input string than any previous error.
            error_checkpoint, _, _ = self._latest_error
            if error_checkpoint < (cp := self._checkpoint()):
                self._latest_error = (cp, exp.token, self._peek_token())
            return None


# Production Helpers


def as_list(type_: Any, list_: Optional[list[Any]] = None) -> list[Any]:
    if not isinstance(type_, (list)):
        return [type_] + (list_ or [])
    return type_ + (list_ or [])
