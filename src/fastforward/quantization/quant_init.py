# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import dataclasses

from typing import Any, Callable, Iterable, Literal, TypeAlias, cast

import torch

from typing_extensions import Self

from fastforward import mpath
from fastforward.exceptions import QuantizationError
from fastforward.mpath import Fragment, Selector, mpath_query_extension
from fastforward.mpath._search import FilterResult
from fastforward.nn.quantizer import Quantizer, QuantizerStub, Tag

_QuanttizerFactory: TypeAlias = Callable[[str, Quantizer], Quantizer]
_OverwriteOptions: TypeAlias = Literal["overwrite"] | Literal["skip"] | Literal["error"]


@mpath_query_extension("quantizer")
@mpath_query_extension("qtag")
class QuantizerTagSelectorFragment(Fragment):
    """Fragment that matches quantizer tags.

    Can be used in query string using `[quantizer:<tag>(,<tag>)+]` or
    "[qtag:<tag>(,<tag>)+]
    """

    def __init__(self, *tags: str | Tag):
        self._tags = tuple(Tag(tag) for tag in tags)

    def match(self, fragment_name: str, module: torch.nn.Module) -> bool:
        """Return True if module's tag metadata includes `self._tags`, False otherwise.

        Args:
            fragment_name: Unused
            module: The module that corresponds to the fragment name.

        Returns: Boolean indicating whether the fragment matches the tag set.
        """
        if not isinstance(module, Quantizer):
            return False
        if module.quant_metadata is None:
            return False
        for tag in self._tags:
            if tag not in module.quant_metadata:
                return False
        return True

    def __str__(self) -> str:
        tags = ", ".join(repr(tag) for tag in self._tags)
        return f"[quantizer:{tags}]"

    @classmethod
    def from_raw_string(cls, raw_str: str) -> Selector:
        """Create a `QuantizerTagSelectorFragment` from a string.

        Tags are expected to be comma separated. Any whitespace between tags
        is ignored.
        """
        tags = [Tag(tag.strip()) for tag in raw_str.split(",")]
        fragment = cls(*tags)
        return Selector(None, fragment)

    def __hash__(self) -> int:
        return hash(("__fragment_hash", type(self), self._tags))


def _initialize_quantizer(
    result: FilterResult,
    quantizer_factory: _QuanttizerFactory,
    overwrite_policy: _OverwriteOptions = "error",
    safe: bool = True,
) -> FilterResult:
    module = result.module
    if not isinstance(module, Quantizer):
        raise TypeError(f"'{result.full_name}' is not a quantizer.")
    if not isinstance(module, QuantizerStub):
        if overwrite_policy == "error":
            raise QuantizationError(
                f"'{result.full_name}' is a quantizer, but is already initialized. If "
                'you want to overwrite the existing quantizer, use overwrite_policy="overwrite" '
                "or if you want to skip re-initializing existing quantizers use "
                'overwrite_policy="skip"'
            )
        if overwrite_policy == "skip":
            return result
        if overwrite_policy != "overwrite":
            raise ValueError(
                f"Overwrite would occur, but overwrite_policy={repr(overwrite_policy)} is illegal. "
                'please use one of "overwrite", "skip", or "error"'
            )

    quantizer = quantizer_factory(result.full_name, cast(Quantizer, result.module))
    return result.update_module(quantizer, safe=safe)


def _ensure_factory_func(
    factory: type[Quantizer] | _QuanttizerFactory, kwargs: dict[str, Any]
) -> _QuanttizerFactory:
    if not isinstance(factory, type):
        return factory

    assert issubclass(factory, Quantizer)

    def quantizer_factory(_name: str, _existing: Quantizer) -> Quantizer:
        return factory(**kwargs)

    return quantizer_factory


class QuantizerCollection(mpath.MPathCollection):
    """MPathCollection that only stores Quantizer subclasses.

    On creation, all non quantizer results are filtered out from the results
    iterable. In contrast, append will throw an error when a non quantizer
    result is provided.

    Args:
        root: The root module that was used to produce the search results
        results: Optional list of search results, may be extended using append
    """

    def __init__(
        self, root: torch.nn.Module, results: Iterable[mpath.FilterResult] | None = None
    ) -> None:
        if results is not None:
            results = (r for r in results if isinstance(r.module, Quantizer))
        super().__init__(root=root, results=results)

    def append(self, item: FilterResult) -> None:
        """Add `FilterResult` to the collection.

        Raises an error if the filter result contains a module that is not a
        Quantizer.

        Args:
            item: Filter result to add to the collection
        """
        if not isinstance(item, Quantizer):
            raise ValueError(
                f"Can only insert a FilterResult of a Quantizer module to {type(self).__name__}"
            )
        super().append(item)

    def _initialize_with_factory(
        self,
        quantizer_factory: _QuanttizerFactory,
        overwrite_policy: _OverwriteOptions = "error",
        safe: bool = True,
    ) -> None:
        _results: list[FilterResult] = []
        for result in self._results:
            _results.append(
                _initialize_quantizer(
                    result=result,
                    quantizer_factory=quantizer_factory,
                    overwrite_policy=overwrite_policy,
                    safe=safe,
                )
            )
        self._results = _results

    def initialize(
        self,
        quantizer_factory: type[Quantizer] | Callable[[str, torch.nn.Module], Quantizer],
        *,
        overwrite_policy: _OverwriteOptions = "error",
        safe: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize all quantizers in the collection using `quantizer_factory`.

        `quantizer_factory` may either be a `Quantizer` subclass or a callable
        that accepts a `str` and `Quantizer`. In the latter case, the provided
        string is the full name relative to the root module and the quantizer
        is the quantizer that is currently referenced at this location. This
        may either be a `QuantizerStub` or an existing quantizer.

        Args:
            quantizer_factory: `Quantizer` or callable to initialize each
                quantizer in the collection. This factory is called for each
                element in the collection. If a `Quantizer` class is passed,
                all `kwargs` will be forwarded to the initializer. In the case
                of a callable, all `kwargs` are ignored.
            overwrite_policy: Either `"skip"`, `"overwrite"`, or `"error"`. The
                `overwrite_policy` indicates what to do when the element in the
                collection is already an initialized quantizer. In case of
                `"skip"` the initialization for that specific quantizer is
                skipped and nothing happens, for `"overwrite"` a new quantizer
                is created and the existing quantizer is overwritten. Lastly,
                in case of `"error"` an error is raised when it is attempted to
                re-initialize an already initialized quantizer.
            safe: Boolean that indicates what to do when the module was
                already replaced in the root module between the creation of
                this collection and the call to initialize. When safe is
                `True`, an error is raised. Otherwise, a (re-)initialization of
                the quantizer is attempted.
            **kwargs: Any extra keyword argument is passed to the `Quantizer`
                initializer.
        """
        self._initialize_with_factory(
            _ensure_factory_func(quantizer_factory, kwargs),
            overwrite_policy=overwrite_policy,
            safe=safe,
        )


def find_quantizers(
    root_module: torch.nn.Module,
    query: str | mpath.selector.BaseSelector,
    *,
    aliases: dict[str, mpath.selector.BaseSelector] | None = None,
) -> QuantizerCollection:
    """Find all quantizers in root_module that match query.

    Args:
        root_module: The module to search for quantizers.
        query: Mpath.Selector or str that represent the filter query.
            Please see the documentation of `fastforward.mpath.query` and
            `fastforward.mpath.search` for more details.
        aliases: Aliases to consider in the query. Any occurrence of `&<alias>`
            is replaced by the corresponding query in aliases.
    """
    if isinstance(query, str):
        query = mpath._parser.parse(
            query, context=mpath._parser.get_caller_context(), aliases=aliases
        )
    results = mpath.search(query, root_module)
    return QuantizerCollection(root_module, results)


@dataclasses.dataclass(frozen=True)
class _QuantConfigRule:
    """Represents a rule for quantization configuration.

    Attributes:
        query: The query to match quantizers.
        factory: The factory to create quantizers.
        kwargs: Additional keyword arguments for the factory.
    """

    query: mpath.selector.BaseSelector
    factory: type[Quantizer] | _QuanttizerFactory
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __repr__(self) -> str:
        if isinstance(self.factory, type) and issubclass(self.factory, Quantizer):
            kwargs = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
            factory_str = f"{self.factory.__name__}({kwargs})"
        else:
            factory_str = f"{self.factory.__name__}(<module_name>, <current_quantizer>)"
        return f"{str(self.query)} => {factory_str}"


@dataclasses.dataclass
class _RuleItem:
    """Represents an item containing a quantization rule and its filter result.

    Attributes:
        rule: The quantization rule.
        filter_result: The result of applying the rule's query.
        score: The precedence score of the rule.
    """

    rule: _QuantConfigRule
    filter_result: mpath.FilterResult
    score: int


class QuantizationConfig:
    """Manages quantization configuration rules and initializes quantizers in a model."""

    def __init__(self) -> None:
        self._rules: list[_QuantConfigRule] = []

    def add_rule(
        self,
        query: mpath.selector.BaseSelector | str,
        factory: type[Quantizer] | _QuanttizerFactory,
        **kwargs: Any,
    ) -> Self:
        """Add a quantizer rule to the collection.

        A rule consists of an mpath query and a quantizer factory. When a query
        matches a quantizer (stub) and has highest precedence, the factory is
        called to initialize a `Quantizer` (see the docstring of
        `precedence_score`).

        `quantizer_factory` may either be a `Quantizer` subclass or a callable
        that accepts a `str` and `Quantizer`. In the latter case, the provided
        string is the full name relative to the root module and the quantizer
        is the quantizer that is currently referenced at this location. This
        may either be a `QuantizerStub` or an existing (and initialized)
        quantizer.

        Args:
            query: Mpath.Selector or str that represent the filter query.
                Please see the documentation of `fastforward.mpath.query` and
                `fastforward.mpath.search` for more details.
            factory: `Quantizer` or callable to initialize each
                quantizer in the collection. This factory is called for each
                element in the collection. If a `Quantizer` class is passed,
                all `kwargs` will be forwarded to the initializer. In the case
                of a callable, all `kwargs` are ignored.
            factory: A function that creates a new quantizer. Either a `Quantizer` type
                or any object that implements `_QuantizerFactory`.
            **kwargs: Any extra keyword argument is passed to the `Quantizer`
                initializer.
        """
        if isinstance(query, str):
            query = mpath.query(query, context=mpath.local_context())
        self._rules.append(_QuantConfigRule(query, factory, kwargs))
        return self

    def precedence_score(self, rule: _QuantConfigRule, result: mpath.FilterResult) -> int:
        """Score precedence given a result.

        Returns an integer value that is used to choose between two matching
        rules. The rule with the highest score is selected. Ties are broken by
        order, i.e., select the rule that was added last. The precedence score
        is computer on a per result basis, this means that the precedence can
        differ per module.

        Args:
            rule: The rule that is scored
            result: The query/filter request for a given module obtained using rule

        Returns:
            integer indicating the rule/result precedence
        """
        # By default, only use the order in which rules are specified. This
        # behavior can be overwritten by subclassing.
        return 1

    def initialize(
        self,
        model: torch.nn.Module,
        *,
        safe: bool = True,
        overwrite_policy: _OverwriteOptions = "error",
    ) -> None:
        """Initialize quantizers in model following the rules in this config.

        For each quantizer (stub) that matches at least one location, the last
        added rule is applied. The factory function of this rule is executed
        to obtain an initialized quantizer.

        Args:
            model: Root module to initialize quantizers in
            overwrite_policy: Either `"skip"`, `"overwrite"`, or `"error"`. The
                `overwrite_policy` indicates what to do when the element in the
                collection is already an initialized quantizer. In case of
                `"skip"` the initialization for that specific quantizer is
                skipped and nothing happens, for `"overwrite"` a new quantizer
                is created and the existing quantizer is overwritten. Lastly,
                in case of `"error"` an error is raised when it is attempted to
                re-initialize an already initialized quantizer.
            safe: Boolean that indicates what to do when the module was
                already replaced in the root module between the creation of
                this collection and the call to initialize. When safe is
                `True`, an error is raised. Otherwise, a (re-)initialization of
                the quantizer is attempted.
        """
        matches: dict[Quantizer, _RuleItem] = {}
        for rule in self._rules:
            for result in find_quantizers(model, rule.query):
                score = self.precedence_score(rule, result)
                module = cast(Quantizer, result.module)
                if module not in matches or score >= matches[module].score:
                    matches[module] = _RuleItem(rule, result, score)

        for best_item in matches.values():
            factory = _ensure_factory_func(best_item.rule.factory, best_item.rule.kwargs)
            _initialize_quantizer(
                result=best_item.filter_result,
                quantizer_factory=factory,
                overwrite_policy=overwrite_policy,
                safe=safe,
            )

    def __repr__(self) -> str:
        if not self._rules:
            return f"{type(self).__name__}()"
        rules_repr = ",\n".join("  " + repr(rule) for rule in self._rules)
        return f"{type(self).__name__}(\n{rules_repr}\n)"
