# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

"""
MPath is a utility for finding and filtering submodules of
a module. For example, it helps in finding all submodules
of the same type or all modules whose name relative to the root module
satisfies a certain condition.
from typing import Any, NewType, Optional, cast

At it's core MPath is used through query strings, however, more complicated
queries can be build programmatically.

MPath is open to extensions. Examples of these are `fragments.RegexPathSelectorFragment`
and `fragments.ClassSelectorFragment`.

Example:
```python
    module = MyModule()
    mpath.search("**/decoder/[cls:torch.nn.Linear]", module)
```

This example will find all linear modules that are part of submodules called
decoder, i.e., the attribute name of the module in the parent module. The
result of `search()` is an `MPathCollection` which is a collection of search
results and metadata. This collection behaves like a set, i.e., union,
intersection and other set operation are supported. Moreover, it can be used to
perform batch updates to the orignal model. See the documentation of
`MPathCollection` for more details.
"""

from typing import Any, NewType, Optional, cast

from . import _parser, fragments, selector
from ._search import FilterResult as FilterResult
from ._search import MPathCollection as MPathCollection
from ._search import search as search

Fragment = selector.Fragment
Selector = selector.Selector

_QueryContext = NewType("_QueryContext", dict[str, Any])


def local_context() -> _QueryContext:
    return _QueryContext(_parser.get_caller_context())


def caller_context() -> _QueryContext:
    return _QueryContext(_parser.get_caller_context(2))


def query(
    query_str: str,
    *,
    context: Optional[_QueryContext] = None,
    aliases: Optional[dict[str, selector.BaseSelector]] = None,
) -> selector.BaseSelector:
    """
    Construct a query object for MPath from a query string.
    A query string consists of multiple fragment strings separated by a '/'
    fragment strings may consists of the following:

    - `*`: wildcard, will match any single fragment
    - `**`: wildcard, will match zero or more fragments
    - `[_a-zA-Z0-9]+`: path selector, will match a module whose name equals the
      fragment string
    - `[<extension_name>: <extension_input>]`: use the extension identified by
      extension_name (see register_mpath_query_extension for more details)
    - `~<other_fragment>`: match if other_fragment evaluates to False and vice versa.

    For example, consider the following module:
        Module(
            (attention): Attention(
                (Q): Linear()
                (K): Linear()
                (V): Linear()
            )
            (output): Linear()
        )

    The query string `attention/Q` will match the linear layer called Q in
    attention. `**/attention/[cls: torch.nn.Linear]` will match all the linear
    layers (Q, K, V) in the attention module. `**/~[cls:torch.nn.Linear]` will match the
    attention layer.

    Args:
        query_str: The query string to be parsed
        context: Context available to extension. If no context is passed, all
            locals and globals from the call site are included automatically.
        aliases: Mapping from string to `Selector`s. These can be used in `query_str`
            as aliases using `&<alias>` syntax. Note that the alias name must be a
            valid python identifier for this to work.

    Note:
        fragment extension have access to the local scope at the call location of
        query. This is, for example, used by the class based matcher. This means
        that the location at which `query()` is called matters. If you want to call
        query outside of the correct context, you can obtain the required context
        using `mpath.local_context()` and pass the result of that to
        `query(query_str, context=context)`
    """
    context = context or caller_context()
    return _parser.parse(query_str, context=context, aliases=aliases)


def aliases(
    *, context: Optional[dict[str, Any]] = None, **kwargs: str
) -> dict[str, selector.BaseSelector]:
    """
    Create aliases that can be used in `mpath.query` and `mpath.search`. Aliases
    are a dictionary that maps identifier strings to subqueries. This function is
    a convenience function for creating that mapping. When aliases are passed to
    `mpath.query` or `mpath.search` they can be referenced in the query using
    `&<alias identifier>`.

    The aliases passed to this function are processed in order, this means that
    an alias can be defined in terms of other aliases, e.g.:

    ```python
    aliases(my_alias="/first/second", other_alias="&my_alias/third")
    ```

    The resulting dictionary can be passed to `mpath.query` or `mpath.search` as
    the aliases argument

    Args:
        context: Context dictionary in which to evaluate the queries. Same as `mpath.query`

    Returns:
        Dictionary that can be used as alias in `mpath.query` and `mpath.search`
    """
    aliases: dict[str, selector.BaseSelector] = {}
    context = context or _parser.get_caller_context()
    for name, raw in kwargs.items():
        aliases[name] = _parser.parse(raw, context=context, aliases=aliases)
    return aliases


root = cast(selector.Selector, query("**"))

register_mpath_query_extension = _parser.register_mpath_query_extension
mpath_query_extension = _parser.mpath_query_extension
MPathQueryExtension = _parser.MpathQueryExtension
MPathQueryContextualExtension = _parser.MpathQueryContextualExtension

register_mpath_query_extension("re", fragments.RegexPathFragment)
register_mpath_query_extension("regex", fragments.RegexPathFragment)
register_mpath_query_extension("cls", fragments.ClassFragment)
register_mpath_query_extension("class", fragments.ClassFragment)
