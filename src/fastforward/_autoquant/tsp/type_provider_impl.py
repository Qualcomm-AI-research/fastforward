# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Shared metadata collection and client lifetime for all type backends."""

import contextlib
import contextvars
import itertools

from pathlib import Path
from typing import Callable, ClassVar, Iterator

import libcst

from libcst.metadata import PositionProvider
from typing_extensions import override

from fastforward._autoquant.tsp.client import TypeClient
from fastforward._autoquant.tsp.context import current_module
from fastforward._autoquant.tsp.types import TypeInfo

_ClientFactory = Callable[[], TypeClient]
_active_clients: contextvars.ContextVar[dict[_ClientFactory, TypeClient]] = contextvars.ContextVar(
    "fastforward_autoquant_type_clients", default={}
)
_document_ids = itertools.count(1)


@contextlib.contextmanager
def _client_scope(factory: _ClientFactory) -> Iterator[TypeClient]:
    client = _active_clients.get().get(factory)
    if client is not None:
        yield client
        return
    client = factory()
    token = _active_clients.set({**_active_clients.get(), factory: client})
    try:
        yield client
    finally:
        _active_clients.reset(token)
        client.close()


def make_call_scoped_cache(
    factory: _ClientFactory,
) -> Callable[[], contextlib.AbstractContextManager[None]]:
    """Reuse a backend client for a call, including nested scopes."""

    @contextlib.contextmanager
    def scope() -> Iterator[None]:
        with _client_scope(factory):
            yield

    return scope


class TspTypeProvider(libcst.VisitorMetadataProvider[TypeInfo]):
    """Collect node metadata through a native or remote type-query client.

    Subclasses supply only a client factory. Both scoped and standalone
    metadata resolution release documents and clients when analysis raises.
    """

    METADATA_DEPENDENCIES = (PositionProvider,)
    CLIENT_FACTORY: ClassVar[_ClientFactory]

    def __init__(self) -> None:
        super().__init__()
        self._client: TypeClient | None = None
        self._uri = ""
        self._snapshot = 0

    @override
    def _gen_impl(self, module: libcst.Module) -> None:
        name, path = current_module()
        # Synchronize virtual contents at the original URI to preserve imports.
        source_path = (
            Path(path).absolute() if path else Path.cwd() / f"_{name}_{next(_document_ids)}.py"
        )
        self._uri = source_path.as_uri()
        with _client_scope(type(self).CLIENT_FACTORY) as client:
            self._client = client
            try:
                client.open_document(self._uri, module.code)
                self._snapshot = client.get_snapshot()
                super()._gen_impl(module)
            finally:
                try:
                    client.close_document(self._uri)
                finally:
                    self._client = None

    @override
    def on_visit(self, node: libcst.CSTNode) -> bool:
        code_range = self.get_metadata(PositionProvider, node, None)
        if self._client is not None and code_range is not None:
            typ = self._client.get_computed_type(self._uri, code_range, self._snapshot)
            if typ is not None:
                self.set_metadata(node, typ)
        return super().on_visit(node)
