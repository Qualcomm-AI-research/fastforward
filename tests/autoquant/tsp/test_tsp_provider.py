# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Check document identity and client lifetime in the shared metadata provider.

A recording client stands in for a type checker: it reports the source text of
each queried range instead of inferring a type. These tests cover how the
provider maps nodes and manages documents, not type inference.
"""

from pathlib import Path
from typing import ClassVar
from unittest.mock import Mock

import libcst
import pytest

from fastforward._autoquant.cst.filter import filter_nodes_by_type
from fastforward._autoquant.tsp.context import module_context
from fastforward._autoquant.tsp.type_provider_impl import TspTypeProvider, make_call_scoped_cache
from fastforward._autoquant.tsp.types import TypeInfo
from libcst.metadata import CodeRange


class _RecordingClient:
    """Answer each query with its document URI and source text."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.closed = False
        self._documents: dict[str, list[str]] = {}

    def open_document(self, uri: str, text: str) -> None:
        self.events.append(("open", uri))
        self._documents[uri] = text.split("\n")

    def close_document(self, uri: str) -> None:
        self.events.append(("close", uri))
        del self._documents[uri]

    def get_snapshot(self) -> int:
        return 1

    def get_computed_type(self, uri: str, code_range: CodeRange, snapshot: int) -> TypeInfo | None:
        del snapshot
        start, end = code_range.start, code_range.end
        if start.line != end.line:
            return None
        text = self._documents[uri][start.line - 1][start.column : end.column]
        return TypeInfo((uri, text), Mock())

    def close(self) -> None:
        self.closed = True


class _RecordingProvider(TspTypeProvider):
    CLIENT_FACTORY: ClassVar[Mock]


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> _RecordingClient:
    client = _RecordingClient()
    monkeypatch.setattr(
        _RecordingProvider, "CLIENT_FACTORY", Mock(return_value=client), raising=False
    )
    return client


def _uri_and_text(module: libcst.Module) -> tuple[str, str]:
    name = next(filter_nodes_by_type(module, libcst.Assign)).value
    metadata = libcst.MetadataWrapper(module, unsafe_skip_copy=True).resolve(_RecordingProvider)
    typ: tuple[str, str] = metadata[name].typ
    return typ


def test_types_are_queried_by_node_range_in_a_document_per_source(
    client: _RecordingClient, tmp_path: Path
) -> None:
    """Query each node at its own range within a document that identifies its source.

    A source module keeps its file URI so a type server resolves its relative
    imports. Standalone code receives a fresh URI per conversion, so identical
    snippets analyzed in sequence cannot be confused with one another.
    """
    # GIVEN identical code from a source file and twice from standalone conversions.
    path = tmp_path / "model.py"
    module = libcst.parse_module("result = value\n")

    # WHEN the provider resolves the assigned name in each document.
    with module_context("package.model", str(path)):
        source = _uri_and_text(module)
    first, second = _uri_and_text(module), _uri_and_text(module)

    # THEN the name's range is queried in the source URI or distinct standalone URIs.
    assert source == (path.as_uri(), "value")
    assert first[1] == second[1] == "value"
    assert len({source[0], first[0], second[0]}) == 3
    assert [event for event, _ in client.events] == ["open", "close"] * 3


def test_call_scope_reuses_one_client_and_releases_each_document(
    client: _RecordingClient,
) -> None:
    """Share one client across a call while closing every analyzed document.

    Starting a type server per function would be slow, and leaving documents
    open would let later analysis see stale source. Nested scopes must neither
    create a second client nor close the shared one early.
    """
    # GIVEN nested call scopes for the provider's client factory.
    factory = _RecordingProvider.CLIENT_FACTORY
    scope = make_call_scoped_cache(factory)
    module = libcst.parse_module("result = value\n")

    # WHEN two documents are analyzed within those scopes.
    with scope():
        with scope():
            first, _ = _uri_and_text(module)
        second, _ = _uri_and_text(module)
        assert not client.closed

    # THEN both documents were opened and closed on the single client, closed once.
    factory.assert_called_once()
    assert client.events == [("open", first), ("close", first), ("open", second), ("close", second)]
    assert client.closed


def test_failed_type_query_releases_the_document_and_client(client: _RecordingClient) -> None:
    """Release resources when metadata collection is interrupted.

    A failed conversion must not leave its document or a standalone client
    open for later calls, and the original error must reach the caller.
    """
    # GIVEN a client whose type query fails after the document is opened.
    failure = RuntimeError("type query failed")
    client.get_computed_type = Mock(side_effect=failure)  # type: ignore[method-assign]
    module = libcst.parse_module("result = value\n")

    # WHEN metadata resolution is interrupted outside any call scope.
    with pytest.raises(RuntimeError, match="type query failed"):
        libcst.MetadataWrapper(module).resolve(_RecordingProvider)

    # THEN the opened document and its client are released despite the error.
    (_, uri), *_ = client.events
    assert client.events == [("open", uri), ("close", uri)]
    assert client.closed
