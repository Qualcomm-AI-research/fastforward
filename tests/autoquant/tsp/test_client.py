# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Check how the TSP client constructs requests and handles server failures.

JSON-RPC is mocked: replies are supplied by the tests, so these assertions check
client behavior, not whether a type checker infers the correct type. Transport
framing is covered separately in test_jsonrpc.py.
"""

from typing import Any, Iterator
from unittest.mock import Mock, call, patch

import pytest

from fastforward._autoquant.tsp.client import TspClient
from fastforward._autoquant.tsp.jsonrpc import (
    LSP_CONTENT_MODIFIED,
    LSP_SERVER_CANCELLED,
    JsonRpcError,
)
from libcst.metadata import CodePosition, CodeRange

_INT_RESPONSE = {"kind": 3, "declaration": {"name": "int"}}


@pytest.fixture
def connected_client() -> Iterator[tuple[TspClient, Mock]]:
    """Complete the handshake, leaving each test to specify its query responses."""
    with patch("fastforward._autoquant.tsp.client.jsonrpc.JsonRpcConnection") as connection:
        rpc = connection.return_value
        rpc.request.side_effect = [{"capabilities": {}}, "0.4.1"]
        client = TspClient(("type-server",))
        rpc.request.reset_mock(side_effect=True)
        rpc.notify.reset_mock()
        try:
            yield client, rpc
        finally:
            rpc.request.side_effect = None
            client.close()


def test_documents_are_synchronized_with_lsp_notifications(
    connected_client: tuple[TspClient, Mock],
) -> None:
    """Keep the server's view of open source documents in step with the client.

    The server analyzes only the text it received through didOpen. Reopening a
    document needs a new version, and closing it twice, as cleanup after a
    failed close can, must not send a didClose for a document the server forgot.
    """
    # GIVEN a document that is opened, closed, and opened again with new text.
    client, rpc = connected_client
    uri = "file:///example.py"

    # WHEN the client synchronizes each change and closes the document twice.
    client.open_document(uri, "value = 1\n")
    client.close_document(uri)
    client.open_document(uri, "value = 2\n")
    client.close_document(uri)
    client.close_document(uri)

    # THEN the server sees versioned opens with the full text and one close per open.
    def opened(version: int, text: str) -> Any:
        document = {"uri": uri, "languageId": "python", "version": version, "text": text}
        return call("textDocument/didOpen", {"textDocument": document})

    closed = call("textDocument/didClose", {"textDocument": {"uri": uri}})
    assert rpc.notify.call_args_list == [
        opened(1, "value = 1\n"),
        closed,
        opened(2, "value = 2\n"),
        closed,
    ]


def test_variable_after_emoji_is_queried_at_its_utf16_source_range(
    connected_client: tuple[TspClient, Mock],
) -> None:
    """Keep a type query pointed at the intended variable after an emoji.

    An emoji in a string before the variable on the same line counts as one
    LibCST character but two TSP/LSP UTF-16 units. Sending the unconverted range
    could return a wrong or missing type, affecting Autoquant's quantizer
    decisions. We check the outgoing range; the returned int type is mocked.
    """
    # GIVEN `value` after an emoji, which occupies two UTF-16 units but one Python character.
    client, rpc = connected_client
    uri = "file:///example.py"
    client.open_document(uri, 'label = "😀"; value = 1\n')
    rpc.request.return_value = _INT_RESPONSE

    # WHEN the provider queries `value` using libcst's character-based columns.
    client.get_computed_type(uri, CodeRange(CodePosition(1, 13), CodePosition(1, 18)), snapshot=7)

    # THEN the server receives the same variable's location in LSP coordinates.
    rpc.request.assert_called_once_with(
        "typeServer/getComputedType",
        {
            "arg": {
                "uri": uri,
                "range": {
                    "start": {"line": 0, "character": 14},
                    "end": {"line": 0, "character": 19},
                },
            },
            "snapshot": 7,
        },
    )


@pytest.mark.parametrize("error_code", [LSP_CONTENT_MODIFIED, LSP_SERVER_CANCELLED])
def test_stale_analysis_snapshot_is_refreshed_before_retrying_the_type_query(
    connected_client: tuple[TspClient, Mock],
    error_code: int,
) -> None:
    """Allow a type query to finish when background analysis changes snapshots.

    A server can invalidate a snapshot between document analysis and the query.
    Autoquant should retry with the latest snapshot instead of aborting that
    conversion. Mocked replies verify recovery, not the inferred int type.
    """
    # GIVEN the server rejects snapshot 7 and has advanced to snapshot 8.
    client, rpc = connected_client
    uri = "file:///example.py"
    client.open_document(uri, "value = 1\n")
    rpc.request.side_effect = [
        JsonRpcError("analysis changed", error_code),
        8,
        _INT_RESPONSE,
    ]

    # WHEN a query made against the old snapshot is rejected.
    info = client.get_computed_type(
        uri, CodeRange(CodePosition(1, 0), CodePosition(1, 5)), snapshot=7
    )

    # THEN the caller receives a type from a retry using the new snapshot.
    rpc.request.assert_any_call("typeServer/getSnapshot")
    method, params = rpc.request.call_args.args
    assert method == "typeServer/getComputedType"
    assert params["snapshot"] == 8
    assert info is not None and info.is_subtype("builtins.int")


def test_unsupported_protocol_version_closes_the_connection() -> None:
    """Reject an incompatible server before interpreting its type responses.

    A different protocol version may use a schema our adapter cannot interpret.
    Initialization must report this and release its connection. The mock checks
    that cleanup is requested; it does not launch a server process.
    """
    # GIVEN a server implementing a protocol this client cannot interpret.
    with patch("fastforward._autoquant.tsp.client.jsonrpc.JsonRpcConnection") as connection:
        rpc = connection.return_value
        rpc.request.side_effect = [{"capabilities": {}}, "0.3.0"]

        # WHEN / THEN initialization fails rather than querying incompatible types.
        with pytest.raises(RuntimeError, match="Unsupported Type Server Protocol"):
            TspClient(("type-server",))
        rpc.close.assert_called_once()


def test_shutdown_failure_still_closes_the_connection() -> None:
    """Request resource cleanup even when the server rejects graceful shutdown.

    Otherwise repeated Autoquant calls could leave server processes and pipes
    open. The mocked connection verifies that close is called despite the error.
    """
    # GIVEN an initialized server which reports an error during shutdown.
    with patch("fastforward._autoquant.tsp.client.jsonrpc.JsonRpcConnection") as connection:
        rpc = connection.return_value
        rpc.request.side_effect = [{"capabilities": {}}, "0.4.1"]
        client = TspClient(("type-server",))
        rpc.request.side_effect = JsonRpcError("shutdown failed")

        # WHEN / THEN the connection is closed even though the shutdown request raises.
        with pytest.raises(JsonRpcError, match="shutdown failed"):
            client.close()
        rpc.close.assert_called_once()
