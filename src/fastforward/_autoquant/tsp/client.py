# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Type-query clients for native adapters and remote TSP servers."""

import itertools

from typing import Any, Protocol, Sequence

from libcst.metadata import CodeRange

from fastforward._autoquant.tsp import jsonrpc
from fastforward._autoquant.tsp.types import TspTypeOperations, TypeInfo


class StaleSnapshotError(RuntimeError):
    """A query refers to an analysis state that is no longer current."""


class TypeClient(Protocol):
    """The document and type-query subset shared by mypy and TSP adapters."""

    def open_document(self, uri: str, text: str) -> None:
        """Analyze source text at the specified URI."""
        ...

    def close_document(self, uri: str) -> None:
        """Release a document's analysis state."""
        ...

    def get_snapshot(self) -> int:
        """Return the current analysis snapshot."""
        ...

    def get_computed_type(self, uri: str, code_range: CodeRange, snapshot: int) -> TypeInfo | None:
        """Look up a source range expressed in libcst coordinates."""
        ...

    def close(self) -> None:
        """Release the client's resources."""
        ...


class TspClient:
    """Translate the shared client interface to TSP 0.4 JSON-RPC requests."""

    def __init__(self, command: Sequence[str]) -> None:
        self._connection = jsonrpc.JsonRpcConnection(command)
        self._documents: dict[str, list[str]] = {}
        self._versions = itertools.count(1)
        self._operations = TspTypeOperations()
        self._snapshot: int | None = None
        try:
            self._connection.request(
                "initialize",
                {
                    "processId": None,
                    "rootUri": None,
                    "capabilities": {"general": {"positionEncodings": ["utf-16"]}},
                },
            )
            self._connection.notify("initialized", {})
            version = self._connection.request("typeServer/getSupportedProtocolVersion")
            if not isinstance(version, str) or not version.startswith("0.4."):
                msg = f"Unsupported Type Server Protocol version: {version!r}; expected 0.4.x."
                raise RuntimeError(msg)
        except BaseException:
            self._connection.close()
            raise

    def open_document(self, uri: str, text: str) -> None:
        """Synchronize the virtual source using LSP document notifications."""
        self._documents[uri] = text.split("\n")
        self._connection.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": "python",
                    "version": next(self._versions),
                    "text": text,
                }
            },
        )

    def close_document(self, uri: str) -> None:
        """Close the virtual document, including after failed analysis."""
        if self._documents.pop(uri, None) is not None:
            self._connection.notify("textDocument/didClose", {"textDocument": {"uri": uri}})

    def get_snapshot(self) -> int:
        """Obtain the server's current snapshot identifier."""
        snapshot: int = self._connection.request("typeServer/getSnapshot")
        self._snapshot = snapshot
        return snapshot

    def get_computed_type(self, uri: str, code_range: CodeRange, snapshot: int) -> TypeInfo | None:
        """Query a node, retrying once if background analysis invalidates its snapshot."""
        lines = self._documents[uri]
        lsp_range = {
            name: {
                "line": position.line - 1,
                "character": len(lines[position.line - 1][: position.column].encode("utf-16-le"))
                // 2,
            }
            for name, position in (("start", code_range.start), ("end", code_range.end))
        }
        params: dict[str, Any] = {
            "arg": {"uri": uri, "range": lsp_range},
            "snapshot": self._snapshot if self._snapshot is not None else snapshot,
        }
        try:
            result = self._connection.request("typeServer/getComputedType", params)
        except jsonrpc.JsonRpcError as exc:
            # LSP uses these when content changes or the server cancels;
            # refresh the snapshot and retry once.
            retryable_codes = (
                jsonrpc.LSP_CONTENT_MODIFIED,
                jsonrpc.LSP_SERVER_CANCELLED,
            )
            if exc.code not in retryable_codes:
                raise
            params["snapshot"] = self.get_snapshot()
            result = self._connection.request("typeServer/getComputedType", params)
        return TypeInfo(result, self._operations) if result is not None else None

    def close(self) -> None:
        """Shut down the server and release its process even if shutdown fails."""
        try:
            self._connection.request("shutdown")
            self._connection.notify("exit")
        finally:
            self._documents.clear()
            self._connection.close()
