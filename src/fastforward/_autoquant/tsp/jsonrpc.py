# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Minimal JSON-RPC client over stdio, using LSP `Content-Length` framing.

The Type Server Protocol (TSP) rides the same connection as LSP: requests and
notifications are framed with `Content-Length` headers, not newline
delimiters. This module knows nothing about TSP's own
methods -- `fastforward._autoquant.tsp.client.TspClient` builds on top of it.
"""

import itertools
import json
import subprocess

from typing import Any, Sequence

JSON_RPC_METHOD_NOT_FOUND = -32601
LSP_CONTENT_MODIFIED = -32801
LSP_SERVER_CANCELLED = -32802


class JsonRpcError(RuntimeError):
    """Raised when a JSON-RPC request returns an `error` object."""

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


class JsonRpcConnection:
    """A blocking JSON-RPC 2.0 connection to a subprocess speaking LSP framing."""

    def __init__(self, command: Sequence[str]) -> None:
        if not command:
            msg = "A type server launch command is required."
            raise ValueError(msg)
        try:
            self._process = subprocess.Popen(
                list(command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError as exc:
            msg = f"Could not start type server: {command[0]!r} not found on PATH."
            raise RuntimeError(msg) from exc
        self._ids = itertools.count(1)

    def request(self, method: str, params: Any = None) -> Any:
        """Send a request and block until its matching response arrives."""
        request_id = next(self._ids)
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        self._write(message)

        while True:
            response = self._read()
            if "method" in response:
                if "id" in response:
                    self._write({
                        "jsonrpc": "2.0",
                        "id": response["id"],
                        "error": {
                            "code": JSON_RPC_METHOD_NOT_FOUND,
                            "message": "Method not supported",
                        },
                    })
                continue
            if response.get("id") != request_id:
                # A notification (e.g. `typeServer/snapshotChanged`) or a
                # response to a request we are no longer waiting for; skip it.
                continue
            if "error" in response:
                error_msg = f"{method} failed: {response['error']}"
                raise JsonRpcError(error_msg, response["error"].get("code"))
            return response.get("result")

    def notify(self, method: str, params: Any = None) -> None:
        """Send a notification; there is no response to wait for."""
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self._write(message)

    def close(self) -> None:
        """Close stdin/stdout and terminate the subprocess."""
        if self._process.stdin is not None:
            self._process.stdin.close()
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        if self._process.stdout is not None:
            self._process.stdout.close()

    def _write(self, message: dict[str, Any]) -> None:
        assert self._process.stdin is not None
        body = json.dumps(message).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        self._process.stdin.write(header + body)
        self._process.stdin.flush()

    def _read(self) -> dict[str, Any]:
        assert self._process.stdout is not None
        content_length: int | None = None
        while True:
            line = self._process.stdout.readline()
            if not line:
                msg = "Type server closed its connection unexpectedly."
                raise RuntimeError(msg)
            stripped = line.strip()
            if not stripped:
                break
            name, _, value = stripped.decode("ascii").partition(":")
            if name.strip().lower() == "content-length":
                content_length = int(value.strip())

        if content_length is None:
            msg = "Type server response is missing a Content-Length header."
            raise RuntimeError(msg)

        body = self._process.stdout.read(content_length)
        if len(body) != content_length:
            msg = "Type server closed its connection during a response."
            raise RuntimeError(msg)
        result: dict[str, Any] = json.loads(body.decode("utf-8"))
        return result
