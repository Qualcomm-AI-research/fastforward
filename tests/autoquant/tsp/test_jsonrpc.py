# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Check JSON-RPC message transport through real subprocess pipes.

The subprocesses only echo messages or emit broken frames. They do not analyze
Python or implement TSP; these tests cover communication needed by a type server.
"""

import contextlib
import sys

import pytest

from fastforward._autoquant.tsp.jsonrpc import JsonRpcConnection

# A single-request echo peer: it decodes our frame and returns the same payload.
# It deliberately emits UTF-8 bytes and an unsolicited notification first.
_ECHO_PEER = """
import json
import sys

def send(message):
    body = json.dumps(message, ensure_ascii=False).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\\r\\n\\r\\n".encode() + body)
    sys.stdout.buffer.flush()

length = int(sys.stdin.buffer.readline().split(b":")[1])
sys.stdin.buffer.readline()
request = json.loads(sys.stdin.buffer.read(length))
send({"jsonrpc": "2.0", "method": "window/logMessage", "params": {"type": 3, "message": "ready"}})
send({"jsonrpc": "2.0", "id": request["id"], "result": request["params"]})
"""


@pytest.mark.slow
def test_unicode_payload_round_trips_despite_an_interleaved_notification() -> None:
    """Read the complete reply even with non-ASCII text and preceding server logs.

    Server messages can contain non-ASCII identifiers or source text. Their
    Content-Length counts UTF-8 bytes, not characters; confusing the two can
    truncate a response or consume bytes from the next frame. The echo peer
    sends raw UTF-8 to exercise this distinction, plus a log notification that
    must not be mistaken for the requested result. No types are inferred here.
    """
    # GIVEN a real pipe peer that echoes requests and sends a notification before its reply.
    with contextlib.closing(
        JsonRpcConnection((sys.executable, "-u", "-c", _ECHO_PEER))
    ) as connection:
        payload = {"text": "image dimensions: 高さ × 幅"}

        # WHEN a request is framed, read by the peer, and returned as UTF-8 JSON.
        result = connection.request("example/echo", payload)

        # THEN byte lengths preserve the payload and the notification is not mistaken for a reply.
        assert result == payload


@pytest.mark.parametrize(
    ("response", "message"),
    [
        pytest.param(
            b"Content-Length: 100\r\n\r\n{}",
            "during a response",
            id="server-exits-before-body-is-complete",
        ),
        pytest.param(
            b"Content-Type: application/json\r\n\r\n{}",
            "missing a Content-Length",
            id="server-omits-body-length",
        ),
    ],
)
def test_broken_server_response_raises(response: bytes, message: str) -> None:
    """Report broken communication instead of returning a missing type result.

    A server that exits mid-response or omits the body length leaves no usable
    reply. Both cases must raise an explicit transport error so Autoquant cannot
    silently proceed as if analysis succeeded but found no type.
    """
    # GIVEN a peer that receives a request, writes a broken frame, and exits.
    peer = (
        "import sys\n"
        + "sys.stdin.buffer.readline()\n"
        + f"sys.stdout.buffer.write({response!r})\n"
        + "sys.stdout.buffer.flush()\n"
    )
    with contextlib.closing(JsonRpcConnection((sys.executable, "-u", "-c", peer))) as connection:
        # WHEN / THEN a broken connection fails explicitly instead of looking like an unknown type.
        with pytest.raises(RuntimeError, match=message):
            connection.request("typeServer/getSnapshot")
