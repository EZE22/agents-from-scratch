"""Tests for post_json, the helper that really sends a request over the network.

What this file is about:
    post_json is the one piece of code that talks over the network. We cannot
    use a real model in tests. So the test starts a tiny pretend Ollama on this
    same computer. post_json talks to it, and we check two things: what the
    pretend Ollama received, and what post_json handed back.

Why this matters to a customer:
    All network trouble, like a slow server, a wrong address or a login header,
    lives in this one small helper. Keeping it small makes problems easy to find.
"""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

# post_json does not exist yet, so this import is the first thing that fails.
# That is the red step.
from agent.model import post_json


def test_post_json_sends_the_request_and_returns_the_reply() -> None:
    """Protects the basic job: send a dictionary as JSON, get a dictionary back.

    Why it matters:
        If this breaks, nothing can reach the model.

    What a reader might get wrong:
        The "server" here is not Ollama. It is a few lines of Python that
        answer with a made up reply. It runs inside the test and is switched off
        when the test ends. No internet is used, and Ollama can be off.
    """
    # The pretend server writes down what it receives. We read this afterwards.
    received: dict[str, object] = {}

    class PretendOllama(BaseHTTPRequestHandler):
        """A tiny server that answers every POST with a made up reply."""

        def do_POST(self) -> None:
            # Read the letter. Content-Length says how many bytes to read.
            length = int(self.headers["Content-Length"])
            received["path"] = self.path
            received["body"] = json.loads(self.rfile.read(length))

            # Send back a made up reply, in the same shape Ollama uses.
            reply = json.dumps(
                {"message": {"role": "assistant", "content": "Hi!"}}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, format: str, *args: object) -> None:
            # The server normally prints a line for every request. We switch
            # that off to keep the test output clean.
            pass

    # Port 0 means "pick any free port". That way the test never fights with a
    # real Ollama, which uses port 11434.
    server = HTTPServer(("127.0.0.1", 0), PretendOllama)

    # Run the server in the background so the test can keep going.
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        port = server.server_address[1]
        result = post_json(f"http://127.0.0.1:{port}/api/chat", {"hello": "world"})
    finally:
        # Always switch the server off, even if the line above fails.
        server.shutdown()
        server.server_close()

    # What came back is the made up reply, as a dictionary.
    assert result == {"message": {"role": "assistant", "content": "Hi!"}}

    # What the server received is what we sent, at the address we used.
    assert received["path"] == "/api/chat"
    assert received["body"] == {"hello": "world"}


def test_post_json_gives_a_clear_error_when_the_server_is_not_running() -> None:
    """Protects the most common problem in the field: the model server is off.

    Why it matters:
        When Ollama is not running, the raw error is a long network message
        that means nothing to most people. A customer, or a teammate, should
        see one plain sentence that says what is wrong and what to try.

    What a reader might get wrong:
        We need an address where nothing is listening, and we cannot just guess
        a port, because a guess might hit something real. So we ask the
        computer for a free port, then close it right away. Nothing is using
        it after that, so the connection is refused every time.
    """
    # Ask for any free port, then let go of it. The "with" block closes the
    # socket at its end, so the port is free and nothing listens on it.
    with socket.socket() as free_socket:
        free_socket.bind(("127.0.0.1", 0))
        port = free_socket.getsockname()[1]

    # We expect our own plain error, not the raw network one. The text "Could
    # not reach" is what a person will see, and what a log search will find.
    with pytest.raises(RuntimeError, match="Could not reach"):
        post_json(f"http://127.0.0.1:{port}/api/chat", {"hello": "world"})
