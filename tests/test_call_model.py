"""Tests for call_model, the one function the rest of the project uses to reach
a model.

What this file is about:
    call_model joins the two pieces we already built: build_request prepares
    what we send, and parse_reply checks what comes back. In between sits the
    network call. To test the joining without a network, we hand call_model a
    stand in for the network call. Real code uses the real one. Tests use ours.

    Handing a function its helpers from outside is called dependency
    injection. It is the same idea as the fake model, one level down: the fake
    model replaces the model, and this stand in replaces the network.

Why this matters to a customer:
    Everything about how we reach a model lives behind this one function.
    Adding a proxy, a login header or a different host for a customer's
    network changes the sender, not the rest of the system.
"""

# call_model does not exist yet, so this import is the first thing that fails.
# That is the red step.
from agent.model import ModelReply, build_request, call_model


def test_call_model_builds_the_request_sends_it_and_parses_the_reply() -> None:
    """Protects the glue: the request we send is the one build_request makes,
    it goes to Ollama's chat address, and the answer comes back as a ModelReply.

    Why it matters:
        Each piece already has its own test. This one proves they are wired
        together in the right order. A mistake here, like sending the wrong
        address, would pass every other test and still fail with a real model.

    What a reader might get wrong:
        No real model is involved. The stand in below returns a canned reply in
        Ollama's format, so this test runs in milliseconds with Ollama off.
        Whether the real network call works is checked by hand, not here.
    """
    messages = [{"role": "user", "content": "hi"}]

    # What the stand in saw, so we can check it afterwards.
    seen: dict[str, object] = {}

    def fake_post(url: str, payload: dict[str, object]) -> dict[str, object]:
        """Stand in for the network call. It records what it was given and
        answers with a canned reply in Ollama's format."""
        seen["url"] = url
        seen["payload"] = payload
        return {
            "model": "qwen3:8b",
            "message": {"role": "assistant", "content": "Hi!"},
            "done": True,
            "done_reason": "stop",
        }

    # post is passed in by name. That is the injection: call_model uses our
    # stand in instead of the real network call.
    reply = call_model(messages, post=fake_post)

    # The answer is parsed into our own type.
    assert reply == ModelReply(content="Hi!")

    # It went to the address where Ollama listens for chat requests.
    assert seen["url"] == "http://localhost:11434/api/chat"

    # The request is exactly what build_request makes. We compare against
    # build_request itself, so this test does not repeat its details.
    assert seen["payload"] == build_request("qwen3:8b", messages)
