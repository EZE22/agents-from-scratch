"""Tests for the fake model.

What this file is about:
    The fake model is a stand in for the real model (qwen3:8b running in
    Ollama). It lets us test our own code without a real model. Real models
    are slow, give different answers each time, and need Ollama running.
    Tests must be fast and repeatable, so they never call a real model.

Why this matters to a customer:
    Because we can script exactly what the "model" says, we can prove how our
    system behaves when the model misbehaves (bad output, wrong tool, endless
    loops) without waiting for a real model to misbehave by chance.
"""

# FakeModel is the stand in we are testing. It does not exist yet, so this
# import is the first thing that fails. That is the "red" step of test first
# development: we write the test, watch it fail for the right reason, and only
# then write the code.
from agent.fake_model import FakeModel

# ModelReply is the single shape every model reply takes, real or fake.
# Using one shared type is what lets the rest of the code treat both the same.
from agent.model import ModelReply


def test_fake_model_returns_scripted_replies_in_order() -> None:
    """Protects the core promise of the fake model: tests get exactly the
    replies they scripted, in the order they scripted them.

    Why it matters:
        Every later test of the agent loop depends on this. If the fake model
        were unpredictable, a failing loop test could not be trusted, because
        we would not know whether the loop or the fake was at fault.
    """
    # Arrange: build two distinct replies we can tell apart.
    # content is the text the "model" says. A reply holds only text for now.
    # The field for tool requests is added later, with its own failing test,
    # because we never write code before the test that needs it.
    first = ModelReply(content="first")
    second = ModelReply(content="second")

    # Hand the script to the fake model. It will give these back one at a
    # time, oldest first.
    fake = FakeModel([first, second])

    # Act and assert: call the fake twice, like a loop asking the model twice.
    # The argument is the conversation so far, a list of messages. The fake
    # ignores its content here, but real models read it, so the fake must
    # accept the same input.
    #
    # We compare with "is", not "==". "is" checks it is the very same object we
    # scripted, which is a stricter check. A reader might expect "==", but "is"
    # would catch the fake quietly returning a copy or a rebuilt reply.
    assert fake([{"role": "user", "content": "hi"}]) is first
    assert fake([{"role": "user", "content": "again"}]) is second
