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

# pytest gives us pytest.raises, which checks that a piece of code fails in the
# way we expect. We need it to test failure behavior, not just success.
import pytest

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


def test_fake_model_raises_clear_error_when_script_runs_out() -> None:
    """Protects against a confusing failure when code asks the model too often.

    Why it matters:
        If the agent loop has a bug and calls the model more times than a test
        expected, we want a message that says exactly that. A bare IndexError
        points at the inside of the fake and sends a reader down the wrong
        path. A clear message shortens debugging, and Phase 1 will rely on it
        to prove the loop stops when it should.
    """
    # Arrange: a script with exactly one reply.
    fake = FakeModel([ModelReply(content="only one")])

    # The first call uses up the only reply. This is normal and should work.
    fake([{"role": "user", "content": "first call"}])

    # Act and assert: the second call has nothing left to return.
    # pytest.raises passes only if the code inside raises RuntimeError.
    # match checks the message with a regular expression, so a reader who sees
    # this failure in a log knows what happened. If any other error type
    # appears (like IndexError), this test fails, and that is the point.
    with pytest.raises(RuntimeError, match="no scripted replies left"):
        fake([{"role": "user", "content": "second call"}])


def test_fake_model_records_what_it_was_sent_as_a_snapshot() -> None:
    """Protects our ability to check what the agent loop sent to the model.

    Why it matters:
        In Phase 1 the loop will build a conversation, call the model, add a
        tool result, and call the model again. To prove the loop did that
        correctly, a test needs to see exactly what the model received on each
        call. The fake records it for us.

    What a reader might get wrong:
        The fake must store a snapshot (a copy), not the live list. The loop
        keeps adding messages to one growing list. If the fake kept that same
        list, every recorded call would quietly show the final conversation,
        and a test could pass or fail for the wrong reason. The second half of
        this test catches that.
    """
    fake = FakeModel([ModelReply(content="ok")])

    # The conversation and the tools on offer, as the loop would pass them.
    # The tool description is a made up example in Ollama's format.
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "get_weather"}}]

    fake(messages, tools=tools)

    # The fake recorded one call: the messages and the tools it was given.
    assert fake.calls == [(messages, tools)]

    # Now behave like the loop: add a message to the SAME list after the call.
    messages.append({"role": "assistant", "content": "ok"})

    # The recorded call must still show only the one message it received at
    # the time. If this fails, the fake stored a reference, not a snapshot.
    assert len(fake.calls[0][0]) == 1
