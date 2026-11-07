"""Tests for the shape of a model reply.

What this file is about:
    Every reply from a model, real or fake, is a ModelReply. This file checks
    that the shape can express the two things a model can do: say something,
    or ask for a tool to be run.

Why this matters to a customer:
    The model never runs a tool itself. It only asks, and our code decides
    whether to allow it. That separation is where safety checks, limits and
    logging live, so the shape of a tool request has to be exact.
"""

# ToolCall is a request from the model to run one tool. ModelReply is the whole
# reply. ToolCall does not exist yet, so this import is the first thing that
# fails. That is the red step.
from agent.model import ModelReply, ToolCall


def test_reply_can_carry_a_tool_call() -> None:
    """Protects the shape of a tool request: a tool name plus its arguments.

    Why it matters:
        The agent loop reads these two fields to decide what to run. If the
        shape drifts, the loop breaks without any model being involved.

    What a reader might get wrong:
        The arguments come from the model, so they are untrusted. This test
        only checks that they are carried. Checking they are valid comes later,
        in the reply validation step.
    """
    # Arrange: a request to run a tool called get_weather for one city.
    # The names here are made up. A real tool arrives in Phase 1.
    call = ToolCall(name="get_weather", arguments={"city": "Atlanta"})

    # content is empty on purpose. When a model asks for a tool, it often says
    # nothing in words, so a reply with no text is normal and valid.
    reply = ModelReply(content="", tool_calls=[call])

    # Assert: the reply hands the request back unchanged.
    # tool_calls is a list because a model may ask for several tools at once.
    assert reply.tool_calls == [call]
    assert reply.tool_calls[0].name == "get_weather"
    assert reply.tool_calls[0].arguments == {"city": "Atlanta"}
