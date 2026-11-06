"""Tests for parse_reply, the function that checks a raw model reply.

What this file is about:
    Ollama answers with a big JSON document. parse_reply turns it into our own
    ModelReply, and refuses anything that does not look right. The rest of our
    code then only ever sees ModelReply, never the raw document.

Why this matters to a customer:
    This is the one place that knows what Ollama's reply looks like. If a
    customer moves to a different model host, only this function changes.
    It is also where we stop bad model output before it can cause harm.

How the test data was made:
    The dictionaries below are real replies captured from qwen3:8b through
    Ollama on 2026-11-05, trimmed to the fields we care about. We used real
    output so our tests match reality, not our guess of it.
"""

# parse_reply does not exist yet, so this import is the first thing that
# fails. That is the red step.
from agent.model import ModelReply, ToolCall, parse_reply

# A real reply where the model simply said something.
# "message" holds what the model produced. "done_reason" is "stop" when the
# model finished on its own, which is the normal case.
TEXT_REPLY = {
    "model": "qwen3:8b",
    "message": {"role": "assistant", "content": "Hi!"},
    "done": True,
    "done_reason": "stop",
}


def test_parse_reply_turns_a_text_reply_into_a_model_reply() -> None:
    """Protects the happy path: a plain text answer becomes a ModelReply.

    Why it matters:
        Most replies are plain text. If this breaks, nothing downstream works.

    What a reader might get wrong:
        We compare two ModelReply objects with ==. That works because a
        dataclass compares field by field. The reply has no tool calls, so the
        expected value relies on tool_calls defaulting to an empty list.
    """
    assert parse_reply(TEXT_REPLY) == ModelReply(content="Hi!")


# A real reply where the model did not answer in words. It asked for the
# get_weather tool instead. Notice three things about the real shape:
#   1. "content" is empty, because the model said nothing in words.
#   2. "tool_calls" is a list, because a model can ask for several tools.
#   3. Each request nests its name and arguments under "function".
# The "id" and "index" fields are ignored for now. We will add them only if a
# later step needs them.
TOOL_REPLY = {
    "model": "qwen3:8b",
    "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_e8jfj198",
                "function": {
                    "index": 0,
                    "name": "get_weather",
                    "arguments": {"city": "Atlanta"},
                },
            }
        ],
    },
    "done": True,
    "done_reason": "stop",
}


def test_parse_reply_turns_a_tool_request_into_tool_calls() -> None:
    """Protects the moment that makes an agent possible: the model asking for
    a tool is read correctly.

    Why it matters:
        The agent loop only works if a tool request arrives as a ToolCall with
        the right name and arguments. Misreading it here would send the wrong
        tool the wrong input.

    What a reader might get wrong:
        Ollama already gives the arguments as a dictionary. Some other model
        hosts send them as a JSON string. This test covers Ollama only. A
        different host would need its own version of this test and function.
    """
    expected = ModelReply(
        content="",
        tool_calls=[ToolCall(name="get_weather", arguments={"city": "Atlanta"})],
    )

    assert parse_reply(TOOL_REPLY) == expected
