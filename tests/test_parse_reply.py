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

# pytest gives us parametrize (run one test over many inputs) and raises
# (check that code fails the way we expect).
import pytest
from typing import Any

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


# Six replies a model or a broken server could send. Each one is wrong in a
# different way. We list them as data so adding a sixth later is one line.
# The ids are labels pytest shows in its report, so a failure names the case.
BAD_REPLIES = [
    pytest.param({"done": True}, id="no_message_at_all"),
    pytest.param({"message": "hello"}, id="message_is_text_not_a_dict"),
    pytest.param(
        {"message": {"role": "assistant", "content": 42}},
        id="content_is_a_number_not_text",
    ),
    pytest.param(
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"arguments": {"city": "Atlanta"}}}],
            }
        },
        id="tool_call_has_no_name",
    ),
    pytest.param(
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": "city=Atlanta"}}
                ],
            }
        },
        id="tool_arguments_are_text_not_a_dict",
    ),
    pytest.param(
        {
            "message": {
                "role": "assistant",
                "content": "",
                # A number where a list should be. A loop over a number
                # crashes with TypeError, not our clear ValueError.
                "tool_calls": 5,
            }
        },
        id="tool_calls_is_a_number_not_a_list",
    ),
]


@pytest.mark.parametrize("bad_reply", BAD_REPLIES)
def test_parse_reply_rejects_malformed_replies(bad_reply: dict[str, Any]) -> None:
    """Protects the rule: never trust what a model returns.

    Why it matters:
        A model can send nonsense, and a broken server can send nonsense too.
        If bad data slips through here, it reaches our tools. For a customer,
        that is how an agent ends up running a tool with garbage inputs.

    What a reader might get wrong:
        Failing loudly is the goal. A reply that is accepted silently is worse
        than one that crashes, because nothing warns anyone. Run this test
        before the fix and you will see both kinds: some cases crash, and some
        are accepted without a word.

    About match:
        Every error starts with "Bad model reply", so a log search finds all of
        them with one phrase.
    """
    with pytest.raises(ValueError, match="Bad model reply"):
        parse_reply(bad_reply)
