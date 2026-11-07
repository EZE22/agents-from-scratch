"""Tests for the agent loop."""

from agent.fake_model import FakeModel
from agent.loop import run_agent
from agent.model import ModelReply, ToolCall


def test_agent_stops_with_answered_when_model_replies_with_text() -> None:
    # Protects: the simplest path through the loop. When the model answers
    # with plain text and asks for no tool, the loop stops right away and
    # says why. The model chose to stop. Our code only names the reason.
    model = FakeModel([ModelReply(content="Paris")])

    result = run_agent("What is the capital of France?", model)

    assert result.answer == "Paris"
    assert result.stop_reason == "answered"
    assert result.steps == 1


def test_agent_runs_the_tool_then_answers() -> None:
    # Protects: the core loop. The model asks for a tool, our code runs it
    # and sends the result back as a message, and the model then uses it
    # to answer. The result must reach the model, or it could never use it.
    model = FakeModel(
        [
            ModelReply(
                content="",
                tool_calls=[
                    ToolCall(
                        name="count_words",
                        arguments={"text": "one two three four"},
                    )
                ],
            ),
            ModelReply(content="The text has 4 words."),
        ]
    )

    result = run_agent("How many words are in 'one two three four'?", model)

    assert result.answer == "The text has 4 words."
    assert result.stop_reason == "answered"
    assert result.steps == 2

    # What the model saw on its second call: the last message is the tool
    # result, with the exact count.
    second_call_messages = model.calls[1][0]
    assert second_call_messages[-1]["role"] == "tool"
    assert second_call_messages[-1]["content"] == "4"
