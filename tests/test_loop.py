"""Tests for the agent loop."""

from agent.fake_model import FakeModel
from agent.loop import MAX_STEPS, run_agent
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


def test_agent_stops_at_the_step_limit_when_model_never_answers() -> None:
    # Protects: the hard step limit. A model can ask for tools forever, and
    # a customer pays for every ask. The script has exactly MAX_STEPS tool
    # requests and nothing else. If the loop asked even once more, FakeModel
    # would raise an error and fail this test loudly.
    tool_request = ModelReply(
        content="",
        tool_calls=[ToolCall(name="count_words", arguments={"text": "a b"})],
    )
    model = FakeModel([tool_request] * MAX_STEPS)

    result = run_agent("Keep counting forever.", model)

    assert result.stop_reason == "step_limit"
    assert result.steps == MAX_STEPS
    assert len(model.calls) == MAX_STEPS


def test_agent_offers_the_tool_to_the_model() -> None:
    # Protects: the model is told which tools exist. A real model can only
    # ask for a tool it has been shown. Without this, it would never call
    # count_words, and the agent would quietly act like a plain chatbot.
    model = FakeModel([ModelReply(content="Paris")])

    run_agent("What is the capital of France?", model)

    tools_offered = model.calls[0][1]
    assert tools_offered is not None
    names = [tool["function"]["name"] for tool in tools_offered]
    assert names == ["count_words"]


def test_agent_records_the_models_tool_request_before_the_result() -> None:
    # Protects: the conversation reads in the order it happened. The model
    # asks for a tool, then the result follows. A real model that sees a
    # tool result with no request before it can get confused about what
    # the result is for. The request is stored the way Ollama expects.
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

    run_agent("How many words are in 'one two three four'?", model)

    second_call_messages = model.calls[1][0]
    request_message = second_call_messages[-2]
    assert request_message["role"] == "assistant"
    assert request_message["tool_calls"][0]["function"]["name"] == "count_words"
