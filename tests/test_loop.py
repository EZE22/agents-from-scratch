"""Tests for the agent loop."""

from agent.fake_model import FakeModel
from agent.loop import run_agent
from agent.model import ModelReply


def test_agent_stops_with_answered_when_model_replies_with_text() -> None:
    # Protects: the simplest path through the loop. When the model answers
    # with plain text and asks for no tool, the loop stops right away and
    # says why. The model chose to stop. Our code only names the reason.
    model = FakeModel([ModelReply(content="Paris")])

    result = run_agent("What is the capital of France?", model)

    assert result.answer == "Paris"
    assert result.stop_reason == "answered"
    assert result.steps == 1
