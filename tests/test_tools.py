"""Tests for the one tool the Phase 1 agent can use."""

from agent.tools import ToolResult, count_words, run_tool


def test_count_words_counts_plain_words() -> None:
    # Protects: the tool gives an exact word count. The model often
    # miscounts words, which is the reason this tool exists. The count
    # is returned as text because tool results travel back to the
    # model as message text.
    result = count_words("one two three four")

    assert result == ToolResult(status="ok", content="4")


def test_run_tool_unknown_name_returns_failed_result() -> None:
    # Protects: a made up tool name never crashes the agent. The model
    # can invent names, so our code returns a failed result instead.
    # The message names the bad tool and lists the real one, so the
    # model can read it and correct itself on the next step.
    result = run_tool("delete_everything", {})

    assert result.status == "failed"
    assert "delete_everything" in result.content
    assert "count_words" in result.content


def test_run_tool_runs_a_known_tool() -> None:
    # Protects: a valid request from the model really runs the tool and
    # returns its result. The arguments arrive as a dict of names and
    # values, which is the shape Ollama sends for tool calls.
    result = run_tool("count_words", {"text": "one two three four"})

    assert result == ToolResult(status="ok", content="4")
