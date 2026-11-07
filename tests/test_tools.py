"""Tests for the one tool the Phase 1 agent can use."""

from agent.tools import ToolResult, count_words


def test_count_words_counts_plain_words() -> None:
    # Protects: the tool gives an exact word count. The model often
    # miscounts words, which is the reason this tool exists. The count
    # is returned as text because tool results travel back to the
    # model as message text.
    result = count_words("one two three four")

    assert result == ToolResult(status="ok", content="4")
