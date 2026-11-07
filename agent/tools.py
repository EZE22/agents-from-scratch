"""The tool the Phase 1 agent can use.

A tool is plain code with one narrow job. The model can ask for it, but our
code decides whether to run it. Nothing in this file talks to a model.
"""

from dataclasses import dataclass


@dataclass
class ToolResult:
    """What a tool gives back to the agent loop.

    Every tool returns this same shape. That way the loop never has to
    guess what came back, and a failure can be told apart from a success.

    status is "ok" for now. A "failed" status comes later, with its own test.
    content is always text, because tool results travel back to the model
    as message text. A reader might expect a number here, but the model only
    reads text, so the count is stored as a string.
    """

    status: str
    content: str


def count_words(text: str) -> ToolResult:
    """Count the words in text and return the count as a ToolResult.

    Why this tool exists: the local model often miscounts words, and plain
    code counts exactly.

    A reader might think text.split(" ") does the same job. It does not.
    split() with no argument splits on any run of spaces, tabs or new lines
    and ignores extra spaces, so "a  b" counts as two words, not three.
    """
    return ToolResult(status="ok", content=str(len(text.split())))
