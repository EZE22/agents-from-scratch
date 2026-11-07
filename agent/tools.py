"""The tool the Phase 1 agent can use.

A tool is plain code with one narrow job. The model can ask for it, but our
code decides whether to run it. Nothing in this file talks to a model.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class ToolResult:
    """What a tool gives back to the agent loop.

    Every tool returns this same shape. That way the loop never has to
    guess what came back, and a failure can be told apart from a success.

    status is "ok" or "failed". A failed result holds a short message the
    model can read, so it can fix its request and try again.
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
    # Type hints are not enforced when the program runs, and the model may
    # send anything. So the tool checks for itself. A reader might think the
    # hint "text: str" already protects us. It does not.
    if not isinstance(text, str):
        return ToolResult(
            status="failed",
            content=(
                f"Bad argument 'text': expected a string, "
                f"got {type(text).__name__}."
            ),
        )
    return ToolResult(status="ok", content=str(len(text.split())))


# The table of tools the model is allowed to ask for. The key is the name the
# model uses, the value is our function. Adding a tool later means adding one
# line here. The model can never reach a function that is not in this table.
TOOLS: dict[str, Callable[..., ToolResult]] = {"count_words": count_words}


def run_tool(name: str, arguments: dict[str, object]) -> ToolResult:
    """Run the tool the model asked for, or say why we could not.

    This is the one door between the model's request and our tools. The
    model only asks. Our code decides what runs.

    An unknown name returns a failed result with a plain message the model
    can read. It names the bad tool and lists the real ones, so the model
    can correct itself. A reader might expect an exception here. We return
    a result instead, because an exception would crash the whole agent run.

    Known gap: this checks argument names, not their types. A model could
    send text=5 and the tool would still run. The next test covers that.
    """
    tool = TOOLS.get(name)
    if tool is None:
        available = ", ".join(TOOLS)
        return ToolResult(
            status="failed",
            content=f"Unknown tool '{name}'. Available tools: {available}.",
        )

    # bind() only checks that the arguments fit the function. It does not
    # run the function. We do not use a plain try around the call, because
    # that would also hide real bugs inside the tool as "bad arguments".
    try:
        inspect.signature(tool).bind(**arguments)
    except TypeError as error:
        return ToolResult(
            status="failed",
            content=f"Bad arguments for '{name}': {error}",
        )
    return tool(**arguments)
