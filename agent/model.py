"""How our code talks to a model.

This file will hold the one small function that reaches the model (Ollama
running qwen3:8b). Keeping that in one place means we can swap in a different
model later by changing this file only, not the whole system.

Right now it holds the shape of a model reply. The function that calls the
model comes later, test first.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A request from the model to run one tool.

    The model never runs a tool itself. It only asks, using this shape, and our
    code decides whether to run it. That is where limits, safety checks and
    logging live.

    Attributes:
        name: The name of the tool the model wants to run.
        arguments: The inputs the model chose for that tool. The type is
            dict[str, object], not dict[str, str], because a model can send any
            kind of value. "object" also makes a type checker force us to
            check a value before using it, which is the habit we want, because
            these values are untrusted.
    """

    name: str
    arguments: dict[str, object]


@dataclass
class ModelReply:
    """One reply from a model, in the single shape the rest of our code uses.

    Both the real model function and the fake model return this type. That is
    what lets the agent loop treat them the same, so tests can swap the fake in
    without the loop noticing.

    A dataclass writes the boring parts for us (the constructor and a readable
    printout), so we only list the fields.

    Attributes:
        content: The text the model said. It can be empty when the model only
            asks for a tool.
        tool_calls: The tools the model asked to run, in order. Empty when the
            model asked for none.
    """

    content: str

    # field(default_factory=list) gives every reply its own fresh empty list.
    # A plain "= []" would be a classic trap: one list shared by every reply,
    # so a change to one reply would silently change all of them. Python's
    # dataclass refuses "= []" for exactly this reason.
    tool_calls: list[ToolCall] = field(default_factory=list)


def parse_reply(raw: dict[str, Any]) -> ModelReply:
    """Turn the raw reply from Ollama into our own ModelReply.

    This is the one place in the project that knows what Ollama's reply looks
    like. Everything else only sees ModelReply. If we move to a different model
    host, this is the function that changes.

    Args:
        raw: The reply document from Ollama's /api/chat, already converted from
            JSON into a Python dictionary. The type is dict[str, Any] because
            JSON from outside our program can hold anything, so we cannot
            promise more than that.

    Returns:
        A ModelReply holding the text the model said.

    What a reader might get wrong:
        This version trusts the document completely. It will crash with a
        KeyError on a reply with no "message". That is known and temporary. We
        add checks one at a time, each with its own failing test first, because
        model output must never be trusted.
    """
    # Ollama puts the model's answer under "message", and the text under
    # "content". Real example: {"message": {"role": "assistant", "content": "Hi!"}}
    message = raw["message"]
    return ModelReply(content=message["content"])
