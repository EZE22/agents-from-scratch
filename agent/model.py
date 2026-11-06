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


def _bad_reply(detail: str) -> ValueError:
    """Build the error we raise for a reply we refuse to accept.

    Every refusal starts with the same phrase, "Bad model reply", so a log
    search finds all of them at once. We return the error instead of raising
    it here so the "raise" stays visible at the place where the check failed.
    """
    return ValueError(f"Bad model reply: {detail}")


def _read_message(raw: dict[str, Any]) -> dict[str, Any]:
    """Return the "message" part of Ollama's reply, or refuse the reply.

    Ollama puts the model's answer under "message". If it is missing, or is
    something other than a dictionary, there is nothing safe to read.
    """
    message = raw.get("message")

    # isinstance checks the real type at run time. We cannot trust the type
    # hints here, because hints are only promises we make to ourselves. This
    # data came from outside the program.
    if not isinstance(message, dict):
        raise _bad_reply("there is no 'message' dictionary")
    return message


def _read_content(message: dict[str, Any]) -> str:
    """Return the text the model said, or refuse the reply.

    The text must be a string. An empty string is fine and normal: a model that
    only asks for a tool usually says nothing in words.
    """
    content = message.get("content")

    # A number or a list here would otherwise flow silently into our code and
    # fail somewhere far from the cause. We stop it at the door.
    if not isinstance(content, str):
        raise _bad_reply("'content' is not text")
    return content


def _read_tool_call(item: Any) -> ToolCall:
    """Turn one tool request from the model into a ToolCall, or refuse it.

    Args:
        item: One entry from the reply's "tool_calls" list. The type is Any
            because it came from outside the program and could be anything.

    Real example of a good entry:
        {"id": "call_x", "function": {"name": "get_weather",
                                      "arguments": {"city": "Atlanta"}}}
    """
    function = item.get("function") if isinstance(item, dict) else None
    if not isinstance(function, dict):
        raise _bad_reply("a tool call has no 'function' dictionary")

    name = function.get("name")
    # A tool with no name cannot be looked up, so there is nothing to run.
    # "not name" also catches an empty string.
    if not isinstance(name, str) or not name:
        raise _bad_reply("a tool call has no name")

    arguments = function.get("arguments")
    # Ollama sends arguments as a dictionary. Some other hosts send a JSON
    # string. We accept only the dictionary, so a surprise shows up here as a
    # clear error instead of as a tool receiving the wrong kind of input.
    if not isinstance(arguments, dict):
        raise _bad_reply(f"the arguments for '{name}' are not a dictionary")

    return ToolCall(name=name, arguments=arguments)


def parse_reply(raw: dict[str, Any]) -> ModelReply:
    """Turn the raw reply from Ollama into our own ModelReply.

    This is the one place in the project that knows what Ollama's reply looks
    like. Everything else only sees ModelReply. If we move to a different model
    host, this is the function that changes.

    It also enforces the rule that model output is never trusted. A reply that
    does not look right is refused with a ValueError, so bad data cannot reach
    our tools.

    Args:
        raw: The reply document from Ollama's /api/chat, already converted from
            JSON into a Python dictionary. The type is dict[str, Any] because
            JSON from outside our program can hold anything, so we cannot
            promise more than that.

    Returns:
        A ModelReply holding the text the model said and any tools it asked
        to run.

    Raises:
        ValueError: If the reply is malformed. The message starts with
            "Bad model reply".

    What a reader might get wrong:
        The checks cover the five bad replies we have tests for. They are not
        complete. For example, "tool_calls" being something other than a list
        is not checked yet. Each gap gets its own failing test before its fix.
    """
    message = _read_message(raw)
    content = _read_content(message)

    # A reply that only speaks has no "tool_calls" key at all. .get(..., [])
    # gives an empty list in that case, instead of crashing.
    tool_calls = [_read_tool_call(item) for item in message.get("tool_calls", [])]

    return ModelReply(content=content, tool_calls=tool_calls)


def build_request(model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    """Build the request we send to Ollama's /api/chat.

    This only builds a dictionary. It does not touch the network, which is why
    we can test it without a model running. Sending the request is a separate,
    thinner step.

    Args:
        model: The Ollama model name, for example "qwen3:8b".
        messages: The conversation so far, a list of {"role", "content"} dicts.

    Returns:
        A dictionary ready to be turned into JSON and sent to Ollama.

    What a reader might get wrong:
        This version cannot offer tools to the model yet. Tools get their own
        failing test first.
    """
    return {
        "model": model,
        "messages": messages,
        # We want the whole reply as one document. Streaming sends it in
        # pieces, which is only worth the extra work if a customer needs words
        # to appear live.
        "stream": False,
        # qwen3 can write long reasoning before it answers. That is slow and
        # adds text we do not use. This is a trade off: thinking can help with
        # hard questions, so we may turn it on later, with a test.
        "think": False,
    }
