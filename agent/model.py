"""How our code talks to a model.

This file will hold the one small function that reaches the model (Ollama
running qwen3:8b). Keeping that in one place means we can swap in a different
model later by changing this file only, not the whole system.

Right now it holds the shape of a model reply. The function that calls the
model comes later, test first.
"""

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


# Where Ollama listens for chat requests on this computer. 11434 is Ollama's
# standard port. A customer's setup may use a different host, so this will
# become a setting we read from the environment. That gets its own test first.
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

# How long to wait for Ollama before giving up, in seconds. Without a limit, a
# stuck server would freeze the whole program forever. Two minutes is generous,
# because the first call after starting Ollama has to load the model first.
REQUEST_TIMEOUT_SECONDS = 120

# The model we chose in step 2. Same note as above: a setting later, a constant
# for now.
DEFAULT_MODEL = "qwen3:8b"


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


def build_request(
    model: str,
    messages: list[dict[str, str]],
    tools: list[dict[str, object]] | None = None,
) -> dict[str, Any]:
    """Build the request we send to Ollama's /api/chat.

    This only builds a dictionary. It does not touch the network, which is why
    we can test it without a model running. Sending the request is a separate,
    thinner step.

    Args:
        model: The Ollama model name, for example "qwen3:8b".
        messages: The conversation so far, a list of {"role", "content"} dicts.
        tools: The tools the model may ask for, in Ollama's format, or None to
            offer none. The same value the fake model accepts, so the two stay
            swappable.

    Returns:
        A dictionary ready to be turned into JSON and sent to Ollama.

    What a reader might get wrong:
        The "tools" key is left out when there are no tools, including when an
        empty list is passed. The model only learns about tools we pass in, so
        a missing key and an empty list both mean "no tools on offer". The empty
        list case has no test of its own yet.
    """
    request: dict[str, Any] = {
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

    # Only add the key when there is something to offer. A model can only ask
    # for a tool it has been told about, so this field is what makes tool calls
    # possible at all.
    if tools:
        request["tools"] = tools

    return request


def call_model(
    messages: list[dict[str, str]],
    tools: list[dict[str, object]] | None = None,
    *,
    post: Callable[[str, dict[str, Any]], dict[str, Any]],
    model: str = DEFAULT_MODEL,
) -> ModelReply:
    """Send a conversation to the model and return its reply.

    This is the one function the rest of the project uses to reach a model. It
    joins three steps: build the request, send it, check the answer.

    The first two parameters match FakeModel, so either can be handed to the
    agent loop. The parameters after the star are extra settings, and the loop
    never has to pass them.

    Args:
        messages: The conversation so far, a list of {"role", "content"} dicts.
        tools: The tools the model may ask for, or None to offer none.
        post: The function that really sends the request. It takes an address
            and a request dictionary and returns the reply as a dictionary. We
            pass it in so tests can hand in a pretend one and never touch a
            network.
        model: The Ollama model name.

    Returns:
        The model's reply as a ModelReply.

    Raises:
        ValueError: If the reply is malformed (raised by parse_reply).

    What a reader might get wrong:
        post has no default yet, so you must pass it. The real sender is the
        next piece we write, and we check it by hand, because tests never call
        a real model. After that, it becomes the default.
    """
    request = build_request(model, messages, tools)
    raw_reply = post(OLLAMA_CHAT_URL, request)
    return parse_reply(raw_reply)


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Send a dictionary to an address as JSON and return the JSON reply.

    This is the only function in the project that uses the network. Keeping it
    this small means that any network problem, like a slow server, a wrong
    address or a login header, has exactly one place to be fixed.

    It uses Python's built in tools, so we need no extra package. That keeps CI
    small.

    Args:
        url: The full address to send to.
        payload: The request, as a dictionary. It is turned into JSON here.

    Returns:
        The reply, turned from JSON back into a dictionary.

    Raises:
        RuntimeError: If the server cannot be reached, for example when Ollama
            is not running. The message says what to try.

    What a reader might get wrong:
        The timeout line has no test of its own yet. A test would need a server
        that never answers. Also, only an unreachable server is turned into a
        plain message. Other problems, like a server error reply, are not
        handled yet. Each gets its own test first.
    """
    # JSON is text, but the network sends bytes. .encode turns text into bytes.
    body = json.dumps(payload).encode("utf-8")

    # Content-Type tells the server what kind of data the body holds.
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    # The most common problem in the field is that the server is not running.
    # The raw network error for that is long and means little to most people.
    # URLError is the error Python raises when it cannot connect, so we catch
    # that one and say what is wrong and what to try.
    try:
        # "with" closes the connection for us when we are done, even on an error.
        with urllib.request.urlopen(
            request, timeout=REQUEST_TIMEOUT_SECONDS
        ) as response:
            return json.loads(response.read())
    except urllib.error.URLError as error:
        # "from error" keeps the original error attached. A person sees our
        # plain sentence first, and a developer can still dig into the details.
        raise RuntimeError(
            f"Could not reach the model server at {url}. "
            "Is Ollama running? Open the Ollama app, or run 'ollama serve'."
        ) from error
