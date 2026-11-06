"""Tests for build_request, the function that prepares what we send to Ollama.

What this file is about:
    Before we can call the model, we need a request in the exact shape Ollama
    expects. build_request makes that request. It only builds a dictionary and
    never touches the network, so we can test it fast, with no model running.

Why this matters to a customer:
    The request shape is one of two places that know the details of Ollama (the
    other is parse_reply). Keeping both in one file means moving to a different
    model host is a contained change, not a rewrite.
"""

# build_request does not exist yet, so this import is the first thing that
# fails. That is the red step.
from agent.model import build_request


def test_build_request_for_a_plain_chat_has_no_tools() -> None:
    """Protects the basic request: a model name and a conversation, no tools.

    Why it matters:
        Every call to the model starts from this shape. If a field is wrong,
        Ollama refuses the request or answers differently than we expect.

    What a reader might get wrong:
        There is no "tools" key at all when we offer no tools. We leave it out
        instead of sending an empty value, because the safest request is the
        smallest one that says what we mean.

    About the two fixed settings:
        stream is False because we want the whole reply as one document.
        Streaming sends it in pieces, which is only worth the extra work if a
        customer needs words to appear live.
        think is False because qwen3 can write long reasoning before it
        answers. That is slow and adds text we do not use. We saw this in our
        first real call. It is a trade off: thinking can improve hard answers,
        so we may turn it on later, with a test.
    """
    messages = [{"role": "user", "content": "hi"}]

    expected = {
        "model": "qwen3:8b",
        "messages": messages,
        "stream": False,
        "think": False,
    }

    assert build_request("qwen3:8b", messages) == expected


def test_build_request_includes_tools_when_they_are_offered() -> None:
    """Protects the moment that lets a model ask for a tool.

    Why it matters:
        A model can only ask for a tool it has been told about. The "tools"
        field is how we tell it. If it is missing or misplaced, the model will
        guess an answer in words instead of asking our code to run the tool,
        and nothing downstream will warn us.

    What a reader might get wrong:
        We pass the tool descriptions through unchanged. Each one holds a name,
        a plain language description and a schema for the inputs. The model
        reads the description to decide whether to use the tool, so how well it
        is written matters as much as the code behind it.
    """
    messages = [{"role": "user", "content": "What is the weather in Atlanta?"}]

    # A made up tool in the format Ollama expects. This is the same shape we
    # sent in our real curl test, where the model asked for it.
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]

    expected = {
        "model": "qwen3:8b",
        "messages": messages,
        "stream": False,
        "think": False,
        "tools": tools,
    }

    assert build_request("qwen3:8b", messages, tools=tools) == expected
