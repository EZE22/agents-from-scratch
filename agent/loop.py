"""The agent loop.

An agent is a system where the model decides which tool to call and when to
stop, in a loop. The model controls the path. Our code runs the loop, sets
the limits and names the reason it stopped.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from agent.model import Message, ModelReply
from agent.tools import TOOL_SPECS, run_tool


# The logger for this file. A logger is a named place that log lines go to.
# Using the file's own name means a person can turn this file's lines up or
# down without touching the rest of the project. We never use print for this,
# because print cannot be filtered, silenced or sent to a log file.
logger = logging.getLogger(__name__)

# The most times we will ask the model in one run. A model can ask for tools
# forever, and every ask costs time and money. This cap is a hard stop that
# the model cannot talk its way past. Five is enough for one tool.
MAX_STEPS = 5


@dataclass
class AgentResult:
    """What one run of the agent gives back.

    answer is the final text from the model.
    stop_reason says why the run ended. Every run must end with a named
    reason, so a person reading the logs never has to guess.
    steps is how many times we asked the model.
    """

    answer: str
    stop_reason: str
    steps: int


def run_agent(question: str, model: Callable[..., ModelReply]) -> AgentResult:
    """Answer a question, using the tool when the model asks for it.

    This is an agent. The model decides whether to call a tool or to answer.
    Our code runs the loop, runs the tool and sets the limit.

    The model is passed in as a function. Tests pass FakeModel. A real run
    passes call_model. That keeps this loop free of any one model.

    Known gap, coming with its own test: the step limit stop is not logged
    yet.
    """
    messages: list[Message] = [{"role": "user", "content": question}]

    # range stops by itself, so this loop cannot run forever. A while True
    # loop would rely on the model to stop, and a model can fail to.
    for step in range(1, MAX_STEPS + 1):
        reply = model(messages, tools=TOOL_SPECS)

        # No tool request means the model chose to answer. We stop.
        if not reply.tool_calls:
            logger.info("step %d: model answered, stopping", step)
            return AgentResult(
                answer=reply.content, stop_reason="answered", steps=step
            )

        # Record what the model asked for, in the shape Ollama expects, so
        # the conversation shows the request and then its result. The
        # arguments stay a dict here. Some hosted models want a JSON string
        # instead, and that difference belongs in call_model, not in the loop.
        messages.append(
            {
                "role": "assistant",
                "content": reply.content,
                "tool_calls": [
                    {"function": {"name": call.name, "arguments": call.arguments}}
                    for call in reply.tool_calls
                ],
            }
        )

        # The model only asked. Our code runs the tool, then sends the
        # result back as a message so the model can use it on the next pass.
        for call in reply.tool_calls:
            # We log the tool name and the argument names, never the values.
            # The values are the user's own text and may be private.
            logger.info(
                "step %d: model asked for tool %s with arguments %s",
                step,
                call.name,
                sorted(call.arguments),
            )
            result = run_tool(call.name, call.arguments)
            logger.info(
                "step %d: tool %s returned %s", step, call.name, result.status
            )
            messages.append(
                {
                    "role": "tool",
                    "content": result.content,
                    "tool_name": call.name,
                }
            )

    # We only get here if every pass ended in a tool request. The cap
    # stopped the run, not the model.
    return AgentResult(answer="", stop_reason="step_limit", steps=MAX_STEPS)
