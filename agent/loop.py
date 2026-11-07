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

# The most refused replies we accept in one run. A model that is stuck
# sending broken tool calls will not fix itself on the tenth try, and every
# try costs time and money. It is lower than MAX_STEPS on purpose, so this
# limit is reached first when every reply is bad. Three is a starting point,
# and the model comparison will show whether it is a good one.
MAX_BAD_REPLIES = 3

# The most failed tool calls we accept in one run. A failed call means the
# model asked for a tool that does not exist, or gave it the wrong inputs.
# The model gets a readable message each time, so a few tries are fair. A
# model that keeps failing is stuck, and each try costs time and money.
MAX_TOOL_FAILURES = 3


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
    """
    messages: list[Message] = [{"role": "user", "content": question}]

    # How many replies have been refused so far in this run.
    bad_replies = 0

    # How many tool calls have failed so far in this run.
    tool_failures = 0

    # range stops by itself, so this loop cannot run forever. A while True
    # loop would rely on the model to stop, and a model can fail to.
    for step in range(1, MAX_STEPS + 1):
        # Only the model call sits inside the try. A ValueError from anywhere
        # else in the loop is a bug in our code, and we want to see it.
        try:
            reply = model(messages, tools=TOOL_SPECS)
        except ValueError as error:
            # The model call raises ValueError when parse_reply refuses a
            # bad reply. We never got a parsed reply, so there is nothing
            # to record from the model. We tell it what went wrong and ask
            # again. The next pass of the loop is the retry. We log the
            # step and the status only, never the error text.
            bad_replies += 1
            logger.warning(
                "step %d: model reply was refused (%d of %d)",
                step,
                bad_replies,
                MAX_BAD_REPLIES,
            )

            # Out of tries. The code stops the run, not the model, and it
            # names the reason so the logs never leave anyone guessing.
            if bad_replies >= MAX_BAD_REPLIES:
                return AgentResult(
                    answer="", stop_reason="bad_replies", steps=step
                )

            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your last reply could not be read: {error} "
                        "Reply again with a valid tool call or a plain "
                        "text answer."
                    ),
                }
            )
            continue

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

            # The failed result is already in the conversation, so the model
            # has been told. Now we count it. The code stops the run, not the
            # model, and names the reason. A reader might get wrong: one
            # reply with two failed calls counts as two failures.
            if result.status == "failed":
                tool_failures += 1
                if tool_failures >= MAX_TOOL_FAILURES:
                    logger.warning(
                        "stopped after %d failed tool calls", tool_failures
                    )
                    return AgentResult(
                        answer="", stop_reason="tool_failures", steps=step
                    )

    # We only get here if every pass ended in a tool request. The cap
    # stopped the run, not the model. This is a warning, not info, because
    # the model never answered and someone watching the logs should see it.
    logger.warning(
        "stopped at the step limit of %d without a final answer", MAX_STEPS
    )
    return AgentResult(answer="", stop_reason="step_limit", steps=MAX_STEPS)
