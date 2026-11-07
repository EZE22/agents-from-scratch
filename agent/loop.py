"""The agent loop.

An agent is a system where the model decides which tool to call and when to
stop, in a loop. The model controls the path. Our code runs the loop, sets
the limits and names the reason it stopped.
"""

from collections.abc import Callable
from dataclasses import dataclass

from agent.model import ModelReply


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
    """Answer a question by asking the model.

    The model is passed in as a function. Tests pass FakeModel. A real run
    passes call_model. That keeps this loop free of any one model.

    Right now this asks the model once and returns its text, because the
    first test only covers that path. The tool calls and the step limit
    come with their own tests. A reader should not mistake this for the
    finished loop.
    """
    messages = [{"role": "user", "content": question}]
    reply = model(messages)
    return AgentResult(answer=reply.content, stop_reason="answered", steps=1)
