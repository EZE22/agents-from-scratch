"""How our code talks to a model.

This file will hold the one small function that reaches the model (Ollama
running qwen3:8b). Keeping that in one place means we can swap in a different
model later by changing this file only, not the whole system.

Right now it holds only the shape of a model reply. The function that calls
the model comes later, test first.
"""

from dataclasses import dataclass


@dataclass
class ModelReply:
    """One reply from a model, in the single shape the rest of our code uses.

    Both the real model function and the fake model return this type. That is
    what lets the agent loop treat them the same, so tests can swap the fake in
    without the loop noticing.

    A dataclass writes the boring parts for us (the constructor and a readable
    printout), so we only list the fields.

    Attributes:
        content: The text the model said.
    """

    content: str
