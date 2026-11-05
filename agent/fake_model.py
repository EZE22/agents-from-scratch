"""A stand in for the real model, used only in tests.

Why it exists:
    Real models are slow, answer differently each time, and need Ollama
    running. Tests must be fast and repeatable, so they use this instead.
    We script the replies in advance, which also lets us prove how our code
    behaves when the model misbehaves.
"""

from agent.model import ModelReply


class FakeModel:
    """Returns pre written replies, one per call, in the order given.

    It is called like a function: fake(messages). That is on purpose. The real
    model function is also called that way, so either can be handed to the
    agent loop.

    What a reader might get wrong:
        Each call uses up one reply. Calling it more times than you scripted
        replies is not handled yet. That behavior gets its own test later.
    """

    def __init__(self, replies: list[ModelReply]) -> None:
        """Store the script.

        Args:
            replies: The replies to hand back, first call gets the first one.
        """
        # list(...) makes our own copy. Without it, later changes to the
        # caller's list would silently change our script too.
        self._replies = list(replies)

        # Which reply to hand out next. Starts at the first one.
        self._next_index = 0

    def __call__(self, messages: list[dict[str, str]]) -> ModelReply:
        """Return the next scripted reply.

        Args:
            messages: The conversation so far, a list of {"role", "content"}
                dicts. The fake ignores it, but the real model reads it, so
                the fake must accept it to stay swappable.

        Returns:
            The next ModelReply from the script.
        """
        reply = self._replies[self._next_index]
        self._next_index += 1
        return reply
