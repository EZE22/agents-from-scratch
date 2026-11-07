"""A stand in for the real model, used only in tests.

Why it exists:
    Real models are slow, answer differently each time, and need Ollama
    running. Tests must be fast and repeatable, so they use this instead.
    We script the replies in advance, which also lets us prove how our code
    behaves when the model misbehaves.
"""

from agent.model import Message, ModelReply


class FakeModel:
    """Returns pre written replies, one per call, in the order given.

    It is called like a function: fake(messages, tools=None). That is on
    purpose. The real model function is called the same way, so either can be
    handed to the agent loop.

    It also records every call it receives in the calls attribute, so a test
    can check exactly what the code under test sent to the model.

    What a reader might get wrong:
        Each call uses up one reply. Calling it more times than you scripted
        replies raises a RuntimeError. That is deliberate: a loop that keeps
        asking the model is a bug we want to see, not hide.

    Attributes:
        calls: One entry per call, oldest first. Each entry is a pair:
            (messages, tools), exactly as the caller passed them.
    """

    def __init__(self, replies: list[ModelReply | Exception]) -> None:
        """Store the script.

        Args:
            replies: The replies to hand back, first call gets the first one.
                An item can also be an error. The fake raises it on that call
                instead of returning it.
        """
        # list(...) makes our own copy. Without it, later changes to the
        # caller's list would silently change our script too.
        self._replies = list(replies)

        # Which reply to hand out next. Starts at the first one.
        self._next_index = 0

        # What we were sent on each call. Starts empty.
        self.calls: list[
            tuple[list[Message], list[dict[str, object]] | None]
        ] = []

    def __call__(
        self,
        messages: list[Message],
        tools: list[dict[str, object]] | None = None,
    ) -> ModelReply:
        """Record the call, then return the next scripted reply.

        Args:
            messages: The conversation so far, a list of {"role", "content"}
                dicts. The fake does not read it to decide what to say, but it
                records it, and it must accept it to stay swappable with the
                real model.
            tools: The tools on offer to the model, or None for no tools. The
                default of None is why older tests that pass only messages
                still work.

        Returns:
            The next ModelReply from the script.

        Raises:
            RuntimeError: If every scripted reply has already been used.
            Exception: The scripted error itself, when the next item in the
                script is an error.
        """
        # Record a snapshot of the messages, not the live list. The agent loop
        # keeps adding to one growing list. If we stored that same list, every
        # recorded call would quietly show the final conversation.
        #
        # What a reader might get wrong: list(...) copies the list, not the
        # message dicts inside it. That is enough while nothing edits a message
        # in place. If that ever changes, a test comes first, then a deeper copy.
        #
        # The tools are stored as given, because nothing changes that list
        # after the call. The same rule applies: if that changes, test first.
        self.calls.append((list(messages), tools))

        # Check before reading. Without this, running off the end of the list
        # raises a bare IndexError that points at the inside of this class.
        # A reader debugging a loop needs to hear "the model was asked more
        # times than the test expected", so we say that in plain words.
        if self._next_index >= len(self._replies):
            raise RuntimeError(
                "FakeModel has no scripted replies left. "
                f"It was called {self._next_index + 1} times "
                f"but only {len(self._replies)} replies were scripted."
            )

        reply = self._replies[self._next_index]
        self._next_index += 1

        # A scripted error is raised, not returned. This mimics the real
        # call_model, which raises ValueError when a reply is refused.
        # The index moved on above, so the error uses up one call, and the
        # next call gets the next item in the script.
        if isinstance(reply, Exception):
            raise reply
        return reply
