"""Prompt values.

A :class:`PromptValue` is the *formatted* result of a prompt template. It has
two equivalent views: a string (for completion-style models) and a list of
messages (for chat models). Templates produce prompt values; models consume
them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from my_langchain.messages.base import BaseMessage, HumanMessage

__all__ = [
    "PromptValue",
    "StringPromptValue",
    "ChatPromptValue",
]


class PromptValue(BaseModel, ABC):
    """The formatted output of a prompt, renderable as text or messages."""

    @abstractmethod
    def to_string(self) -> str:
        """Render the prompt as a single string."""

    @abstractmethod
    def to_messages(self) -> list[BaseMessage]:
        """Render the prompt as a list of chat messages."""


class StringPromptValue(PromptValue):
    """A prompt value backed by a plain string."""

    text: str

    def to_string(self) -> str:
        """Return the underlying text."""
        return self.text

    def to_messages(self) -> list[BaseMessage]:
        """Wrap the text in a single human message."""
        return [HumanMessage(content=self.text)]


class ChatPromptValue(PromptValue):
    """A prompt value backed by a list of messages."""

    messages: list[BaseMessage]

    def to_string(self) -> str:
        """Join the messages' text with double newlines."""
        return "\n\n".join(message.text for message in self.messages)

    def to_messages(self) -> list[BaseMessage]:
        """Return a copy of the underlying messages."""
        return list(self.messages)
