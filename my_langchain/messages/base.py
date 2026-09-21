"""Chat message primitives.

Messages are the currency of chat models: a conversation is a list of messages
with distinct roles. We model them on top of pydantic v2 so that each message
is validated, serializable, and cheap to compare or copy.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "MessageContent",
    "BaseMessage",
    "HumanMessage",
    "AIMessage",
    "SystemMessage",
    "FunctionMessage",
    "ToolMessage",
    "ChatMessage",
    "message_to_dict",
    "messages_from_dict",
]

MessageContent = str | list[str | dict[str, Any]]
"""Message payload: either a plain string or a list of content blocks."""


class BaseMessage(BaseModel):
    """Base class for all messages exchanged with a chat model."""

    content: MessageContent
    """The message body."""

    additional_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Provider-specific payload that does not belong to the standard fields."""

    response_metadata: dict[str, Any] = Field(default_factory=dict)
    """Metadata about the provider response that produced this message."""

    type: str
    """Discriminator identifying the concrete message class."""

    name: str | None = None
    """Optional name of the author of the message."""

    id: str | None = None
    """Optional unique identifier for the message."""

    model_config = ConfigDict(extra="allow")

    @property
    def text(self) -> str:
        """Return the concatenated text of the message content."""
        if isinstance(self.content, str):
            return self.content
        parts: list[str] = []
        for block in self.content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)


class HumanMessage(BaseMessage):
    """A message authored by the user."""

    type: str = "human"


class AIMessage(BaseMessage):
    """A message authored by the model."""

    type: str = "ai"
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    invalid_tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage_metadata: dict[str, Any] | None = None


class SystemMessage(BaseMessage):
    """A message that primes the model's behavior."""

    type: str = "system"


class FunctionMessage(BaseMessage):
    """The result of calling a legacy function."""

    type: str = "function"
    name: str


class ToolMessage(BaseMessage):
    """The result of calling a tool."""

    type: str = "tool"
    tool_call_id: str


class ChatMessage(BaseMessage):
    """A message with an arbitrary role, useful for non-standard providers."""

    type: str = "chat"
    role: str


def _message_type_to_class() -> dict[str, type[BaseMessage]]:
    """Return the mapping from message type identifiers to classes."""
    from my_langchain.messages import chunk as chunk_module

    return {
        "human": HumanMessage,
        "ai": AIMessage,
        "system": SystemMessage,
        "function": FunctionMessage,
        "tool": ToolMessage,
        "chat": ChatMessage,
        "HumanMessageChunk": chunk_module.HumanMessageChunk,
        "AIMessageChunk": chunk_module.AIMessageChunk,
        "SystemMessageChunk": chunk_module.SystemMessageChunk,
        "ToolMessageChunk": chunk_module.ToolMessageChunk,
        "ChatMessageChunk": chunk_module.ChatMessageChunk,
    }


def message_to_dict(message: BaseMessage) -> dict[str, Any]:
    """Serialize a message into a plain dictionary."""
    return {"type": message.type, "data": message.model_dump()}


def messages_from_dict(messages: list[dict[str, Any]]) -> list[BaseMessage]:
    """Reconstruct messages from dictionaries produced by :func:`message_to_dict`."""
    return [_message_from_dict(message) for message in messages]


def _message_from_dict(message: dict[str, Any]) -> BaseMessage:
    """Reconstruct a single message from its serialized dictionary form."""
    type_to_class = _message_type_to_class()
    message_type = message["type"]
    try:
        cls = type_to_class[message_type]
    except KeyError as err:
        raise ValueError(f"Unknown message type: {message_type!r}") from err
    data = message.get("data", {})
    return cls(**{**data, "type": message_type})
