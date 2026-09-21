"""Streaming message chunks.

Chat models stream output token by token. Each streamed piece is a message
*chunk*, and chunks of the same kind can be added together to rebuild the full
message. This module defines those chunk types and the merge logic.
"""

from __future__ import annotations

from typing import Any

from my_langchain.messages.base import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    MessageContent,
    SystemMessage,
    ToolMessage,
)

__all__ = [
    "BaseMessageChunk",
    "HumanMessageChunk",
    "AIMessageChunk",
    "SystemMessageChunk",
    "ToolMessageChunk",
    "ChatMessageChunk",
]


def _merge_content(first: MessageContent, second: MessageContent) -> MessageContent:
    """Concatenate two message contents, preserving list content blocks."""
    if isinstance(first, str) and isinstance(second, str):
        return first + second
    first_blocks: list[str | dict[str, Any]] = [first] if isinstance(first, str) else list(first)
    second_blocks: list[str | dict[str, Any]] = (
        [second] if isinstance(second, str) else list(second)
    )
    return first_blocks + second_blocks


def _merge_dicts(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Merge two dictionaries, preferring the second when both are non-empty."""
    if not first:
        return second
    if not second:
        return first
    return {**first, **second}


class BaseMessageChunk(BaseMessage):
    """A partial message that can be merged with a later chunk."""

    def __add__(self, other: object) -> BaseMessageChunk:
        """Merge this chunk with another chunk of the same kind."""
        if not isinstance(other, BaseMessageChunk):
            return NotImplemented
        merged = self.model_dump()
        merged["content"] = _merge_content(self.content, other.content)
        merged["additional_kwargs"] = _merge_dicts(self.additional_kwargs, other.additional_kwargs)
        merged["response_metadata"] = _merge_dicts(self.response_metadata, other.response_metadata)
        merged["id"] = self.id or other.id
        merged["name"] = self.name or other.name
        return self.__class__(**merged)


class HumanMessageChunk(HumanMessage, BaseMessageChunk):
    """A partial :class:`HumanMessage`."""

    type: str = "HumanMessageChunk"


class AIMessageChunk(AIMessage, BaseMessageChunk):
    """A partial :class:`AIMessage`."""

    type: str = "AIMessageChunk"


class SystemMessageChunk(SystemMessage, BaseMessageChunk):
    """A partial :class:`SystemMessage`."""

    type: str = "SystemMessageChunk"


class ToolMessageChunk(ToolMessage, BaseMessageChunk):
    """A partial :class:`ToolMessage`."""

    type: str = "ToolMessageChunk"


class ChatMessageChunk(ChatMessage, BaseMessageChunk):
    """A partial :class:`ChatMessage`."""

    type: str = "ChatMessageChunk"
