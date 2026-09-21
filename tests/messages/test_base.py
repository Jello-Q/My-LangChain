"""Tests for message primitives."""

import pytest

from my_langchain.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_to_dict,
    messages_from_dict,
)


def test_human_message_has_type_and_text() -> None:
    message = HumanMessage(content="hello")
    assert message.type == "human"
    assert message.text == "hello"


def test_text_concatenates_list_content_blocks() -> None:
    message = HumanMessage(content=[{"type": "text", "text": "foo"}, "bar"])
    assert message.text == "foobar"


def test_ai_message_carries_tool_calls() -> None:
    message = AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": "1"}])
    assert message.tool_calls[0]["name"] == "search"


def test_tool_message_requires_tool_call_id() -> None:
    message = ToolMessage(content="42", tool_call_id="call-1")
    assert message.tool_call_id == "call-1"


def test_chat_message_stores_arbitrary_role() -> None:
    message = ChatMessage(content="hi", role="developer")
    assert message.role == "developer"


@pytest.mark.parametrize(
    "message",
    [
        HumanMessage(content="u"),
        AIMessage(content="a"),
        SystemMessage(content="s"),
        ToolMessage(content="t", tool_call_id="id"),
    ],
)
def test_roundtrip_serialization(message: BaseMessage) -> None:
    restored = messages_from_dict([message_to_dict(message)])
    assert restored == [message]


def test_unknown_message_type_raises() -> None:
    with pytest.raises(ValueError, match="Unknown message type"):
        messages_from_dict([{"type": "bogus", "data": {"content": "x"}}])
