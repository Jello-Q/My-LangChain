"""Tests for prompt values."""

import pytest

from my_langchain.messages import HumanMessage, SystemMessage
from my_langchain.prompts.prompt_values import (
    ChatPromptValue,
    PromptValue,
    StringPromptValue,
)


def test_prompt_value_is_abstract() -> None:
    with pytest.raises(TypeError):
        PromptValue()  # type: ignore[abstract]


def test_string_prompt_value_round_trips_text() -> None:
    value = StringPromptValue(text="hello")
    assert value.to_string() == "hello"
    assert value.to_messages() == [HumanMessage(content="hello")]


def test_chat_prompt_value_joins_text_and_returns_messages() -> None:
    messages = [SystemMessage(content="sys"), HumanMessage(content="hi")]
    value = ChatPromptValue(messages=messages)
    assert value.to_messages() == messages
    assert value.to_string() == "sys\n\nhi"
