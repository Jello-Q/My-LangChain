"""Tests for chat prompt templates."""

import pytest

from my_langchain.core.exceptions import InputValidationError
from my_langchain.messages import AIMessage, HumanMessage, SystemMessage
from my_langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)


def test_from_messages_renders_tuples_and_placeholder() -> None:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "Answer in {language}."),
        ]
    )
    messages = prompt.format_messages(
        history=[HumanMessage(content="hi"), AIMessage(content="hello")],
        language="English",
    )
    assert [type(message) for message in messages] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
        HumanMessage,
    ]
    assert messages[-1].content == "Answer in English."


def test_input_variables_are_inferred_from_templates() -> None:
    prompt = ChatPromptTemplate.from_messages(
        [("system", "ctx: {ctx}"), MessagesPlaceholder(variable_name="history")]
    )
    assert prompt.input_variables == ["ctx", "history"]


def test_missing_variable_raises() -> None:
    prompt = ChatPromptTemplate.from_messages([("human", "Answer in {language}.")])
    with pytest.raises(InputValidationError, match="Missing required input variables"):
        prompt.format_messages()


def test_direct_constructor_coerces_tuples() -> None:
    prompt = ChatPromptTemplate(messages=[("human", "hi {name}")])
    assert prompt.format_messages(name="Ada")[0].content == "hi Ada"


def test_format_prompt_returns_chat_prompt_value() -> None:
    prompt = ChatPromptTemplate.from_messages([("system", "sys")])
    value = prompt.format_prompt()
    assert value.to_messages() == [SystemMessage(content="sys")]
    assert value.to_string() == "sys"


def test_optional_placeholder_can_be_omitted() -> None:
    prompt = ChatPromptTemplate.from_messages(
        [MessagesPlaceholder(variable_name="history", optional=True)]
    )
    assert prompt.input_variables == []
    assert prompt.format_messages() == []


def test_partial_binds_variable() -> None:
    prompt = ChatPromptTemplate.from_messages([("human", "{greeting}, {name}!")])
    partial = prompt.partial(greeting="Hello")
    assert partial.input_variables == ["name"]
    assert partial.format_messages(name="Ada")[0].content == "Hello, Ada!"


def test_message_prompt_template_from_template() -> None:
    template = SystemMessagePromptTemplate.from_template("You are {role}.")
    assert template.input_variables == ["role"]
    assert template.format_messages(role="tester") == [SystemMessage(content="You are tester.")]
