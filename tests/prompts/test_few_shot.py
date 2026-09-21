"""Tests for few-shot prompt templates."""

import pytest

from my_langchain.core.exceptions import InputValidationError
from my_langchain.messages import AIMessage, HumanMessage
from my_langchain.prompts.chat import ChatPromptTemplate
from my_langchain.prompts.few_shot import (
    FewShotChatMessagePromptTemplate,
    FewShotPromptTemplate,
)
from my_langchain.prompts.prompt_template import PromptTemplate


def _make_string_few_shot() -> FewShotPromptTemplate:
    return FewShotPromptTemplate(
        example_prompt=PromptTemplate.from_template("{input} -> {output}"),
        examples=[{"input": "1+1", "output": "2"}, {"input": "2+2", "output": "4"}],
        prefix="Examples:",
        suffix="Input: {question}\nOutput:",
    )


def test_input_variables_come_from_prefix_and_suffix() -> None:
    assert _make_string_few_shot().input_variables == ["question"]


def test_format_includes_examples_and_suffix() -> None:
    rendered = _make_string_few_shot().format(question="3+3")
    assert rendered == "Examples:\n\n1+1 -> 2\n\n2+2 -> 4\n\nInput: 3+3\nOutput:"


def test_missing_variable_raises() -> None:
    with pytest.raises(InputValidationError, match="Missing required input variables"):
        _make_string_few_shot().format()


def test_chat_few_shot_renders_example_messages() -> None:
    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")]),
        examples=[{"input": "1+1", "output": "2"}],
    )
    prompt = ChatPromptTemplate.from_messages([few_shot, ("human", "{question}")])
    messages = prompt.format_messages(question="3+3")
    assert [type(message) for message in messages] == [HumanMessage, AIMessage, HumanMessage]
    assert messages[1].content == "2"
    assert messages[-1].content == "3+3"
