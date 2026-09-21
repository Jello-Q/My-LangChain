"""Tests for string prompt templates."""

import pytest

from my_langchain.core.exceptions import InputValidationError
from my_langchain.prompts.prompt_template import PromptTemplate, get_template_variables


def test_get_template_variables_preserves_order_and_dedupes() -> None:
    assert get_template_variables("{a} then {b} then {a}") == ["a", "b"]


def test_get_template_variables_handles_escaped_braces() -> None:
    assert get_template_variables("{{literal}} {real}") == ["real"]


def test_from_template_infers_input_variables() -> None:
    template = PromptTemplate.from_template("Tell me about {topic}")
    assert template.input_variables == ["topic"]


def test_format_substitutes_variables() -> None:
    template = PromptTemplate.from_template("Tell me about {topic}")
    assert template.format(topic="cats") == "Tell me about cats"


def test_format_raises_when_variable_missing() -> None:
    template = PromptTemplate.from_template("Tell me about {topic}")
    with pytest.raises(InputValidationError, match="Missing required input variables"):
        template.format()


def test_format_prompt_returns_string_prompt_value() -> None:
    template = PromptTemplate.from_template("Hi {name}")
    value = template.format_prompt(name="Ada")
    assert value.to_string() == "Hi Ada"
    assert value.to_messages()[0].content == "Hi Ada"


def test_partial_binds_variable_and_drops_it_from_inputs() -> None:
    template = PromptTemplate.from_template("{greeting}, {name}!")
    partial = template.partial(greeting="Hello")
    assert partial.input_variables == ["name"]
    assert partial.format(name="Ada") == "Hello, Ada!"
