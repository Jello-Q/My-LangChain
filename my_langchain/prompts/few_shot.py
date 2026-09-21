"""Few-shot prompt templates.

Few-shot prompting prepends worked examples before the real request. These
templates render a fixed list of ``examples`` through an ``example_prompt``,
optionally wrapped by a ``prefix`` and ``suffix``.
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, Field, model_validator

from my_langchain.core.exceptions import InputValidationError
from my_langchain.messages.base import BaseMessage
from my_langchain.prompts.chat import BaseMessagePromptTemplate, ChatPromptTemplate
from my_langchain.prompts.prompt_template import PromptTemplate, get_template_variables
from my_langchain.prompts.prompt_values import PromptValue, StringPromptValue

__all__ = [
    "FewShotPromptTemplate",
    "FewShotChatMessagePromptTemplate",
]


class FewShotPromptTemplate(BaseModel):
    """A string prompt that includes a fixed set of worked examples."""

    example_prompt: PromptTemplate
    """The template used to render each example dictionary."""

    examples: list[dict[str, Any]]
    """Example dictionaries substituted into ``example_prompt``."""

    prefix: str = ""
    """Text placed before the examples."""

    suffix: str = ""
    """Text placed after the examples, usually containing the real question."""

    input_variables: list[str] = Field(default_factory=list)
    """Variables required by ``prefix`` and ``suffix``; inferred when omitted."""

    example_separator: str = "\n\n"
    """Separator inserted between examples and surrounding text."""

    @model_validator(mode="after")
    def _infer_input_variables(self) -> Self:
        """Infer input variables from the prefix and suffix templates."""
        if not self.input_variables:
            variables: list[str] = []
            for text in (self.prefix, self.suffix):
                for variable in get_template_variables(text):
                    if variable not in variables:
                        variables.append(variable)
            self.input_variables = variables
        return self

    def format(self, **kwargs: Any) -> str:
        """Render the prefix, examples, and suffix into one string."""
        missing = [variable for variable in self.input_variables if variable not in kwargs]
        if missing:
            raise InputValidationError(f"Missing required input variables for prompt: {missing}")
        rendered_examples = [self.example_prompt.format(**example) for example in self.examples]
        pieces: list[str] = []
        if self.prefix:
            pieces.append(self.prefix.format(**kwargs))
        pieces.append(self.example_separator.join(rendered_examples))
        if self.suffix:
            pieces.append(self.suffix.format(**kwargs))
        return self.example_separator.join(pieces)

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Render the prompt as a :class:`StringPromptValue`."""
        return StringPromptValue(text=self.format(**kwargs))


class FewShotChatMessagePromptTemplate(BaseMessagePromptTemplate):
    """A chat prompt that expands to a fixed set of example messages."""

    example_prompt: ChatPromptTemplate
    """The chat template used to render each example dictionary."""

    examples: list[dict[str, Any]]
    """Example dictionaries substituted into ``example_prompt``."""

    @property
    def input_variables(self) -> list[str]:
        """Few-shot examples are fixed, so no caller variables are required."""
        return []

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Render every example into concrete messages."""
        messages: list[BaseMessage] = []
        for example in self.examples:
            messages.extend(self.example_prompt.format_messages(**example))
        return messages
