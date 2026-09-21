"""Chat prompt templates.

A chat prompt is a sequence of message templates. Each entry renders to one or
more messages; :class:`MessagesPlaceholder` injects a list of prior messages
(conversation history), and tuples like ``("human", "{question}")`` are
shorthand for a human message template.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from my_langchain.core.exceptions import InputValidationError
from my_langchain.messages.base import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from my_langchain.prompts.prompt_template import PromptTemplate
from my_langchain.prompts.prompt_values import ChatPromptValue, PromptValue

__all__ = [
    "MessageLike",
    "BaseMessagePromptTemplate",
    "HumanMessagePromptTemplate",
    "AIMessagePromptTemplate",
    "SystemMessagePromptTemplate",
    "ChatMessagePromptTemplate",
    "MessagesPlaceholder",
    "ChatPromptTemplate",
]


class BaseMessagePromptTemplate(BaseModel, ABC):
    """A template that renders to one or more chat messages."""

    @property
    @abstractmethod
    def input_variables(self) -> list[str]:
        """The variables this template requires."""

    @abstractmethod
    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Render the template into concrete messages."""


def _format_message_prompt(prompt: PromptTemplate, **kwargs: Any) -> str:
    """Format an underlying string prompt, normalizing errors."""
    return prompt.format(**kwargs)


class HumanMessagePromptTemplate(BaseMessagePromptTemplate):
    """Renders to a :class:`HumanMessage`."""

    prompt: PromptTemplate

    @property
    def input_variables(self) -> list[str]:
        """Variables required by the wrapped prompt."""
        return self.prompt.input_variables

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Render the template into a single human message."""
        return [HumanMessage(content=_format_message_prompt(self.prompt, **kwargs))]

    @classmethod
    def from_template(cls, template: str, **kwargs: Any) -> Self:
        """Build a human message template from a raw string."""
        return cls(prompt=PromptTemplate.from_template(template, **kwargs))


class AIMessagePromptTemplate(BaseMessagePromptTemplate):
    """Renders to an :class:`AIMessage`."""

    prompt: PromptTemplate

    @property
    def input_variables(self) -> list[str]:
        """Variables required by the wrapped prompt."""
        return self.prompt.input_variables

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Render the template into a single AI message."""
        return [AIMessage(content=_format_message_prompt(self.prompt, **kwargs))]

    @classmethod
    def from_template(cls, template: str, **kwargs: Any) -> Self:
        """Build an AI message template from a raw string."""
        return cls(prompt=PromptTemplate.from_template(template, **kwargs))


class SystemMessagePromptTemplate(BaseMessagePromptTemplate):
    """Renders to a :class:`SystemMessage`."""

    prompt: PromptTemplate

    @property
    def input_variables(self) -> list[str]:
        """Variables required by the wrapped prompt."""
        return self.prompt.input_variables

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Render the template into a single system message."""
        return [SystemMessage(content=_format_message_prompt(self.prompt, **kwargs))]

    @classmethod
    def from_template(cls, template: str, **kwargs: Any) -> Self:
        """Build a system message template from a raw string."""
        return cls(prompt=PromptTemplate.from_template(template, **kwargs))


class ChatMessagePromptTemplate(BaseMessagePromptTemplate):
    """Renders to a :class:`ChatMessage` with a fixed role."""

    prompt: PromptTemplate
    role: str

    @property
    def input_variables(self) -> list[str]:
        """Variables required by the wrapped prompt."""
        return self.prompt.input_variables

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Render the template into a single chat message."""
        return [ChatMessage(content=_format_message_prompt(self.prompt, **kwargs), role=self.role)]

    @classmethod
    def from_template(cls, template: str, *, role: str, **kwargs: Any) -> Self:
        """Build a chat message template with the given role."""
        return cls(prompt=PromptTemplate.from_template(template, **kwargs), role=role)


class MessagesPlaceholder(BaseMessagePromptTemplate):
    """Injects a list of messages supplied at format time."""

    variable_name: str
    optional: bool = False

    @property
    def input_variables(self) -> list[str]:
        """The placeholder variable, unless the placeholder is optional."""
        return [] if self.optional else [self.variable_name]

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Return the messages bound to this placeholder's variable."""
        if self.variable_name in kwargs:
            value = kwargs[self.variable_name]
            if isinstance(value, BaseMessage):
                return [value]
            return list(value)
        if self.optional:
            return []
        raise InputValidationError(f"Missing required input variable: {self.variable_name!r}")


MessageLike = BaseMessagePromptTemplate | BaseMessage | str | tuple[str, str]
"""Anything :meth:`ChatPromptTemplate.from_messages` accepts."""


def _convert_message_like(item: MessageLike) -> BaseMessagePromptTemplate | BaseMessage:
    """Normalize a message-like shorthand into a template or message."""
    if isinstance(item, (BaseMessagePromptTemplate, BaseMessage)):
        return item
    if isinstance(item, str):
        return HumanMessagePromptTemplate.from_template(item)
    if isinstance(item, tuple) and len(item) == 2:
        role, template = item
        normalized = role.lower()
        if normalized in {"human", "user"}:
            return HumanMessagePromptTemplate.from_template(template)
        if normalized in {"ai", "assistant"}:
            return AIMessagePromptTemplate.from_template(template)
        if normalized == "system":
            return SystemMessagePromptTemplate.from_template(template)
        return ChatMessagePromptTemplate.from_template(template, role=role)
    raise InputValidationError(f"Unsupported message-like entry: {item!r}")


class ChatPromptTemplate(BaseModel):
    """An ordered sequence of message templates rendered with keyword arguments."""

    messages: list[MessageLike]
    input_variables: list[str] = Field(default_factory=list)
    partial_variables: dict[str, Any] = Field(default_factory=dict)

    @field_validator("messages", mode="before")
    @classmethod
    def _coerce_messages(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [_convert_message_like(item) for item in value]
        return value

    def _resolved_messages(self) -> list[BaseMessagePromptTemplate | BaseMessage]:
        """Return messages with any shorthand entries converted to templates."""
        return [_convert_message_like(item) for item in self.messages]

    @model_validator(mode="after")
    def _infer_input_variables(self) -> Self:
        """Collect the input variables required by all message templates."""
        if not self.input_variables:
            variables: list[str] = []
            for message in self._resolved_messages():
                if isinstance(message, BaseMessage):
                    continue
                for variable in message.input_variables:
                    if variable not in variables:
                        variables.append(variable)
            self.input_variables = variables
        return self

    @classmethod
    def from_messages(cls, messages: Sequence[MessageLike]) -> Self:
        """Build a chat prompt from messages, templates, tuples, or strings."""
        return cls(messages=list(messages))

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Render every message template into concrete messages."""
        merged = {**self.partial_variables, **kwargs}
        missing = [variable for variable in self.input_variables if variable not in merged]
        if missing:
            raise InputValidationError(f"Missing required input variables for prompt: {missing}")
        result: list[BaseMessage] = []
        for message in self._resolved_messages():
            if isinstance(message, BaseMessage):
                result.append(message)
            else:
                result.extend(message.format_messages(**merged))
        return result

    def format(self, **kwargs: Any) -> str:
        """Render the prompt to a single string."""
        return "\n\n".join(message.text for message in self.format_messages(**kwargs))

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Render the prompt as a :class:`ChatPromptValue`."""
        return ChatPromptValue(messages=self.format_messages(**kwargs))

    def partial(self, **kwargs: Any) -> ChatPromptTemplate:
        """Return a copy with some input variables pre-bound."""
        merged = {**self.partial_variables, **kwargs}
        remaining = [variable for variable in self.input_variables if variable not in merged]
        return self.model_copy(update={"partial_variables": merged, "input_variables": remaining})
