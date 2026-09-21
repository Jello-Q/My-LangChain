"""String prompt templates.

A :class:`PromptTemplate` is a string containing ``{placeholders}`` plus the
list of variables it needs. Formatting validates that every required variable
was supplied, so mistakes surface as :class:`InputValidationError` instead of a
confusing ``KeyError`` deep inside ``str.format``.
"""

from __future__ import annotations

import string
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator

from my_langchain.core.exceptions import InputValidationError
from my_langchain.prompts.prompt_values import PromptValue, StringPromptValue

__all__ = [
    "get_template_variables",
    "PromptTemplate",
]


def get_template_variables(template: str, template_format: str = "f-string") -> list[str]:
    """Return the base variable names referenced by a template, in order."""
    if template_format != "f-string":
        raise InputValidationError(f"Unsupported template format: {template_format!r}")

    variables: list[str] = []
    formatter = string.Formatter()
    try:
        for _, field_name, _, _ in formatter.parse(template):
            if field_name is None:
                continue
            base_name = field_name.split(".")[0].split("[")[0]
            if base_name and base_name not in variables:
                variables.append(base_name)
    except ValueError as err:
        raise InputValidationError(f"Invalid template string: {template!r}") from err
    return variables


class PromptTemplate(BaseModel):
    """A string template rendered with keyword arguments."""

    template: str
    """The template string containing ``{placeholders}``."""

    input_variables: list[str] = Field(default_factory=list)
    """The variables the template requires, inferred when omitted."""

    template_format: str = "f-string"
    """The template syntax used; only ``"f-string"`` is supported."""

    partial_variables: dict[str, Any] = Field(default_factory=dict)
    """Variables pre-bound via :meth:`partial`."""

    @model_validator(mode="after")
    def _infer_input_variables(self) -> Self:
        """Infer input variables from the template when none were given."""
        if not self.input_variables:
            self.input_variables = get_template_variables(self.template, self.template_format)
        return self

    @classmethod
    def from_template(
        cls,
        template: str,
        *,
        template_format: str = "f-string",
        partial_variables: dict[str, Any] | None = None,
    ) -> Self:
        """Build a template, inferring its input variables from the string."""
        return cls(
            template=template,
            template_format=template_format,
            input_variables=get_template_variables(template, template_format),
            partial_variables=partial_variables or {},
        )

    def format(self, **kwargs: Any) -> str:
        """Render the template, raising if a required variable is missing."""
        merged = {**self.partial_variables, **kwargs}
        missing = [var for var in self.input_variables if var not in merged]
        if missing:
            raise InputValidationError(f"Missing required input variables for prompt: {missing}")
        try:
            return self.template.format(**merged)
        except (KeyError, IndexError) as err:
            raise InputValidationError(f"Failed to format template: {self.template!r}") from err

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Render the template as a :class:`StringPromptValue`."""
        return StringPromptValue(text=self.format(**kwargs))

    def partial(self, **kwargs: Any) -> PromptTemplate:
        """Return a copy with some input variables pre-bound."""
        merged = {**self.partial_variables, **kwargs}
        remaining = [var for var in self.input_variables if var not in merged]
        return self.model_copy(update={"partial_variables": merged, "input_variables": remaining})
