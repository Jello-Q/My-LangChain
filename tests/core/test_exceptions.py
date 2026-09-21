"""Tests for the My-LangChain exception hierarchy."""

import pytest

from my_langchain.core.exceptions import (
    ConfigurationError,
    InputValidationError,
    MyLangChainError,
    ParsingError,
    ProviderError,
)


def test_base_error_subclasses_exception() -> None:
    assert issubclass(MyLangChainError, Exception)


@pytest.mark.parametrize(
    "exc_type",
    [ConfigurationError, InputValidationError, ProviderError, ParsingError],
)
def test_domain_errors_subclass_base(exc_type: type[MyLangChainError]) -> None:
    assert issubclass(exc_type, MyLangChainError)


def test_error_can_be_raised_and_caught_by_base() -> None:
    with pytest.raises(MyLangChainError, match="boom"):
        raise ParsingError("boom")
