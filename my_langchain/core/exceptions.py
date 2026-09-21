"""Domain exception hierarchy for My-LangChain.

Every framework error derives from :class:`MyLangChainError`, so callers can
catch framework-specific failures without matching broad builtin exceptions.
Specific subclasses let callers react to *why* something failed (bad config,
bad input, provider failure, unparseable output) instead of *where* it happened.
"""

from __future__ import annotations

__all__ = [
    "MyLangChainError",
    "ConfigurationError",
    "InputValidationError",
    "ProviderError",
    "ParsingError",
]


class MyLangChainError(Exception):
    """Base class for all errors raised by My-LangChain."""


class ConfigurationError(MyLangChainError):
    """Raised when the framework is misconfigured, e.g. missing credentials."""


class InputValidationError(MyLangChainError):
    """Raised when user input fails validation at a public boundary."""


class ProviderError(MyLangChainError):
    """Raised when a model or embedding provider call fails."""


class ParsingError(MyLangChainError):
    """Raised when model output cannot be parsed into the expected shape."""
