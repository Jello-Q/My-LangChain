# AGENTS.md

Guidance for agentic coding agents working in this repository.

## Repository status

This is an early-stage, from-scratch reimplementation of the LangChain framework
(`README.md`: "从零开始构建的LangChain框架"). As of the initial commit the repo
contains **no application source code, no packaging metadata, no tests, and no
lint/format tooling** — only `README.md`, `LICENSE`, and IDE files.

Implications for agents:
- There are currently **no build, lint, or test commands that actually exist.**
  Do not claim code "passes lint/tests" until the tooling below has been added.
- When you add the first module, you are also establishing the conventions.
  Prefer the standards described here and keep the project consistent.
- Do not commit `.idea/` (it is untracked; IDE-local state).

The intended standard toolchain is documented below. Treat it as the target
contract: if a command is missing because its config file has not been added
yet, add the config as part of the change that first needs it.

## Environment

- Python **3.13** (project SDK per `.idea/misc.xml`).
- Use a virtual environment (`.venv/`). Never install into the system Python.
- Prefer `pyproject.toml` (PEP 621) for metadata and tool config; do not add a
  `setup.py` unless a legacy need is proven.

## Commands

Once `pyproject.toml` exists these are the canonical commands.

Setup:
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (this repo's platform)
pip install -e ".[dev]"
```

Lint / format / type-check:
```bash
ruff check .                  # lint
ruff check --fix .            # autofix
ruff format .                 # format (black-compatible)
mypy my_langchain             # static types
```

Test:
```bash
pytest                        # full suite
pytest -q                     # quiet
pytest tests/test_chains.py::test_simple_chain   # single test by node id
pytest -k "chain and not slow"                   # single test by keyword
pytest --cov=my_langchain --cov-report=term-missing  # coverage
```

If a command above does not work, check for the config file it depends on
(`pyproject.toml`, `[tool.ruff]`, `[tool.pytest.ini_options]`, `[tool.mypy]`)
before assuming a different invocation is required. Do not invent alternatives
without adding the corresponding config.

## Project layout

Target layout (create directories only when first needed):

```
my_langchain/          # importable package (snake_case)
  __init__.py
  core/                # base abstractions (Runnable, Prompt, Message)
  llms/                # model wrappers / providers
  chains/              # composition over runnables
  memory/              # conversation state
  prompts/             # templates and formatting
tests/                 # mirrors package structure, test_*.py
```

Keep modules small and single-purpose. Mirror source paths in `tests/`.

## Code style

- Follow **PEP 8**. Line length **100** (configure it in ruff).
- Formatting is non-negotiable: `ruff format` output is the source of truth.
  Do not hand-format or align code manually.
- Prefer `ruff` for both linting and formatting; do not add flake8/black/isort
  alongside it.

### Imports

- Group in this order, each separated by one blank line:
  1. standard library
  2. third-party
  3. first-party (`my_langchain`)
  4. local/relative
- Use absolute imports for first-party code
  (`from my_langchain.core.runnable import Runnable`).
- Avoid wildcard imports and avoid import-time side effects (no network calls,
  no reading API keys at import time).

### Types

- Annotate all public functions, methods, and class attributes. Use
  `from __future__ import annotations` at the top of modules for cheap forward
  references.
- Use built-in generics (`list[str]`, `dict[str, Any]`, `X | None`), not
  `typing.List`/`Optional`, on Python 3.13.
- Prefer `Protocol` or `ABC` for extension points (e.g. `Runnable`, `BaseLLM`)
  so third-party providers can implement them without inheriting heavy bases.
- Run `mypy` cleanly; avoid `# type: ignore` unless narrowly scoped with a
  comment explaining why.

### Naming

- `snake_case` for functions, methods, variables, modules.
- `PascalCase` for classes.
- `UPPER_SNAKE_CASE` for module-level constants.
- Prefix intentional non-public APIs with a single underscore; keep `__all__`
  in package `__init__.py` files curated.
- Test functions: `test_<behavior>`; one behavior per test.

### Docstrings and comments

- Public classes and functions get a concise docstring (imperative summary
  line, then params/returns only when not obvious). NumPy or Google style,
  used consistently within a module.
- Comment the "why", not the "what". Do not add narrating comments.
- Write user-facing docs and README content in Chinese when matching the
  existing `README.md`; keep code, identifiers, and docstrings in English.

### Error handling

- Raise specific exceptions; define domain errors under
  `my_langchain/core/exceptions.py` (e.g. `MyLangChainError` as the base,
  subclasses for provider/parsing failures).
- Never use bare `except:` or swallow exceptions silently. Catch the narrowest
  type; re-raise with context using `raise ... from err` when wrapping.
- Validate inputs at public boundaries and fail fast with a clear message.
- Do not log secrets, API keys, tokens, or full prompts containing user data.
  Read credentials from environment variables; never hardcode them.

### Async

- Provide async variants where the underlying provider is async
  (`ainvoke`/`astream`). Keep sync and async paths behaviorally identical and
  test both.

## Testing

- Framework: `pytest` (plus `pytest-asyncio` for async code).
- Put tests in `tests/`, mirroring the package path.
- Prefer fast, deterministic unit tests; mock providers/network. Mark any
  test needing live APIs with `@pytest.mark.integration` and keep them opt-in.
- No hidden test command: run the full suite with `pytest` and a single test
  with its node id (see Commands).

## Git

- Do not create commits unless explicitly asked.
- Keep commits focused; match the existing message style (currently short
  descriptive subjects, e.g. `Initial commit`).
- Never commit `.env`, credentials, `.venv/`, or `.idea/`.

## Agent workflow

1. Read this file and the surrounding files before editing.
2. Make the smallest change that satisfies the request; match existing patterns.
3. Run `ruff format`, `ruff check`, `mypy`, and `pytest` before claiming success,
   once the tooling exists. Report the actual command output — do not assume.
4. If a required tool/config is missing, add it rather than silently skipping
   verification, and note what you added.
