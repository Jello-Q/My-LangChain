"""Tests for streaming message chunks."""

from my_langchain.messages import AIMessageChunk, HumanMessageChunk


def test_chunk_addition_concatenates_content() -> None:
    chunk = AIMessageChunk(content="Hel") + AIMessageChunk(content="lo")
    assert chunk.content == "Hello"


def test_chunk_addition_preserves_concrete_type() -> None:
    chunk = HumanMessageChunk(content="a") + HumanMessageChunk(content="b")
    assert isinstance(chunk, HumanMessageChunk)
    assert chunk.type == "HumanMessageChunk"


def test_chunk_addition_merges_additional_kwargs() -> None:
    first = AIMessageChunk(content="a", additional_kwargs={"x": 1})
    second = AIMessageChunk(content="b", additional_kwargs={"y": 2})
    merged = first + second
    assert merged.additional_kwargs == {"x": 1, "y": 2}


def test_chunk_addition_keeps_first_non_empty_id() -> None:
    first = AIMessageChunk(content="a", id="first")
    second = AIMessageChunk(content="b", id="second")
    assert (first + second).id == "first"
    assert (AIMessageChunk(content="a") + second).id == "second"
