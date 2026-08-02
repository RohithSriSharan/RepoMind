"""
Basic tests for the RepoMind pipeline. Run with: pytest tests/
These are lightweight sanity tests, not exhaustive coverage - they check
that each stage produces output in the shape we expect, which is exactly
the kind of thing GitHub Actions will run automatically on every push.
"""
import pytest
from src.chunking import split_sentences, group_into_chunks, chunk_documents
from src.ingestion import clean_text


def test_clean_text_removes_linebreaks():
    dirty = "This is\na sentence\nsplit across lines.   Extra   spaces."
    cleaned = clean_text(dirty)
    assert "\n" not in cleaned
    assert "  " not in cleaned  # no double spaces left


def test_split_sentences_basic():
    text = "This is one sentence. This is another! Is this a third?"
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "This is one sentence."


def test_split_sentences_empty_string():
    assert split_sentences("") == []


def test_group_into_chunks_respects_size_limit():
    sentences = ["word " * 50 + "." for _ in range(10)]  # 10 sentences, ~50 words each
    chunks = group_into_chunks(sentences, chunk_size_words=100, overlap_words=10)
    for chunk in chunks:
        word_count = len(chunk.split())
        # allow some slack since we only check AFTER adding a sentence that pushed it over
        assert word_count <= 160


def test_chunk_documents_produces_traceable_metadata():
    records = [{"text": "This is page one. It has two sentences.", "source": "test.pdf", "page_number": 1}]
    chunks = chunk_documents(records)

    assert len(chunks) >= 1
    assert chunks[0]["source"] == "test.pdf"
    assert chunks[0]["page_number"] == 1
    assert chunks[0]["chunk_id"] == "test.pdf_page1_chunk0"


def test_chunk_documents_empty_input():
    assert chunk_documents([]) == []