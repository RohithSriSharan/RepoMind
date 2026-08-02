"""
Chunking: split cleaned page text into overlapping, retrieval-sized chunks.
"""
import re
from src.logging_config import get_logger

logger = get_logger(__name__)


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def group_into_chunks(sentences: list[str], chunk_size_words: int = 180, overlap_words: int = 30) -> list[str]:
    chunks = []
    current_words = []

    for sentence in sentences:
        sentence_words = sentence.split()
        if len(current_words) + len(sentence_words) > chunk_size_words and current_words:
            chunks.append(" ".join(current_words))
            current_words = current_words[-overlap_words:]
        current_words.extend(sentence_words)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def chunk_documents(records: list[dict], chunk_size_words: int = 180, overlap_words: int = 30) -> list[dict]:
    """
    Takes loaded page records (from ingestion.py) and produces a flat list
    of chunk dicts, each with text + traceability metadata (source, page,
    chunk_index, chunk_id).
    """
    all_chunks = []

    for record in records:
        sentences = split_sentences(record["text"])
        page_chunks = group_into_chunks(sentences, chunk_size_words, overlap_words)

        for idx, chunk_text in enumerate(page_chunks):
            all_chunks.append({
                "text": chunk_text,
                "source": record["source"],
                "page_number": record.get("page_number"),
                "chunk_index": idx,
                "chunk_id": f"{record['source']}_page{record.get('page_number')}_chunk{idx}",
            })

    logger.info(f"Chunked {len(records)} record(s) into {len(all_chunks)} chunks.")
    return all_chunks


if __name__ == "__main__":
    from src.ingestion import load_document
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_docs/48LawsOfPower.pdf"
    docs = load_document(path)
    chunks = chunk_documents(docs)

    print(f"{len(docs)} record(s) -> {len(chunks)} chunk(s)")
    print(f"\nExample chunk:\n{chunks[20]}")