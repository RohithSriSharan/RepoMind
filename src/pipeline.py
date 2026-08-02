"""
Pipeline: ties every stage together into two simple, high-level functions
that app.py (and scripts/build_index.py) will actually call.
"""
from src.ingestion import load_document
from src.chunking import chunk_documents
from src.embeddings import embed_chunks, embed_query
from src.vectorstore import add_chunks, query_chunks
from src.generation import generate_answer
from src.logging_config import get_logger

logger = get_logger(__name__)


def ingest_file(filepath: str) -> int:
    """
    Full ingestion pipeline: load -> chunk -> embed -> store.
    Returns the number of chunks added, so the caller (e.g. the app)
    can show the user confirmation ("added 342 chunks").
    """
    logger.info(f"Starting ingestion pipeline for: {filepath}")

    records = load_document(filepath)
    chunks = chunk_documents(records)
    vectors = embed_chunks(chunks)
    add_chunks(chunks, vectors)

    logger.info(f"Ingestion pipeline complete for: {filepath} ({len(chunks)} chunks)")
    return len(chunks)


def ask(question: str, n_results: int = 3) -> dict:
    """
    Full query pipeline: embed question -> retrieve -> generate.
    Returns a dict with the answer AND the raw sources, so the app can
    show citations/sources separately from the answer text if desired.
    """
    logger.info(f"Question received: '{question[:60]}...'")

    query_vector = embed_query(question)
    results = query_chunks(query_vector, n_results=n_results)

    retrieved_chunks = results["documents"][0]
    retrieved_metas = results["metadatas"][0]

    answer = generate_answer(question, retrieved_chunks, retrieved_metas)

    return {
        "answer": answer,
        "sources": retrieved_metas,
        "retrieved_chunks": retrieved_chunks,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        count = ingest_file(filepath)
        print(f"Ingested {count} chunks from {filepath}")
    else:
        # quick end-to-end test assuming something's already in Chroma
        result = ask("What does the book say about enemies?")
        print(result["answer"])
        print("\nSources:", result["sources"])