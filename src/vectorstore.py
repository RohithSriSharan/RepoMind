"""
Vector store: Chroma setup, insertion, and similarity search.
Bundles vector + text + metadata together per chunk - no fragile
index-position matching required.
"""
import chromadb
from src.logging_config import get_logger

logger = get_logger(__name__)

_COLLECTION_NAME = "repomind_docs"
_DB_PATH = "./chroma_db"

_client = None
_collection = None


def get_collection():
    """Lazily create/connect to the Chroma collection - same lazy pattern as the embedding model."""
    global _client, _collection
    if _collection is None:
        logger.info(f"Connecting to Chroma at '{_DB_PATH}'")
        _client = chromadb.PersistentClient(path=_DB_PATH)
        _collection = _client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Collection '{_COLLECTION_NAME}' ready. Existing items: {_collection.count()}")
    return _collection


def add_chunks(chunks: list[dict], vectors: list[list[float]]) -> None:
    """
    Insert chunks + their vectors into Chroma, bundled with metadata.
    chunks and vectors must be in the SAME order (same contract as our
    notebook version) - this is the last place that ordering matters,
    since after this, everything is linked by chunk_id inside Chroma.
    """
    collection = get_collection()

    collection.add(
        embeddings=vectors,
        documents=[c["text"] for c in chunks],
        metadatas=[
            {
                "source": c["source"],
                "page_number": c.get("page_number") or 0,  # Chroma metadata can't store None
                "chunk_index": c["chunk_index"],
            }
            for c in chunks
        ],
        ids=[c["chunk_id"] for c in chunks],
    )
    logger.info(f"Inserted {len(chunks)} chunks. Collection now has {collection.count()} items.")


def query_chunks(query_vector: list[float], n_results: int = 3) -> dict:
    """Search for the top-n most similar stored chunks to the given query vector."""
    collection = get_collection()
    results = collection.query(query_embeddings=[query_vector], n_results=n_results)
    logger.info(f"Query returned {len(results['documents'][0])} result(s).")
    return results

def delete_by_source(source_filename: str) -> None:
    """Remove all chunks belonging to a specific source file from the collection."""
    collection = get_collection()
    collection.delete(where={"source": source_filename})
    logger.info(f"Deleted all chunks with source='{source_filename}'. Collection now has {collection.count()} items.")

    
if __name__ == "__main__":
    from src.ingestion import load_document
    from src.chunking import chunk_documents
    from src.embeddings import embed_chunks, embed_query
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_docs/48LawsOfPower.pdf"
    docs = load_document(path)
    chunks = chunk_documents(docs)[:20]  # just 20, for a quick test
    vectors = embed_chunks(chunks)

    add_chunks(chunks, vectors)

    test_query = "What does the book say about enemies?"
    query_vec = embed_query(test_query)
    results = query_chunks(query_vec, n_results=2)

    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"--- page {meta['page_number']} ---")
        print(doc[:150], "...\n")