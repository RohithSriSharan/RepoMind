"""
Embeddings: convert chunk text into 384-dim vectors using a pretrained
sentence-transformer model. No training happens here — pure inference.
"""
from sentence_transformers import SentenceTransformer
from src.logging_config import get_logger

logger = get_logger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None  # loaded lazily, once, on first use


def get_model() -> SentenceTransformer:
    """
    Load the embedding model once and reuse it (loading takes a few seconds -
    we don't want to reload it on every single function call).
    """
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {_MODEL_NAME}")
        _model = SentenceTransformer(_MODEL_NAME)
        logger.info(f"Model loaded. Embedding dimension: {_model.get_sentence_embedding_dimension()}")
    return _model


def embed_chunks(chunks: list[dict], batch_size: int = 32) -> list[list[float]]:
    """
    Embed a list of chunk dicts (from chunking.py). Returns a list of
    vectors, in the SAME order as the input chunks.
    """
    model = get_model()
    texts = [c["text"] for c in chunks]

    logger.info(f"Embedding {len(texts)} chunks...")
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    logger.info(f"Embedding complete. Shape: {vectors.shape}")

    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single question/query string - same model, same vector space."""
    model = get_model()
    return model.encode(text).tolist()


if __name__ == "__main__":
    from src.ingestion import load_document
    from src.chunking import chunk_documents
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_docs/48LawsOfPower.pdf"
    docs = load_document(path)
    chunks = chunk_documents(docs)
    vectors = embed_chunks(chunks[:5])  # just first 5, for a quick test

    print(f"Embedded {len(vectors)} chunks")
    print(f"Vector length: {len(vectors[0])}")