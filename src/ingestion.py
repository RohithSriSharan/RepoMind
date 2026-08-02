"""
Ingestion: load and clean documents (PDF, Markdown, text).
"""
import re
from pathlib import Path
from pypdf import PdfReader

from src.logging_config import get_logger

logger = get_logger(__name__)


def clean_text(text: str) -> str:
    """Fix PDF-extraction structural noise: line breaks -> spaces, collapse whitespace."""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_pdf(filepath: str) -> list[dict]:
    """Extract text from a PDF, one record per page. Skips blank/image-only pages."""
    reader = PdfReader(filepath)
    filename = Path(filepath).name
    records = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = clean_text(text)
        if text:
            records.append({
                "text": text,
                "source": filename,
                "page_number": i + 1,
            })

    logger.info(f"Loaded PDF '{filename}': {len(records)}/{len(reader.pages)} pages had extractable text.")
    return records


def load_text_like(filepath: str) -> list[dict]:
    """Load .md / .txt files as a single record."""
    filename = Path(filepath).name
    content = Path(filepath).read_text(encoding="utf-8", errors="ignore")
    content = clean_text(content)
    logger.info(f"Loaded text file '{filename}': {len(content.split())} words.")
    return [{"text": content, "source": filename, "page_number": None}]


def load_document(filepath: str) -> list[dict]:
    """Dispatch to the right loader based on file extension."""
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        return load_pdf(filepath)
    elif ext in (".md", ".txt"):
        return load_text_like(filepath)
    else:
        logger.error(f"Unsupported file type: {ext}")
        raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .md, .txt")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        docs = load_document(sys.argv[1])
        print(f"Loaded {len(docs)} record(s)")
        if docs:
            print(docs[0]["text"][:200])
    else:
        print("Usage: python -m src.ingestion <filepath>")