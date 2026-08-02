
"""
CLI script to build/rebuild the Chroma index from a document.
Usage: python -m scripts.build_index data/sample_docs/48LawsOfPower.pdf
"""
import sys
import argparse
from src.pipeline import ingest_file
from src.logging_config import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build the RepoMind vector index from a document.")
    parser.add_argument("filepath", type=str, help="Path to a PDF, .md, or .txt file to ingest.")
    args = parser.parse_args()

    print(f"Building index from: {args.filepath}")
    try:
        count = ingest_file(args.filepath)
        print(f"Success: {count} chunks added to the index.")
    except Exception as e:
        logger.error(f"Failed to build index: {e}")
        print(f"Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()