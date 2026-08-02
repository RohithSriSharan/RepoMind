"""
Generation: build a grounded prompt from retrieved chunks and call the LLM.
"""
import os
from dotenv import load_dotenv
from groq import Groq
from src.logging_config import get_logger
import streamlit

load_dotenv()  # reads .env file, loads GROQ_API_KEY into environment

logger = get_logger(__name__)

_MODEL_NAME = "llama-3.3-70b-versatile"
_client = None


def get_client() -> Groq:
    """Lazily create the Groq client - same lazy pattern as embeddings/vectorstore."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.error("GROQ_API_KEY not found in environment. Check your .env file.")
            raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")
        _client = Groq(api_key=api_key)
        logger.info("Groq client initialized.")
    return _client


def build_prompt(question: str, retrieved_chunks: list[str], retrieved_metas: list[dict]) -> str:
    """
    Construct a grounded prompt: instructs the model to answer ONLY from
    context, cite sources, and say so explicitly if context is insufficient.
    """
    context_lines = []
    for chunk, meta in zip(retrieved_chunks, retrieved_metas):
        page = meta.get("page_number", "unknown")
        source = meta.get("source", "unknown")
        context_lines.append(f"[Source: {source}, Page {page}]: {chunk}")

    context_block = "\n\n".join(context_lines)

    prompt = f"""Answer the question using ONLY the context below.
If the context does not contain enough information to answer, say so clearly - do not use outside knowledge.
Cite the source and page number(s) your answer is based on.

Context:
{context_block}

Question: {question}

Answer:"""
    return prompt


def generate_answer(question: str, retrieved_chunks: list[str], retrieved_metas: list[dict]) -> str:
    """Send the grounded prompt to the LLM and return its answer."""
    client = get_client()
    prompt = build_prompt(question, retrieved_chunks, retrieved_metas)

    logger.info(f"Sending prompt to LLM for question: '{question[:60]}...'")
    response = client.chat.completions.create(
        model=_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    logger.info("LLM response received.")
    return answer

def get_client() -> Groq:
    """Lazily create the Groq client - works both locally (.env) and on Streamlit Cloud (st.secrets)."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets["GROQ_API_KEY"]
            except (ImportError, KeyError, FileNotFoundError):
                pass

        if not api_key:
            logger.error("GROQ_API_KEY not found in environment or Streamlit secrets.")
            raise ValueError("GROQ_API_KEY not set. Add it to your .env file or Streamlit secrets.")

        _client = Groq(api_key=api_key)
        logger.info("Groq client initialized.")
    return _client


if __name__ == "__main__":
    # quick isolated test - fake retrieved chunks, no need to touch Chroma here
    fake_chunks = ["Surrender gives you time to recover and wait for your enemy's power to wane."]
    fake_metas = [{"source": "48LawsOfPower.pdf", "page_number": 163}]

    answer = generate_answer("What does the book say about surrender?", fake_chunks, fake_metas)
    print(answer)