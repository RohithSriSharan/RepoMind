"""
RepoMind - Streamlit app.
Upload a document, ask questions, get grounded answers with citations.
"""
import streamlit as st
import tempfile
import os
from src.pipeline import ingest_file, ask
from src.logging_config import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="RepoMind", page_icon="📚", layout="centered")

st.title("📚 RepoMind")
st.caption("Upload a document (PDF, .md, .txt) and ask questions - answers are grounded in the document, with citations.")

# --- Upload section ---
st.subheader("1. Upload a document")
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "md", "txt"])

if uploaded_file is not None:
    if st.button("Ingest this document"):
        with st.spinner("Reading, chunking, and embedding your document... this can take a minute."):
            # Streamlit gives us an in-memory file - write it to a temp path
            # on disk first, since our pipeline expects a real filepath.
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                count = ingest_file(tmp_path)
                st.success(f"Done! Added {count} chunks from '{uploaded_file.name}' to the index.")
            except Exception as e:
                logger.error(f"Ingestion failed for {uploaded_file.name}: {e}")
                st.error(f"Something went wrong: {e}")
            finally:
                os.unlink(tmp_path)  # clean up the temp file either way

st.divider()

# --- Ask section ---
st.subheader("2. Ask a question")
question = st.text_input("Your question")

if st.button("Ask") and question:
    with st.spinner("Retrieving relevant passages and generating an answer..."):
        try:
            result = ask(question)
            st.markdown("### Answer")
            st.write(result["answer"])

            with st.expander("Sources used"):
                for src in result["sources"]:
                    st.write(f"- **{src['source']}**, page {src['page_number']}")
        except Exception as e:
            logger.error(f"Query failed: {e}")
            st.error(f"Something went wrong: {e}")