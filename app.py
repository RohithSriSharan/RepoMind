"""
Cortex - AI-powered document Q&A using RAG.
Upload a document, ask questions, get grounded answers with citations.
"""
import streamlit as st
import tempfile
import os
from src.pipeline import ingest_file, ask
from src.logging_config import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="Cortex — Document Q&A", page_icon="🧠", layout="centered")

# ---------- Custom styling ----------
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .cortex-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .cortex-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .cortex-subtitle {
        color: #9ca3af;
        font-size: 1rem;
        margin-top: -0.3rem;
    }
    .section-label {
        font-weight: 600;
        font-size: 1.05rem;
        margin-top: 1.2rem;
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .answer-box {
        background-color: rgba(99, 102, 241, 0.08);
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-top: 0.5rem;
    }
    .source-chip {
        display: inline-block;
        background-color: rgba(168, 85, 247, 0.12);
        color: #a855f7;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.82rem;
        margin: 0.2rem 0.3rem 0.2rem 0;
    }
    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="cortex-header">
    <div class="cortex-title">🧠 Cortex</div>
    <div class="cortex-subtitle">Ask anything about your documents — grounded, cited, no guessing.</div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------- Upload section ----------
st.markdown('<div class="section-label">📄 Upload a document</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drop a PDF, Markdown, or text file",
    type=["pdf", "md", "txt"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"Selected: **{uploaded_file.name}**")
    with col2:
        ingest_clicked = st.button("Ingest ⚡", use_container_width=True)

    if ingest_clicked:
        with st.spinner("Reading, chunking, and embedding your document..."):
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                count = ingest_file(tmp_path)
                st.success(f"✅ Added **{count} chunks** from '{uploaded_file.name}' to the index.")
            except Exception as e:
                logger.error(f"Ingestion failed for {uploaded_file.name}: {e}")
                st.error(f"Something went wrong: {e}")
            finally:
                os.unlink(tmp_path)

st.divider()

# ---------- Ask section ----------
st.markdown('<div class="section-label">💬 Ask a question</div>', unsafe_allow_html=True)
question = st.text_input("Your question", placeholder="e.g. What does this document say about...", label_visibility="collapsed")
ask_clicked = st.button("Ask Cortex 🔍", use_container_width=True)

if ask_clicked and question:
    with st.spinner("Retrieving relevant passages and generating an answer..."):
        try:
            result = ask(question)

            st.markdown('<div class="section-label">✨ Answer</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("📚 Sources used"):
                for src in result["sources"]:
                    st.markdown(
                        f'<span class="source-chip">{src["source"]} · page {src["page_number"]}</span>',
                        unsafe_allow_html=True
                    )
        except Exception as e:
            logger.error(f"Query failed: {e}")
            st.error(f"Something went wrong: {e}")

# ---------- Admin (hidden by default, for cleanup only) ----------
with st.expander("🔧 Admin"):
    source_to_delete = st.text_input("Exact source filename to delete")
    if st.button("Delete this source") and source_to_delete:
        from src.vectorstore import delete_by_source
        delete_by_source(source_to_delete)
        st.success(f"Deleted chunks for '{source_to_delete}'")