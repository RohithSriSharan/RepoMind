# """
# Cortex - AI-powered document Q&A using RAG.
# Upload a document, ask questions, get grounded answers with citations.
# """
# import streamlit as st
# import tempfile
# import os
# from src.pipeline import ingest_file, ask
# from src.logging_config import get_logger

# logger = get_logger(__name__)

# st.set_page_config(page_title="Cortex — Document Q&A", page_icon="🧠", layout="centered")

# # ================= Sidebar: About =================
# with st.sidebar:
#     st.markdown("## 🧠 About Cortex")
#     st.markdown(
#         "Cortex is a **Retrieval-Augmented Generation (RAG)** system — "
#         "upload a document, and it answers your questions using only "
#         "that document's content, with page-level citations."
#     )

#     st.markdown("### How it works")
#     st.markdown(
#         "- **Ingest** — parse & clean the document\n"
#         "- **Chunk** — split into overlapping passages\n"
#         "- **Embed** — convert chunks to vectors (`sentence-transformers`)\n"
#         "- **Store** — index in a vector DB (`Chroma`)\n"
#         "- **Retrieve** — find the most relevant chunks for your question\n"
#         "- **Generate** — an LLM (`Llama 3.3` via Groq) answers, grounded "
#         "strictly in the retrieved context"
#     )

#     st.markdown("### Tech stack")
#     st.markdown(
#         "`Python` · `Streamlit` · `sentence-transformers` · `ChromaDB` · "
#         "`Groq (Llama 3.3)` · `pytest`"
#     )

#     st.divider()
#     st.markdown(
#         "[![GitHub](https://img.shields.io/badge/GitHub-View_Source-181717?logo=github)]"
#         "(https://github.com/RohithSriSharan/RepoMind)"
#     )
#     st.caption("Built by Rohith Jangam")

# # ================= Custom Styling =================
# st.markdown("""
# <style>
#     /* Overall canvas */
#     .main {
#         padding-top: 0.5rem;
#     }
#     .block-container {
#         max-width: 780px;
#         padding-top: 2rem;
#     }

#     /* Header */
#     .cortex-header {
#         text-align: center;
#         padding: 1rem 0 0.3rem 0;
#     }
#     .cortex-title {
#         font-size: 2.8rem;
#         font-weight: 800;
#         letter-spacing: -0.02em;
#         background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         margin-bottom: 0.2rem;
#     }
#     .cortex-subtitle {
#         color: #9ca3af;
#         font-size: 1.05rem;
#         font-weight: 400;
#     }

#     /* Sidebar collapse arrow - label it "About" */
#     [data-testid="stSidebarCollapsedControl"] {
#         display: flex !important;
#         align-items: center !important;
#         gap: 0.3rem;
#     }
#     [data-testid="stSidebarCollapsedControl"]::before {
#         content: "About";
#         font-weight: 700;
#         font-size: 0.85rem;
#         color: #a855f7;
#         white-space: nowrap;
#     }

#     /* Section labels */
#         font-weight: 700;
#         font-size: 1.05rem;
#         margin-bottom: 0.6rem;
#         display: flex;
#         align-items: center;
#         gap: 0.5rem;
#     }

#     /* Answer box */
#     .answer-box {
#         background: linear-gradient(135deg, rgba(99,102,241,0.10), rgba(168,85,247,0.06));
#         border-left: 4px solid #8b5cf6;
#         border-radius: 12px;
#         padding: 1.1rem 1.3rem;
#         margin-top: 0.6rem;
#         line-height: 1.6;
#     }

#     /* Source chips */
#     .source-chip {
#         display: inline-block;
#         background-color: rgba(168, 85, 247, 0.12);
#         color: #c084fc;
#         padding: 0.25rem 0.8rem;
#         border-radius: 999px;
#         font-size: 0.82rem;
#         margin: 0.2rem 0.35rem 0.2rem 0;
#         border: 1px solid rgba(168, 85, 247, 0.25);
#     }

#     /* Buttons */
#     div.stButton > button {
#         border-radius: 10px;
#         font-weight: 600;
#         border: none;
#         background: linear-gradient(90deg, #6366f1, #a855f7);
#         color: white;
#         transition: opacity 0.15s ease;
#     }
#     div.stButton > button:hover {
#         opacity: 0.85;
#         color: white;
#     }

#     /* Text inputs */
#     div[data-baseweb="input"] > div {
#         border-radius: 10px;
#     }

#     /* File uploader */
#     section[data-testid="stFileUploaderDropzone"] {
#         border-radius: 14px;
#     }

#     /* Divider spacing tighter */
#     hr {
#         margin: 1.2rem 0;
#     }
# </style>
# """, unsafe_allow_html=True)

# # ================= Header =================
# st.markdown("""
# <div class="cortex-header">
#     <div class="cortex-title">🧠 Cortex</div>
#     <div class="cortex-subtitle">Ask anything about your documents — grounded, cited, no guessing.</div>
# </div>
# """, unsafe_allow_html=True)

# st.markdown("<br>", unsafe_allow_html=True)

# # ================= Upload Section =================
# with st.container(border=True):
#     st.markdown('<div class="section-label">📄 Upload a document</div>', unsafe_allow_html=True)

#     uploaded_file = st.file_uploader(
#         "Drop a PDF, Markdown, or text file",
#         type=["pdf", "md", "txt"],
#         label_visibility="collapsed"
#     )

#     if uploaded_file is not None:
#         col1, col2 = st.columns([3, 1])
#         with col1:
#             st.caption(f"Selected: **{uploaded_file.name}**")
#         with col2:
#             ingest_clicked = st.button("Ingest ⚡", use_container_width=True)

#         if ingest_clicked:
#             with st.spinner("Reading, chunking, and embedding your document..."):
#                 suffix = os.path.splitext(uploaded_file.name)[1]
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#                     tmp.write(uploaded_file.getvalue())
#                     tmp_path = tmp.name

#                 try:
#                     count = ingest_file(tmp_path)
#                     st.success(f"✅ Added **{count} chunks** from '{uploaded_file.name}' to the index.")
#                 except Exception as e:
#                     logger.error(f"Ingestion failed for {uploaded_file.name}: {e}")
#                     st.error(f"Something went wrong: {e}")
#                 finally:
#                     os.unlink(tmp_path)

# # ================= Ask Section =================
# with st.container(border=True):
#     st.markdown('<div class="section-label">💬 Ask a question</div>', unsafe_allow_html=True)

#     question = st.text_input(
#         "Your question",
#         placeholder="e.g. What does this document say about...",
#         label_visibility="collapsed"
#     )
#     ask_clicked = st.button("Ask Cortex 🔍", use_container_width=True)

#     if ask_clicked and question:
#         with st.spinner("Retrieving relevant passages and generating an answer..."):
#             try:
#                 result = ask(question)

#                 st.markdown('<div class="section-label" style="margin-top:1rem;">✨ Answer</div>', unsafe_allow_html=True)
#                 st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

#                 st.markdown("<br>", unsafe_allow_html=True)
#                 with st.expander("📚 Sources used"):
#                     for src in result["sources"]:
#                         st.markdown(
#                             f'<span class="source-chip">{src["source"]} · page {src["page_number"]}</span>',
#                             unsafe_allow_html=True
#                         )
#             except Exception as e:
#                 logger.error(f"Query failed: {e}")
#                 st.error(f"Something went wrong: {e}")

# # ================= Admin (cleanup only, not user-facing) =================
# with st.expander("🔧 Admin"):
#     source_to_delete = st.text_input("Exact source filename to delete")
#     if st.button("Delete this source") and source_to_delete:
#         from src.vectorstore import delete_by_source
#         delete_by_source(source_to_delete)
#         st.success(f"Deleted chunks for '{source_to_delete}'")


















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

# ================= Sidebar: About =================
with st.sidebar:
    st.markdown("## 🧠 About Cortex")
    st.markdown(
        "Cortex is a **Retrieval-Augmented Generation (RAG)** system — "
        "upload a document, and it answers your questions using only "
        "that document's content, with page-level citations."
    )

    st.markdown("### How it works")
    st.markdown(
        "- **Ingest** — parse & clean the document\n"
        "- **Chunk** — split into overlapping passages\n"
        "- **Embed** — convert chunks to vectors (`sentence-transformers`)\n"
        "- **Store** — index in a vector DB (`Chroma`)\n"
        "- **Retrieve** — find the most relevant chunks for your question\n"
        "- **Generate** — an LLM (`Llama 3.3` via Groq) answers, grounded "
        "strictly in the retrieved context"
    )

    st.markdown("### Tech stack")
    st.markdown(
        "`Python` · `Streamlit` · `sentence-transformers` · `ChromaDB` · "
        "`Groq (Llama 3.3)` · `pytest`"
    )

    st.divider()
    st.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-View_Source-181717?logo=github)]"
        "(https://github.com/RohithSriSharan/RepoMind)"
    )
    st.caption("Built by Rohith Jangam")

# ================= Custom Styling =================
st.markdown("""
<style>
    /* Overall canvas */
    .main {
        padding-top: 0.5rem;
    }
    .block-container {
        max-width: 780px;
        padding-top: 2rem;
    }

    /* Header */
    .cortex-header {
        text-align: center;
        padding: 1rem 0 0.3rem 0;
    }
    .cortex-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .cortex-subtitle {
        color: #9ca3af;
        font-size: 1.05rem;
        font-weight: 400;
    }

    /* Sidebar collapse arrow - label it "About" */
    [data-testid="stSidebarCollapsedControl"] {
        display: flex !important;
        align-items: center !important;
        gap: 0.3rem;
    }
    [data-testid="stSidebarCollapsedControl"]::before {
        content: "About";
        font-weight: 700;
        font-size: 0.85rem;
        color: #a855f7;
        white-space: nowrap;
    }

    /* Section labels */
    .section-label {
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Answer box */
    .answer-box {
        background: linear-gradient(135deg, rgba(99,102,241,0.10), rgba(168,85,247,0.06));
        border-left: 4px solid #8b5cf6;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-top: 0.6rem;
        line-height: 1.6;
    }

    /* Source chips */
    .source-chip {
        display: inline-block;
        background-color: rgba(168, 85, 247, 0.12);
        color: #c084fc;
        padding: 0.25rem 0.8rem;
        border-radius: 999px;
        font-size: 0.82rem;
        margin: 0.2rem 0.35rem 0.2rem 0;
        border: 1px solid rgba(168, 85, 247, 0.25);
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: none;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        color: white;
        transition: opacity 0.15s ease;
    }
    div.stButton > button:hover {
        opacity: 0.85;
        color: white;
    }

    /* Text inputs */
    div[data-baseweb="input"] > div {
        border-radius: 10px;
    }

    /* File uploader */
    section[data-testid="stFileUploaderDropzone"] {
        border-radius: 14px;
    }

    /* Divider spacing tighter */
    hr {
        margin: 1.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ================= Header =================
st.markdown("""
<div class="cortex-header">
    <div class="cortex-title">🧠 Cortex</div>
    <div class="cortex-subtitle">Ask anything about your documents — grounded, cited, no guessing.</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ================= Upload Section =================
with st.container(border=True):
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

# ================= Ask Section =================
with st.container(border=True):
    st.markdown('<div class="section-label">💬 Ask a question</div>', unsafe_allow_html=True)

    question = st.text_input(
        "Your question",
        placeholder="e.g. What does this document say about...",
        label_visibility="collapsed"
    )
    ask_clicked = st.button("Ask Cortex 🔍", use_container_width=True)

    if ask_clicked and question:
        with st.spinner("Retrieving relevant passages and generating an answer..."):
            try:
                result = ask(question)

                st.markdown('<div class="section-label" style="margin-top:1rem;">✨ Answer</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("📚 Sources used"):
                    for src in result.get("sources", []):
                        st.markdown(
                            f'<span class="source-chip">{src.get("source", "Unknown")} · page {src.get("page_number", "N/A")}</span>',
                            unsafe_allow_html=True
                        )
            except Exception as e:
                logger.error(f"Query failed: {e}")
                st.error(f"Something went wrong: {e}")

# ================= Admin =================
with st.expander("🔧 Admin"):
    source_to_delete = st.text_input("Exact source filename to delete")
    if st.button("Delete this source") and source_to_delete:
        from src.vectorstore import delete_by_source
        delete_by_source(source_to_delete)
        st.success(f"Deleted chunks for '{source_to_delete}'")