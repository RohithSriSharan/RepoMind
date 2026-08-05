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
Cortex — AI-powered document Q&A using RAG.
Upload a document, ask questions, get grounded answers with citations.
"""
import streamlit as st
import tempfile
import os
from src.pipeline import ingest_file, ask
from src.logging_config import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title="Cortex — Document Q&A", 
    page_icon="🧠", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# ================= Load Icons & Custom Styling =================
st.markdown("""
<!-- Load Font Awesome 6 -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">

<style>
    /* Overall canvas styling */
    .main {
        background-color: #0f172a;
    }
    .block-container {
        max-width: 820px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header styling */
    .cortex-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .cortex-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.75rem;
    }
    .cortex-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 400;
    }

    /* Modern Card Layouts */
    .card-container {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }

    /* Section Labels with FontAwesome Icons */
    .section-label {
        font-weight: 700;
        font-size: 1.1rem;
        color: #f1f5f9;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .section-label i {
        color: #c084fc;
        font-size: 1.2rem;
    }

    /* Answer box styling */
    .answer-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(168, 85, 247, 0.08));
        border-left: 4px solid #a855f7;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-top: 0.8rem;
        line-height: 1.7;
        color: #e2e8f0;
        font-size: 1rem;
    }

    /* Modern Pill Badges / Source Chips */
    .source-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background-color: rgba(168, 85, 247, 0.15);
        color: #e9d5ff;
        padding: 0.3rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.25rem 0.35rem 0.25rem 0;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }

    /* Tech Stack Badges for Sidebar */
    .tech-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(255, 255, 255, 0.05);
        color: #cbd5e1;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.15rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Button Enhancements */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: none;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        color: white;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3);
    }
    div.stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(124, 58, 237, 0.4);
        color: white;
    }

    /* Input focus & borders */
    div[data-baseweb="input"] > div {
        border-radius: 10px;
        background-color: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.12);
    }

    /* File Uploader area */
    section[data-testid="stFileUploaderDropzone"] {
        border-radius: 14px;
        background-color: rgba(15, 23, 42, 0.4);
        border: 1.5px dashed rgba(168, 85, 247, 0.4);
    }

    hr {
        border-color: rgba(255, 255, 255, 0.1);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ================= Sidebar: About =================
with st.sidebar:
    st.markdown("## <i class='fa-solid fa-brain' style='color:#a855f7;'></i> Cortex RAG", unsafe_allow_html=True)
    st.markdown(
        "Cortex is a **Retrieval-Augmented Generation (RAG)** system — "
        "upload a document, and it answers questions strictly grounded in "
        "that content with exact source citations."
    )

    st.markdown("---")
    st.markdown("### <i class='fa-solid fa-gears' style='color:#818cf8;'></i> How it works", unsafe_allow_html=True)
    st.markdown("""
    * <i class='fa-solid fa-file-arrow-up' style='color:#a855f7;'></i> **Ingest** — Parse & clean documents
    * <i class='fa-solid fa-scissors' style='color:#a855f7;'></i> **Chunk** — Split into context passages
    * <i class='fa-solid fa-network-wired' style='color:#a855f7;'></i> **Embed** — `sentence-transformers`
    * <i class='fa-solid fa-database' style='color:#a855f7;'></i> **Store** — Vector DB (`ChromaDB`)
    * <i class='fa-solid fa-magnifying-glass' style='color:#a855f7;'></i> **Retrieve** — Find key context
    * <i class='fa-solid fa-robot' style='color:#a855f7;'></i> **Generate** — LLM (`Llama 3.3` via Groq)
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### <i class='fa-solid fa-layer-group' style='color:#f472b6;'></i> Tech Stack", unsafe_allow_html=True)
    st.markdown("""
    <span class="tech-badge"><i class="fa-brands fa-python"></i> Python</span>
    <span class="tech-badge"><i class="fa-solid fa-bolt"></i> Streamlit</span>
    <span class="tech-badge"><i class="fa-solid fa-cube"></i> ChromaDB</span>
    <span class="tech-badge"><i class="fa-solid fa-microchip"></i> Groq</span>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown(
        '<a href="https://github.com/RohithSriSharan/RepoMind" target="_blank" style="text-decoration:none;">'
        '<div style="background:rgba(255,255,255,0.08); padding:0.6rem 1rem; border-radius:8px; text-align:center; color:white; font-weight:600;">'
        '<i class="fa-brands fa-github"></i> View Source Code'
        '</div></a>',
        unsafe_allow_html=True
    )
    st.caption("Built by Rohith Jangam")

# ================= Header =================
st.markdown("""
<div class="cortex-header">
    <div class="cortex-title">
        <i class="fa-solid fa-brain"></i> Cortex
    </div>
    <div class="cortex-subtitle">Grounded document intelligence — precise answers with zero hallucination.</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ================= Upload Section =================
with st.container():
    st.markdown('<div class="section-label"><i class="fa-solid fa-cloud-arrow-up"></i> Upload Document</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop a PDF, Markdown, or text file",
        type=["pdf", "md", "txt"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"📁 Selected: **{uploaded_file.name}**")
        with col2:
            ingest_clicked = st.button("Ingest ⚡", use_container_width=True)

        if ingest_clicked:
            with st.spinner("Reading, chunking, and embedding document..."):
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    count = ingest_file(tmp_path)
                    st.success(f"✅ Indexed **{count} chunks** from `{uploaded_file.name}`.")
                except Exception as e:
                    logger.error(f"Ingestion failed for {uploaded_file.name}: {e}")
                    st.error(f"Something went wrong: {e}")
                finally:
                    os.unlink(tmp_path)

# ================= Ask Section =================
with st.container():
    st.markdown('<div class="section-label"><i class="fa-solid fa-comments"></i> Ask Questions</div>', unsafe_allow_html=True)

    question = st.text_input(
        "Your question",
        placeholder="e.g. What are the key findings or takeaways in this document?",
        label_visibility="collapsed"
    )
    ask_clicked = st.button("Ask Cortex 🔍", use_container_width=True)

    if ask_clicked and question:
        with st.spinner("Searching vectors & synthesizing response..."):
            try:
                result = ask(question)

                st.markdown('<div class="section-label" style="margin-top:1.5rem;"><i class="fa-solid fa-sparkles" style="color:#f472b6;"></i> Answer</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("📚 Sources & References"):
                    for src in result.get("sources", []):
                        st.markdown(
                            f'<span class="source-chip"><i class="fa-solid fa-file-lines"></i> {src.get("source", "Unknown")} · Page {src.get("page_number", "N/A")}</span>',
                            unsafe_allow_html=True
                        )
            except Exception as e:
                logger.error(f"Query failed: {e}")
                st.error(f"Something went wrong: {e}")

# ================= Admin Section =================
with st.expander("🔧 System Admin"):
    source_to_delete = st.text_input("Exact source filename to delete")
    if st.button("Delete this source") and source_to_delete:
        from src.vectorstore import delete_by_source
        delete_by_source(source_to_delete)
        st.success(f"Deleted chunks for `{source_to_delete}`")