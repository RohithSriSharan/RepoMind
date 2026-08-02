# 🧠 Cortex 

A Retrieval-Augmented Generation (RAG) system that lets you upload a document (PDF, Markdown, or text) and ask natural-language questions about it — with answers grounded strictly in the document's content, and page-level citations.

**🔗 Live demo:** [repomind.streamlit.app](https://repomind-ceecagubbgpx22w6x7c9xx.streamlit.app/)
**📦 Repo:** [github.com/RohithSriSharan/RepoMind](https://github.com/RohithSriSharan/RepoMind)

---

## Why this project

Most RAG demos stop at "it works." This one is built to demonstrate the full pipeline end-to-end, with the engineering practices around it: structured logging, automated tests, CI, and an honest accounting of what a free-tier deployment can and can't do.

The demo ships pre-loaded with *The 48 Laws of Power* so you can try it immediately — or upload your own PDF/Markdown/text file and ask questions about that instead.

---

## Architecture

```
Upload → Ingest → Clean → Chunk → Embed → Store (Chroma)
                                              │
User question → Embed → Retrieve top-k chunks ┘
                              │
                              ▼
                  Grounded prompt → LLM (Llama 3.3, via Groq)
                              │
                              ▼
                  Answer + page citations
```

**Pipeline stages:**
1. **Ingestion** — extract text from PDF (`pypdf`) or read Markdown/text directly; fix PDF-extraction structural noise (line breaks, whitespace)
2. **Chunking** — sentence-aware splitting into ~180-word overlapping chunks, so context isn't lost at chunk boundaries
3. **Embedding** — `all-MiniLM-L6-v2` (`sentence-transformers`) converts each chunk into a 384-dim vector — pretrained, no training happens in this pipeline
4. **Storage** — [Chroma](https://www.trychroma.com/) (local vector DB), storing vector + text + metadata (source, page number) as one linked record per chunk
5. **Retrieval** — cosine similarity search returns the top-k most relevant chunks for a question
6. **Generation** — retrieved chunks + question are assembled into a grounding prompt and sent to Llama 3.3 (via [Groq](https://groq.com/)), instructed to answer *only* from the provided context and cite sources — or say so explicitly if the context doesn't contain the answer

---

## Tech stack

| Layer | Tool |
|---|---|
| Language | Python 3.11 |
| PDF parsing | pypdf |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector DB | ChromaDB |
| LLM | Llama 3.3 70B, via Groq API |
| UI | Streamlit |
| Testing | pytest |
| CI | GitHub Actions |

---

## Project structure

```
repomind/
├── src/
│   ├── ingestion.py       # load & clean PDF/md/txt
│   ├── chunking.py        # sentence-aware overlapping chunking
│   ├── embeddings.py      # sentence-transformers wrapper
│   ├── vectorstore.py     # Chroma setup, insert, query
│   ├── generation.py      # prompt construction + LLM call
│   ├── pipeline.py        # ties every stage together
│   ├── analytics.py       # usage tracking (uploads/questions/sessions)
│   └── logging_config.py  # centralized logging
├── scripts/
│   └── build_index.py     # CLI to (re)build the index from a document
├── tests/
│   └── test_pipeline.py   # unit tests for ingestion/chunking logic
├── app.py                 # Streamlit UI
└── requirements.txt
```

---

## Running it locally

```bash
git clone https://github.com/RohithSriSharan/RepoMind.git
cd RepoMind

conda create -n repomind python=3.11 -y
conda activate repomind
pip install -r requirements.txt
```

Create a `.env` file with your [Groq API key](https://console.groq.com):
```
GROQ_API_KEY=your_key_here
```

Build the index from a document:
```bash
python -m scripts.build_index data/sample_docs/your_file.pdf
```

Run the app:
```bash
streamlit run app.py
```

Run tests:
```bash
python -m pytest tests/ -v
```

---

## Known limitations

- **PDF extraction noise:** pages with non-standard layouts (tables of contents, indexes) show more text-extraction noise than body text — a known `pypdf` limitation, not something this pipeline attempts to fully repair, since it doesn't meaningfully affect embedding-based retrieval.
- **No conversational memory:** each question is answered independently; there's no follow-up/context-carrying between questions in the current version.
- **Ephemeral storage on free-tier hosting:** documents uploaded through the live demo, and usage analytics, reset when the app restarts, since Streamlit Community Cloud's filesystem isn't persistent. The pre-loaded demo book is unaffected (it's committed to the repo). A production deployment would persist uploads and analytics to external storage (e.g., managed vector DB, cloud database).
- **Free-tier compute:** the live demo can be CPU-throttled under load on Streamlit Community Cloud's free tier.

## Possible extensions

- Conversational memory (multi-turn follow-up questions)
- Swap Chroma for a managed vector DB (Pinecone/Weaviate) for production-scale persistence
- Retrieval evaluation harness (measuring retrieval precision against a labeled question set)
- Reranking retrieved chunks before generation

---

## Author

Rohith Jangam — [GitHub](https://github.com/RohithSriSharan)