# 📚 Technical-Document RAG Chatbot

> Query technical PDFs with an LLM — and get back not just text, but the **images**
> and **tables** from the source pages too.

A Retrieval-Augmented Generation (RAG) chatbot that ingests technical PDFs,
preserves their images and tables, and answers questions with **source
attribution** (document name + page number) for explainable (XAI) results.

- **Embeddings:** local & offline via [`fastembed`](https://github.com/qdrant/fastembed) — model `BAAI/bge-small-en-v1.5` (ONNX, **no PyTorch**)
- **LLM:** Anthropic Claude (answering + summarization)
- **Vector store:** Chroma (collection `Project_NAME`, persisted at `./chroma_Name`)
- **UI:** Streamlit

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Preparing Documents](#preparing-documents)
- [Building the Vector Store](#building-the-vector-store)
- [Running the App](#running-the-app)
- [Embedding Manager GUI](#embedding-manager-gui)
- [Validating the Deployment](#validating-the-deployment)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Teardown](#teardown)
- [Production Notes](#production-notes)
- [Quick Start (TL;DR)](#quick-start-tldr)

---

## Features

- 🔎 Semantic search over technical PDFs
- 🖼️ Returns **images** from the relevant source page (toggle: *Include Image*)
- 📊 Returns **tables** from the relevant source page (toggle: *Include Tables*)
- 🧾 Source attribution: clickable document link + page number per answer
- 🧠 "Ask Claude directly" mode for a concise, synthesized answer
- 📄 `RAW` keyword returns unsummarized source text
- 🔢 **Table content is semantically searchable** — each table's cell values are
  embedded as their own chunk, so queries matching values inside a table retrieve it
- 🧭 **Sidebar question index** — click any past question to jump to that Q&A
- 🧹 **Per-page de-duplication** — the same image/table renders only once per answer
- 🗂️ **Embedding Manager GUI** (`embeddingGUI.py`) — view / upload / delete documents
- 💾 **Chat-to-PDF export** — download the whole conversation (incl. images/tables)
- 💡 Lightweight footprint — local embeddings run on CPU without PyTorch

---

## Architecture

```
                   ┌──────────────────────────────────────────────┐
   PDFs in pdf/ ──▶│  embeddingEngine.py  (run once / on update)   │
                   │   • extracts text (pymupdf4llm)               │
                   │   • extracts tables  -> table/<doc>/<page>/   │
                   │       (lines_strict strategy; cell values     │
                   │        also embedded as searchable text)      │
                   │   • extracts images  -> pdf/<doc>/<page>/      │
                   │   • embeds text (fastembed bge-small)          │
                   └───────────────┬──────────────────────────────┘
                                   ▼
                        Chroma vector store  ./chroma_Name
                                   ▲
                                   │ query (same embedding model)
                   ┌───────────────┴──────────────────────────────┐
   Browser  ◀────▶ │  chatBOT_engine.py  (Streamlit, port 8501)    │
                   │   • retrieves chunks                           │
                   │   • summarizes with Claude                     │
                   │   • renders page images/tables                 │
                   └───────────────┬──────────────────────────────┘
                                   │  <img>/<a> URLs → http://localhost:8000/...
                                   ▼
                   ┌──────────────────────────────────────────────┐
                   │  Static file server (port 8000)               │
                   │   serves PDFs, extracted images, tables, logos │
                   └──────────────────────────────────────────────┘
```

> ⚠️ **Two services are required.** The Streamlit UI embeds asset links as
> `http://localhost:8000/...`, so a **static file server on port 8000** must run
> alongside the app or images, tables, PDF links, and logos will appear broken.

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.10 – 3.12 (3.11 recommended) |
| **pip / venv** | Included with Python |
| **Anthropic API key** | With access to the configured model (default `claude-sonnet-5`) |
| **Internet access** | For pip installs, one-time ONNX model download (~130 MB), and Claude API calls |
| **Disk** | ~2 GB free (venv + model + vector store + extracted assets) |
| **RAM** | 4 GB minimum (8 GB recommended for large PDFs) |

> **SDK caveat:** this project targets `anthropic` **1.11.0**, whose
> `messages.create()` does **not** accept a `temperature` argument — the code omits
> it (Claude uses its server-side default). Do not re-add it.
>
> **pandas caveat:** `DataFrame.applymap` was removed in recent pandas; the code uses
> `DataFrame.map`. Retrieval uses `retriever.invoke(...)` (not the removed
> `get_relevant_documents`).

---

## Installation

```bash
git clone <your-repo-url>
cd ChatBOT

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip

pip install -r requirements.txt
```

Verify dependencies installed:

```bash
pip show fastembed anthropic chromadb streamlit langchain-chroma >/dev/null && echo "DEPS_OK"
```

> ℹ️ `Spire.PDF` and `PyMuPDF` are only needed for **ingestion**
> (`embeddingEngine.py`). A host that only serves the chatbot against a prebuilt
> vector store can omit them — see [Production Notes](#production-notes).

---

## Configuration

Create your environment file from the template and add your key:

```bash
cp .env.example .env
```

```dotenv
# .env
ANTHROPIC_API_KEY=sk-ant-...your-key...
# Optional override:
# ANTHROPIC_MODEL=claude-sonnet-5
```

> 🔐 `.env` is git-ignored. **Never commit your API key** or hardcode it in source.

**(Optional) Branding/avatar assets** — place these in the project root so the
static server can serve them (the UI references them, but the chat works without
them):

- `ericssonlogo.png` (sidebar logo)
- `AI.png` (bot avatar)
- `users.png` (user avatar)
- `icon.png` (alt bot avatar)

---

## Preparing Documents

```bash
mkdir -p pdf
cp /path/to/your/*.pdf pdf/
```

> 📄 By default the extractor **processes every page**. To skip leading
> front-matter pages (e.g. cover + table of contents), set `SKIP_PAGES` in your
> `.env` (or as an env var) to the number of pages to skip — e.g. `SKIP_PAGES=4`
> reproduces the original behavior of skipping the first 4 pages.

---

## Building the Vector Store

Run the ingestion script. It extracts tables/images to disk and writes embeddings
to `./chroma_Name`. Re-run whenever documents change.

```bash
# Optional clean rebuild (required if you previously used a different embedding model)
rm -rf chroma_Name

python embeddingEngine.py
```

This will:

- Create `chroma_Name/` (Chroma persistence)
- Create `table/<doc>/<page>/tableN.pkl` for pages with tables
- Create `pdf/<doc>/<page>/imageN.png` for pages with images
- Download the `bge-small-en-v1.5` ONNX model on first run

> `embeddingEngine.py` calls `reset_collection()` at startup, so each run rebuilds
> the collection cleanly (no duplicate embeddings).

**Sanity check:**

```bash
python - <<'PY'
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
emb = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vs = Chroma("Project_NAME", persist_directory="./chroma_Name",
            embedding_function=emb, collection_metadata={"hnsw:space": "cosine"})
print("DOC_COUNT:", vs._collection.count())
PY
```

Expected: a `DOC_COUNT:` greater than 0.

---

## Running the App

Start **both** services (two terminals, both with the venv activated).

**Terminal 1 — static file server (port 8000):**

```bash
python -m http.server 8000
```

**Terminal 2 — Streamlit chatbot (port 8501):**

```bash
streamlit run chatBOT_engine.py
```

> ⚠️ Launch the chatbot with `streamlit run`, **not** `python chatBOT_engine.py`.

Then open **http://localhost:8501** in your browser.

### Using the chatbot

- Tick **Include Image** / **Include Tables** to render page figures/tables inline.
- Click a past question in the **sidebar** to jump to that Q&A; **Show all** returns
  to the full thread.
- **Ask Claude directly** returns one concise synthesized answer.
- Append `RAW` to a question for unsummarized source text.
- **Prepare PDF of chat** (sidebar) builds a downloadable PDF of the conversation,
  including images and tables.

---

## Embedding Manager GUI

A separate Streamlit app to manage the document library without the CLI:

```bash
streamlit run embeddingGUI.py --server.port 8502
```

- Shows the number of embedded documents and total chunks, and lists each document.
- **Upload a PDF** → saved to `pdf/` → embedded **incrementally** (no collection
  reset) → shows a pages/tables/images summary.
- **Delete a document** → removes its Chroma chunks, its `table/<doc>/` and
  `pdf/<doc>/` assets, and the source PDF. Requires a confirmation checkbox.

> ⚠️ Re-embedding a PDF with the **same filename** appends duplicate chunks (no
> delete-before-readd). Delete it first for a clean re-embed; the GUI warns you.

---

## Validating the Deployment

| # | Action | Expected result |
|---|---|---|
| 1 | Ask a question about your PDF content | Bot response with per-chunk Claude summaries |
| 2 | Check the response header | Shows a **From: `<filename>`** link and **Page Number** markers |
| 3 | Tick **Include Image**, ask about a page with figures | Relevant page image(s) render inline |
| 4 | Tick **Include Tables**, ask about a page with a table | Table renders as HTML |
| 5 | Click **Ask Claude directly** | Concise 3-sentence synthesized answer |
| 6 | Append `RAW` to your question | Raw (unsummarized) chunk text returned |
| 7 | Query a value that lives **inside** a table | The table's page is retrieved (content-level search) |
| 8 | Click a question in the **sidebar** | Main window shows only that Q&A; **Show all** restores the thread |
| 9 | Click **Prepare PDF of chat**, then download | A PDF with the conversation (incl. images/tables) |
| 10 | `streamlit run embeddingGUI.py --server.port 8502` | Shows document count/list; upload & delete work |

Service health checks:

```bash
curl -s -o /dev/null -w "static:%{http_code}\n" http://localhost:8000/
curl -s -o /dev/null -w "app:%{http_code}\n"    http://localhost:8501/
```

Both should return `200`.

---

## Project Structure

```
ChatBOT/
├── chatBOT_engine.py      # Streamlit chatbot (retrieval + Claude + rendering + chat-to-PDF)
├── embeddingEngine.py     # Ingestion: extract text/images/tables + embed (reusable fns)
├── embeddingGUI.py        # Streamlit GUI: view / upload / delete embedded documents
├── htmlTemplates.py       # CSS + chat message HTML templates
├── requirements.txt       # Python dependencies
├── .env.example           # Env template (copy to .env, add API key)
├── .gitignore             # Ignores .env and __pycache__/
├── README.md              # Project overview / write-up
├── INSTALL.md             # This installation / deployment guide
│
├── pdf/                   # Source PDFs + extracted images (pdf/<doc>/<page>/*.png)
├── table/                 # Extracted tables (table/<doc>/<page>/*.pkl)
└── chroma_Name/           # Generated Chroma vector store (git-ignored recommended)
```

> 💡 Recommended `.gitignore` additions for generated artifacts:
> `chroma_Name/`, `table/`, and the extracted image subfolders under `pdf/`.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Images/tables/PDF links broken | Static server (8000) not running / wrong dir | Start `python -m http.server 8000` from project root |
| Auth / `ANTHROPIC_API_KEY` error | `.env` missing or key unset | Recheck [Configuration](#configuration) |
| Anthropic 404 / model not found | `ANTHROPIC_MODEL` unavailable to your account | Set a valid Claude model id in `.env` |
| Empty/garbled answers, dim mismatch | Store built with a different embedding model | `rm -rf chroma_Name` and rebuild |
| First query very slow | One-time ONNX model download | Wait for `bge-small` download to finish |
| `streamlit: command not found` | venv not active / deps missing | `source .venv/bin/activate`, reinstall deps |
| `spire` / `fitz` import error on ingest | Ingestion deps missing | Ensure `Spire.PDF` + `PyMuPDF` installed |
| Content seems skipped | `SKIP_PAGES` set too high | Lower `SKIP_PAGES` in `.env` (default `0` processes all pages) |
| `temperature` TypeError | `temperature` re-added to a Claude call | Remove it — anthropic 1.11.0 doesn't support it |
| `get_relevant_documents` AttributeError | Old LangChain API | Use `retriever.invoke(...)` (already applied) |
| `applymap` AttributeError | Newer pandas | Use `DataFrame.map` (already applied) |
| Garbled table values | Imperfect extraction on complex layouts | `lines_strict` strategy reduces this; verify via the source page link |
| Chat shows nothing after a question | Static server down, or stale UI | Ensure port 8000 is up; the chat renders before the sidebar and PDF export is lazy |

---

## Teardown

```bash
# Stop services with Ctrl+C in each terminal

# Remove generated artifacts (destructive — review first)
rm -rf chroma_Name
rm -rf table
# pdf/ holds BOTH source PDFs and extracted images; delete only extraction
# subfolders if you want to keep sources:
#   find pdf -mindepth 1 -type d -exec rm -rf {} +   # review before running

# Remove the virtual environment
deactivate 2>/dev/null; rm -rf .venv
```

> ⚠️ `rm -rf` is irreversible. Back up `pdf/` source documents before cleanup.

---

## Production Notes

For anything beyond local/dev use:

- **Split hosts:** run ingestion (needs `Spire.PDF`, `PyMuPDF`, `pymupdf4llm`) on
  one host, then copy `chroma_Name/`, `pdf/`, and `table/` to a slimmer serving
  host. Both hosts **must use the same embedding model** (`BAAI/bge-small-en-v1.5`).
- **Reverse proxy + TLS:** front Streamlit (8501) and the static server (8000)
  with nginx/Caddy; terminate TLS there.
- **Do not expose `python -m http.server` to the internet.** Replace with nginx
  static serving scoped to the asset directories, behind authentication.
- **Auth:** put SSO/auth in front of the UI via the reverse proxy.
- **Process supervision:** run both services under systemd/supervisord/containers.
- **Secrets:** store `ANTHROPIC_API_KEY` in a secrets manager, not plaintext `.env`.
- **Reproducibility:** pin dependency versions in `requirements.txt`.

---

## Quick Start (TL;DR)

```bash
git clone <your-repo-url> && cd ChatBOT
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
cp .env.example .env            # edit .env → add ANTHROPIC_API_KEY
mkdir -p pdf && cp /path/to/*.pdf pdf/
python embeddingEngine.py       # build vector store + extract images/tables

# Terminal 1
python -m http.server 8000
# Terminal 2
streamlit run chatBOT_engine.py
# → open http://localhost:8501
```
