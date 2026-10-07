# 📚 Technical-Document RAG Chatbot

> Query technical **PDFs and PowerPoint decks** with an LLM — and get back not just
> text, but the **images** and **tables** from the source pages/slides too.

A Retrieval-Augmented Generation (RAG) chatbot that ingests technical **PDFs and
PPTX decks**, preserves their images and tables, and answers questions with
**source attribution** (document name + page/slide number) for explainable (XAI)
results.

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

- 🔎 Semantic search over technical **PDFs and PPTX decks** (**MMR** retrieval:
  `k=8`, `fetch_k=24`, `lambda_mult=0.6`)
- 💬 **Chat UI** built on Streamlit chat primitives (`st.chat_input` / `st.chat_message`)
- 🧠 **Conversational memory** — follow-ups like "what about at 85°C?" are rewritten
  into standalone queries using chat history before retrieval
- 💡 **Suggested follow-up questions** rendered as clickable buttons after each answer
- 🌊 **Streamed, synthesized answer** — one grounded answer per question, not a
  per-chunk dump
- 🧾 **Inline `[document p.N]` citations** for each claim
- 🖼️ Returns **images** from cited pages (sidebar toggle: *Include figures from cited pages*)
- 📊 Returns **tables** from cited pages (sidebar toggle: *Include tables from cited pages*)
- ⬇️ **Download button** for each cited source document
- 🚧 **Grounded**: says "I don't know" when the answer isn't in the documents
- 🔢 **Table content is semantically searchable** — each table's cell values are
  embedded as their own chunk, so queries matching values inside a table retrieve it
- 🧹 **Per-page de-duplication** — the same image/table renders only once per answer
- 🧹 **Clear chat** and 📄 **chat-to-PDF export** in the sidebar
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
                   │   • renders page images/tables inline          │
                   │     (base64 data: URIs / HTML — no server)     │
                   └────────────────────────────────────────────────┘
```

> **One service only.** The Streamlit UI inlines images, tables, and the source-PDF
> link directly into each answer (images/PDFs as base64 `data:` URIs, tables as
> HTML), so no companion static file server is required. Source PDFs larger than
> 8 MB are shown as a plain filename label rather than an inline download link.

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
# Optional: path to the local Qwen GGUF (only for the "Qwen" answer engine)
# QWEN_GGUF_PATH=models/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

> 🔐 `.env` is git-ignored and holds **only secrets** (the API key). **Never commit
> your API key** or hardcode it in source.

### Answer engine (LLM provider)

The chatbot's answer backend is single-select, stored in **`config.json`** (tracked,
non-secret, default `{"llm_provider": "none"}`). Choose it in the Embedding Manager
GUI sidebar; the chatbot reads it once per session and uses it transparently.

| Provider | Needs |
|---|---|
| `claude` | `ANTHROPIC_API_KEY` in `.env` |
| `qwen` | `llama-cpp-python` (in `requirements.txt`) + a Qwen2.5-1.5B-Instruct **GGUF** (~1 GB) |
| `none` | nothing — extractive (shows retrieved passages verbatim) |

For the **Qwen** option, download the GGUF once (not shipped in the repo):

```bash
mkdir -p models
wget -c -O models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf?download=true"
# or set QWEN_GGUF_PATH in .env to an existing .gguf path
```

> ⚠️ **Qwen trade-offs:** it runs locally on **CPU**, so it's **slow (~1–2 min per
> answer)** and may **omit the inline `[file p.N]` citations** that Claude produces.
> The chatbot shows a warning banner when Qwen is active. Source **download buttons**
> still appear in all modes. Building `llama-cpp-python` needs a C/C++ toolchain
> (e.g. `build-essential`).

If the selected engine isn't available at runtime, the chatbot auto-falls back to
`none` (extractive), so it never breaks. A fresh clone defaults to `none` and runs
with no API key and no model download. **Provider changes take effect on the next
chat session — refresh the chatbot tab after switching.**

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
mkdir -p pdf pptx
cp /path/to/your/*.pdf  pdf/      # PDFs
cp /path/to/your/*.pptx pptx/     # PowerPoint decks
```

> 📄 By default the extractor **processes every page/slide**. To skip leading
> front-matter pages (e.g. cover + table of contents), set `SKIP_PAGES` in your
> `.env` (or as an env var) to the number of pages/slides to skip — e.g.
> `SKIP_PAGES=4` skips the first 4.
>
> 📊 **PPTX:** `python-pptx` extracts native PowerPoint tables and inserted raster
> pictures. Charts, SmartArt, and vector/grouped graphics are **not** extracted.

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

Only **one** service is required — the UI inlines images, tables, and source-PDF
links directly into the page (base64 `data:` URIs / HTML), so no static file
server is needed.

**Streamlit chatbot (port 8501):**

```bash
streamlit run chatBOT_engine.py
```

> ⚠️ Launch the chatbot with `streamlit run`, **not** `python chatBOT_engine.py`.

Then open **http://localhost:8501** in your browser.

### Using the chatbot

- Type a question in the chat box and press Enter; the answer **streams in** as a
  single synthesized response with inline **`[document p.N]` citations**.
- Tick **Include figures from cited pages** / **Include tables from cited pages** in
  the sidebar to render the figures/tables from the pages the answer cited.
- A **download button** appears under each answer for every cited source document.
- **Clear chat** (sidebar) resets the conversation.
- **Prepare PDF of chat** (sidebar) builds a downloadable PDF of the conversation,
  including images and tables.

---

## Embedding Manager GUI

A separate Streamlit app to manage the document library without the CLI:

```bash
streamlit run embeddingGUI.py --server.port 8502
```

- Shows the number of embedded documents and total chunks, and lists each document.
- **Upload a PDF or PPTX** → saved to `pdf/` or `pptx/` → embedded **incrementally**
  (no collection reset) → shows a pages/tables/images summary.
- **Delete a document** → removes its Chroma chunks, its `table/<doc>/` and
  `pdf/<doc>/` assets, and the source file. Requires a confirmation checkbox.

> ⚠️ Re-embedding a PDF with the **same filename** appends duplicate chunks (no
> delete-before-readd). Delete it first for a clean re-embed; the GUI warns you.

---

## Validating the Deployment

| # | Action | Expected result |
|---|---|---|
| 1 | Ask a question about your PDF content | A single **streamed, synthesized** answer appears in a chat bubble |
| 2 | Read the answer | Each claim carries an inline **`[filename p.N]`** citation |
| 3 | Ask something not in the docs | The assistant says it doesn't know (grounded, no hallucination) |
| 3b | Ask a question, then a short follow-up (e.g. "what about at 85°C?") | The follow-up is answered in context; clickable **suggested follow-ups** appear under answers |
| 4 | Tick **Include figures from cited pages**, ask about a cited page with figures | Image(s) render below the answer |
| 5 | Tick **Include tables from cited pages**, ask about a cited page with a table | Table renders as HTML below the answer |
| 6 | Query a value that lives **inside** a table | The table's page is retrieved (content-level search) |
| 7 | Check under an answer | A **⬇️ Download source** button per cited document |
| 8 | Ask a very long question | Full answer streams; if capped, a "_…(response truncated)_" note is appended |
| 9 | Click **Clear chat** / **Prepare PDF of chat** | Conversation resets / a PDF transcript (incl. images/tables) downloads |
| 10 | `streamlit run embeddingGUI.py --server.port 8502` | Shows document count/list; upload & delete work |

Service health checks:

```bash
curl -s -o /dev/null -w "app:%{http_code}\n" http://localhost:8501/
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
| Images not rendering | Image never extracted for that page, or asset deleted | Confirm `pdf/<doc>/<page>/*.png` exists; re-embed the PDF if missing |
| Auth / `ANTHROPIC_API_KEY` error | `.env` missing or key unset | Recheck [Configuration](#configuration) |
| Anthropic 404 / model not found | `ANTHROPIC_MODEL` unavailable to your account | Set a valid Claude model id in `.env` |
| Empty/garbled answers, dim mismatch | Store built with a different embedding model | `rm -rf chroma_Name` and rebuild |
| First query very slow | One-time ONNX model download | Wait for `bge-small` download to finish |
| `streamlit: command not found` | venv not active / deps missing | `source .venv/bin/activate`, reinstall deps |
| `spire` / `fitz` import error on ingest | Ingestion deps missing | Ensure `Spire.PDF` + `PyMuPDF` installed |
| Content seems skipped | `SKIP_PAGES` set too high | Lower `SKIP_PAGES` in `.env` (default `0` processes all pages) |
| Provider change not applied | Provider read once per session | Refresh the chatbot tab for a new session |
| Selected Qwen but answers are extractive | GGUF missing / `llama-cpp-python` not built → auto-fallback | Download the GGUF to `models/` (or set `QWEN_GGUF_PATH`); `pip install -r requirements.txt` |
| Qwen very slow | 1.5B model on CPU | Expected (~1–2 min/answer); use Claude for speed |
| Qwen omits `[file p.N]` citations | Small-model limitation | Expected; source download buttons still appear; use Claude for strict citations |
| `llama-cpp-python` build fails | Missing C/C++ toolchain | Install `build-essential` (Debian/Ubuntu) or skip Qwen (use `claude`/`none`) |
| `temperature` TypeError | `temperature` re-added to a Claude call | Remove it — anthropic 1.11.0 doesn't support it |
| `get_relevant_documents` AttributeError | Old LangChain API | Use `retriever.invoke(...)` (already applied) |
| `applymap` AttributeError | Newer pandas | Use `DataFrame.map` (already applied) |
| Garbled table values | Imperfect extraction on complex layouts | `lines_strict` strategy reduces this; verify via the source page link |
| Chat shows nothing after a question | Stale UI | Re-run the query; the chat renders before the sidebar and PDF export is lazy |

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
- **Auth:** put SSO/auth in front of the UI via the reverse proxy.
- **Process supervision:** run the app under systemd/supervisord/containers.
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

streamlit run chatBOT_engine.py
# → open http://localhost:8501
```
