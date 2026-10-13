# 📚 Technical-Document RAG Chatbot (with Tables & Images)

Query technical **PDFs and PowerPoint decks** with an LLM — and get back not just
text, but the **images** and **tables** from the source pages/slides too, with
**source attribution** (document name + page/slide number) for explainable (XAI)
results.

![The chatbot answering a question with a table rendered inline from the source
document, plus an inline page citation](docs/images/chatbot-inline-table.png)

> *Ask a question → get a synthesized answer with the **actual table from the
> source page** rendered inline and a `[document p.N]` citation. Most RAG apps
> would have lost that table during ingestion.*

## The problem this solves

A technical document is not just text — it has images (architecture diagrams,
formulas) and tables (specifications, parameters) that often carry more value than
the prose. A regular RAG pipeline embeds only text chunks, so during vectorization
the image and table information is lost. Even if table text is scraped, the table
*structure* is lost. Querying such a store returns text only — frequently unhelpful
and not confidence-inspiring.

This project keeps the images and tables. It is "smart" during ingestion (extracting
and saving tables/images keyed by document + page) and uses a little ingenuity at
query time (reconstructing those assets from metadata and rendering them inline).

## How it works (three parts)

- **Part 1 — Ingestion (`embeddingEngine.py`):** for each **PDF page or PPTX slide**,
  extract text, tables, and images. Embed the text (and the table content too) into a
  Chroma vector store with metadata; save tables and images to disk keyed by document
  and page/slide number. A small dispatcher (`ingest_file`) routes `.pdf` and
  `.pptx`/`.ppt` to the right extractor; **slides map to page numbers**, so everything
  downstream treats a deck exactly like a PDF.
- **Part 2 — Chatbot (`chatBOT_engine.py`):** a Streamlit chat app (`st.chat_input` /
  `st.chat_message`). For a **follow-up**, it first rewrites the question into a
  standalone query using recent chat history, then runs MMR vector search, then
  **streams a single synthesized answer** grounded in the retrieved context with
  **inline `[document p.N]` citations**. Based on sidebar toggles it also
  reconstructs and renders the images/tables from the **cited** pages, offers a
  download button for each cited source, and suggests **clickable follow-up
  questions**.
- **Part 3 — Embeddable widget (`api_server.py` + `widget/`):** the same retrieval
  and answer logic (via a shared, UI-free `generate_answer()`) exposed over an HTTP
  API (FastAPI) and a one-line **JavaScript widget** that drops a floating chat bubble
  onto *any* website. The Streamlit app and the widget share the exact same backend;
  the widget just renders in the host page instead of Streamlit, with **clickable
  source citations** and suggested follow-ups. See
  [Embeddable chat widget](#embeddable-chat-widget).

## Toolset

- **Embeddings:** local & offline via [`fastembed`](https://github.com/qdrant/fastembed),
  model `BAAI/bge-small-en-v1.5` (ONNX, **no PyTorch**)
- **Answer engine (pluggable, single-select):** **Claude** (Anthropic API),
  **Qwen 1.5B** (local SLM via `llama-cpp-python`, offline), or **None** (extractive —
  show the retrieved passages verbatim, no generation). Selected in the Embedding
  Manager GUI; the chatbot uses it transparently.
- **Retrieval:** Chroma with **MMR** (max-marginal-relevance; `k=8`, `fetch_k=24`,
  `lambda_mult=0.6`) for better recall and less near-duplicate noise
- **Vector store:** Chroma — collection `Project_NAME`, persisted at `./chroma_Name`
- **PDF processing:** `pymupdf4llm` (text), `pymupdf`/`fitz` (tables), `Spire.PDF` (images)
- **PPTX processing:** `python-pptx` (slide text, native tables, inserted pictures)
- **UI:** Streamlit chat primitives (`st.chat_input` / `st.chat_message`)
- **Chat-to-PDF export:** `xhtml2pdf`
- **Q&A history logging:** Postgres (`postgres:16-alpine`) via `psycopg` (psycopg 3)
  — every answered question + metadata is logged to a `qa_log` table (fail-safe)

---

## Core strategy: "page-as-anchor" sidecar storage

The join key between embedded text and non-text assets is **`file_name` + `pagenumber`**
(for PPTX, the **slide index is the `pagenumber`**).

| Asset  | Extracted with (PDF / PPTX)              | Stored as                               |
|--------|------------------------------------------|-----------------------------------------|
| Text   | `pymupdf4llm` / `python-pptx` text frames | embedding in Chroma                     |
| Tables | `pymupdf` `find_tables` / `python-pptx` table shapes | pickled DataFrame `table/<doc>/<page>/tableN.pkl` |
| Images | `Spire.PDF` ImageHelper / `python-pptx` picture shapes | PNG `pdf/<doc>/<page>/imageN.png`       |

Every text chunk is embedded with metadata:

```python
{"file_name": file, "pagenumber": page,
 "imagePresent": 0/1, "tablePresent": 0/1}
```

At query time, vector search returns text chunks. If a chunk's page has an
image/table and the user ticked the box, the app reconstructs the on-disk path from
`(file_name, pagenumber)` and renders the asset inline.

**Table content is also semantically searchable:** each extracted table's cell values
are rendered to text and embedded as their own chunk (`contentType: "table"`), so a
query matching the table's *content* (not just nearby prose) retrieves that page.

---

## Files

| File | Purpose |
|---|---|
| `embeddingEngine.py` | Ingestion: extract text/tables/images + embed. Reusable functions `get_vectorstore`, `ingest_pdf`, `ingest_pptx`, `ingest_file` (dispatcher), `list_embedded_documents`, `delete_document`. |
| `chatBOT_engine.py` | Streamlit chatbot (retrieval + Claude + rendering + chat-to-PDF). Also hosts the UI-free `generate_answer()` shared with the widget API. |
| `embeddingGUI.py` | Streamlit GUI to view / upload / delete embedded documents. |
| `api_server.py` | FastAPI backend for the embeddable widget (`/api/chat`, `/api/health`, `/api/source/{name}`). |
| `db.py` | Postgres-backed Q&A history logging (writes every answered question + metadata to the `qa_log` table; stats/query/purge helpers). Fail-safe. |
| `widget/` | Embeddable JS chat widget (`widget.js`, `widget.css`). |
| `demo-site/` | Fake third-party site that embeds the widget (cross-origin demo). |
| `htmlTemplates.py` | CSS + chat message HTML templates. |
| `requirements.txt` | Python dependencies. |
| `.env` | Holds `ANTHROPIC_API_KEY` and the Postgres credentials incl. the DB password (git-ignored). |

Generated at runtime: `chroma_Name/` (vector store), `table/<doc>/<page>/*.pkl`,
`pdf/<doc>/<page>/*.png`.

---

## Prerequisites

| Requirement | Details |
|---|---|
| Python | 3.10 – 3.12 |
| Anthropic API key | Only for the **Claude** answer engine (optional — `none`/`qwen` need no key) |
| C/C++ toolchain | Only for the **Qwen** engine (to build `llama-cpp-python`) — e.g. `build-essential` on Debian/Ubuntu |
| Internet | pip installs, one-time ONNX model download (~130 MB); Claude API calls / one-time ~1 GB Qwen GGUF download only if those engines are used |

> **Note on the Anthropic SDK:** this project targets `anthropic` **1.11.0**, whose
> `messages.create()` does **not** accept a `temperature` argument. The code therefore
> omits `temperature` (Claude uses its server-side default). Do not re-add it.

---

## Installation

```bash
cd doc-intelligence-rag
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create `.env` and add your key:

```dotenv
ANTHROPIC_API_KEY=sk-ant-...your-key...
# Optional:
# ANTHROPIC_MODEL=<model-id>
# SKIP_PAGES=0        # leading pages to skip per PDF (cover/TOC); default 0

# Postgres Q&A history logging (the DB PASSWORD is a secret → it lives here):
POSTGRES_DB=techdoc
POSTGRES_USER=techdoc
POSTGRES_PASSWORD=change-me
DB_HOST=db           # 'db' is the compose service name; use 'localhost' outside Docker
DB_PORT=5432
```

> **Secret vs non-secret split (firm project rule):** only *secrets* belong in
> `.env` — including the DB **password** (`POSTGRES_PASSWORD`) and the Postgres
> connection settings above. The non-secret Q&A-logging **toggle** and **retention**
> (`qa_logging_enabled`, `qa_log_retention_days`) live in `config.json` and are
> editable from the GUI — see [Q&A history logging](#qa-history-logging-postgres).

Logging verbosity and an optional log file are **non-secret settings**, so they
live in **`config.json`** (`log_level`, `log_file`) alongside the answer-engine and
confidence-threshold settings — set them from the Embedding Manager GUI sidebar
(**Logging** section) or by editing `config.json`. The app logs one structured line
per answered question (provider, retrieval breadth, best relevance score,
low-confidence/refusal flags, source count, latency). Logs go to stderr, so
`docker compose logs -f chatbot` shows them; set `log_file` to also persist them.
To get a log file **on the host** when running in Docker, use a path under the
mounted `logs/` folder — e.g. set `log_file` to `logs/app.log` and it appears at
`./logs/app.log` on your machine (the compose file mounts `./logs` into each
service). `config.json` is the single source of truth for logging — there are no
logging env vars.

Optional branding assets in the project root (inlined into the page as base64 at
load time, if present): `AI.png`, `users.png`, `logo.png`. Missing assets
are skipped gracefully — no broken image icons.

---

## Building the vector store

Put PDFs in `pdf/` and PowerPoint decks in `pptx/`, then run the batch ingester
(this **resets** the collection for a clean full rebuild):

```bash
mkdir -p pdf pptx
cp /path/to/*.pdf  pdf/
cp /path/to/*.pptx pptx/
python embeddingEngine.py
```

This creates `chroma_Name/`, extracts tables to `table/<doc>/<page>/` and images to
`pdf/<doc>/<page>/` (slides use the **slide index** as the page), embeds page/slide
text, and embeds each table's content as a searchable chunk. First run downloads the
`bge-small-en-v1.5` ONNX model.

For **incremental** add/remove without a full rebuild, use the Embedding Manager GUI
(below) or the `ingest_file` / `delete_document` functions.

> **Newly ingested documents are retrievable immediately — no restart needed.** The
> chatbot and the widget API are long-lived processes, while documents are ingested
> by a *separate* process (the Embedding Manager GUI, a batch `embeddingEngine.py`
> run, or another container). Chroma caches its in-memory vector index per process,
> so a reader process would otherwise keep answering from a stale index and report
> "I don't have any information about that" for a just-added document until it was
> restarted. To prevent this, the chatbot reloads the Chroma index from disk on every
> query (`_open_vectorstore()` clears Chroma's cached client first), so a document
> embedded a moment ago is found on the very next question with no restart.

> **PPTX note:** `python-pptx` extracts **native PowerPoint tables** and **inserted
> raster pictures**. Charts, SmartArt, and grouped/vector graphics are not raster
> images and are **not** extracted — reference them in text, or paste them as a
> picture if you need them rendered.

---

## Running the app

Only **one** service is required. Images, tables, and the source-PDF links are
inlined directly into the page (images and PDFs as base64 `data:` URIs, tables as
HTML), so **no static file server is needed**.

**Streamlit chatbot (port 8501):**

```bash
streamlit run chatBOT_engine.py
```

> Launch with `streamlit run`, **not** `python chatBOT_engine.py`.

Open **http://localhost:8501**.

> Images and the source-PDF download link are embedded in the answer itself, so
> they render without any companion server. Very large source PDFs (>8 MB) are
> shown as a plain filename label instead of an inline download link to keep the
> page lightweight.

### Docker (whole app in containers)

The entire application — chatbot **and** Embedding Manager GUI — runs from one
image as services that share all runtime data via volumes:

```bash
cp .env.example .env            # add ANTHROPIC_API_KEY (Claude engine only)
docker compose up --build       # chatbot → :8501, GUI → :8502
```

Both the chatbot and GUI expose a health endpoint (`/_stcore/health`). See
[INSTALL.md → Docker deployment](INSTALL.md#docker-deployment) for the plain-`docker`
fallback, volume layout, and the optional Qwen model mount.

> `docker compose up` also starts the **widget API** (`:8000`), a **demo site**
> (`:9000`), and a **Postgres `db`** service — see
> [Embeddable chat widget](#embeddable-chat-widget) and
> [Q&A history logging](#qa-history-logging-postgres) below.

In total the compose stack runs **five services**: `chatbot` (:8501), `gui`
(:8502), `api` (:8000), `demo` (:9000), and `db` (Postgres `postgres:16-alpine`,
container `techdoc-db`; internal port **5432**, **not** published to the host). The
`db` service persists its data in a named volume `pgdata` (see
[Q&A history logging](#qa-history-logging-postgres)).

---

## Embeddable chat widget

Besides the standalone Streamlit chatbot, the same RAG backend is exposed as an
**embeddable chat widget** that drops onto any website with a single `<script>` tag.
It's a floating chat bubble (bottom-right) that opens a panel, talks to an HTTP API,
and renders answers with **clickable source citations** and suggested follow-ups.

![The embeddable chat widget open on a host web page, answering a question with
inline citations, a source chip, and clickable suggested follow-up
questions](docs/images/embeddable-widget.png)

Three pieces:

| Piece | File(s) | What it is |
|---|---|---|
| **API** | `api_server.py` | FastAPI over the shared `generate_answer()` logic: `POST /api/chat`, `GET /api/health`, `GET /api/source/{name}` (serve a cited doc), plus the widget static files. |
| **Widget** | `widget/widget.js`, `widget/widget.css` | Vanilla-JS floating bubble; no framework; embeds anywhere. |
| **Demo site** | `demo-site/index.html` | A fake third-party website that embeds the widget to test cross-origin. |

### Run it (Docker)

`docker compose up --build` starts everything, including:
- **Widget API** → http://localhost:8000 (`/api/health`, `/api/chat`)
- **Demo site** → http://localhost:9000 (fake site with the widget embedded)

Open **http://localhost:9000**, click the chat bubble, and ask a question — the
answer comes from the RAG backend on `:8000` (a *different origin*, proving the
cross-origin embed works).

### Run it (without Docker)

```bash
# API (serves the widget + answers)
.venv/bin/uvicorn api_server:app --host 0.0.0.0 --port 8000
# Fake site on a different origin (new terminal)
.venv/bin/python -m http.server 9000 --directory demo-site
# → open http://localhost:9000
```

### Putting the widget on a third-party website

The site owner adds **one line** before `</body>`:

```html
<script src="https://YOUR_HOST/widget/widget.js"
        data-api="https://YOUR_HOST/api"></script>
```

`data-api` points at your hosted API. For this to work from another domain, **your
side** must be set up:

- **Public, HTTPS API.** `localhost` only works on your own machine; a real embed
  needs a public host with TLS (browsers block `https://` pages from calling an
  `http://` API as mixed content). Put the API behind nginx/Caddy for TLS.
- **CORS.** The API currently allows all origins (`allow_origins=["*"]`) — fine for
  the demo. For production, restrict it to the domains you authorize.
- **⚠️ Authentication / abuse control.** The API is currently **open and
  unauthenticated** — anyone who knows the URL can call `/api/chat` (spending your
  Claude budget) and `/api/source/...` (downloading your documents). This is fine for
  a local demo but **must** be addressed before exposing it publicly, especially for
  internal/confidential material — add an origin allowlist, an API key, and rate
  limiting. (Tracked as the authentication wishlist item.)

The `/api/source/{name}` endpoint only serves whitelisted document types
(`.pdf`/`.pptx`/`.ppt`) by basename from the `pdf/`/`pptx/` folders, with
path-traversal protection.

### Configuring the widget greeting

The message shown when a visitor opens the chat panel is **configurable** via
`config.json` (`widget_welcome_message`), editable from the Embedding Manager GUI
sidebar (**Chat widget** section). The widget fetches it from `GET /api/health` each
time it opens, so a change takes effect on the **next session with no restart**. Set
it to an empty string to show **no greeting** at all.

### Configuring the widget appearance

The widget's **title**, input **placeholder**, and **accent color** are also
configurable in `config.json` (editable from the GUI **Chat widget** section):

| Key | Default | Notes |
|---|---|---|
| `widget_title` | `📚 Document Assistant` | Header title |
| `widget_input_placeholder` | `Ask about the documents…` | Text-box placeholder |
| `widget_accent_color` | `#2e7d32` | Any valid CSS color; drives the whole theme via a CSS variable |

The widget reads these from `GET /api/health` when it loads, so changes apply on the
next widget load. The accent color is validated server-side (must be a valid
`#hex`/`rgb()`/`hsl()`/named color) to prevent injection.

### Configuring the Streamlit chatbot UI

The standalone chatbot's **title**, **subtitle**, **input placeholder**, and sidebar
**labels** are configurable in `config.json` (editable from the Admin Console
**🤖 Chatbot** tab). You can also **hide the figures/tables answer options entirely**
(`chatbot_show_figures_option` / `chatbot_show_tables_option`) for deployments whose
documents have no images or tables — when hidden the feature is simply off. The
chatbot reads these once per session (refresh the chatbot tab to apply).

---

## Embedding Manager GUI

A separate Streamlit app to manage the document library without the CLI:

![The Admin Console on the Documents tab — document/chunk counts, the embedded
documents table, and the tab bar (Documents, Answer engine, Chatbot, Chat widget,
Logging, Q&A History)](docs/images/admin-console.png)

```bash
streamlit run embeddingGUI.py --server.port 8502
```

Features:
- **Answer engine selector (sidebar):** choose **Claude**, **Qwen (local)**, or
  **None (extractive)** — single-select, one at a time. The choice is written to
  `config.json` (tracked, non-secret) and the chatbot reads it **once per session**,
  transparently (chat users don't see or choose the engine). See **Answer engine**
  below.
- Shows the number of embedded documents and total chunks, and lists each document.
- **Upload a PDF or PPTX** → saved to `pdf/` or `pptx/` → embedded **incrementally**
  (no collection reset) → shows a pages/tables/images summary.
- **Delete a document** → removes its Chroma chunks, its `table/<doc>/` and `pdf/<doc>/`
  assets, and the source file. Requires a confirmation checkbox (destructive).

> **Re-embed is a safe upsert (content-hash keyed).** Each document is identified by
> a SHA-256 hash of its contents, stored on every chunk as `doc_hash`. Re-embedding a
> file with **identical content** is a no-op (skipped — no duplicate chunks).
> Re-embedding **changed content** (or the same filename with new bytes) first deletes
> the old chunks and stale extracted assets, then re-adds — so duplicates never
> accumulate.

The Admin Console is organized into tabs: **📚 Documents | ⚙️ Answer engine |
🤖 Chatbot | 💬 Chat widget | 🔧 Logging | 📊 Q&A History**. The last of these is
described next.

---

## Q&A history logging (Postgres)

Every answered question is logged to a Postgres table (`qa_log`) along with its
response and metadata — for both the **Streamlit chatbot** and the **embeddable
widget**, because the write happens in the shared `generate_answer()` core (plus the
Streamlit `answer_question` path). Each row is tagged `source='chatbot'` or
`source='widget'`.

**Stored columns:** `asked_at`, `source`, `question`, `answer`, `provider`,
`best_score`, `low_confidence` (bool), `refusal` (bool), `sources` (JSONB list of
cited docs), `latency_ms`.

**Fail-safe by design.** Logging is best-effort: if Postgres is unreachable,
answering still works — the write silently no-ops. The connection fails **fast**, so
logging never adds latency to an answer.

**Config split (firm project rule):**

| Setting | Lives in | Keys | Default |
|---|---|---|---|
| DB **password** + connection (secret) | `.env` | `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `DB_HOST`, `DB_PORT` | — |
| Logging **toggle** + **retention** (non-secret) | `config.json` | `qa_logging_enabled`, `qa_log_retention_days` | `true`, `90` |

The non-secret toggle and retention are editable from the Admin Console **📊 Q&A
History** tab, which provides:

- **Stats metrics** — totals and breakdowns at a glance.
- **Filters** — by `source`, free-text search, and a row limit.
- A **read-only table** of recent Q&A rows.
- **CSV export** of the filtered rows.
- **Maintenance** — purge rows older than `qa_log_retention_days`, or clear all
  (destructive).

**Dependency & storage.** Logging uses `psycopg[binary,pool]` (psycopg 3). Under
Docker the `db` service is `postgres:16-alpine` (container `techdoc-db`, internal
port **5432**, not published to the host) and persists its data in a named volume
`pgdata` — it survives `docker compose down`/`up`. `docker compose down -v` **wipes**
`pgdata` (destructive).

---

## Answer engine (pluggable LLM provider)

The chatbot's answer backend is **single-select** and chosen in the Embedding Manager
GUI sidebar. Exactly one is active at a time, stored in `config.json` (tracked,
non-secret). The chatbot reads it **once per session** and uses it transparently.

| Provider | What it does | Requirements | Notes |
|---|---|---|---|
| **Claude** | Streamed synthesized answer with inline citations (best quality) | `ANTHROPIC_API_KEY` in `.env` | Fast; reliable `[file p.N]` citations |
| **Qwen (local)** | Same synthesis, run **offline** by a small local model | `llama-cpp-python` + the Qwen GGUF file (~1 GB) | **Slow on CPU (~1–2 min/answer)**; may omit inline citations; shows a warning banner |
| **None (extractive)** | No generation — shows the retrieved passages verbatim under `[file p.N]` headers | nothing (uses only the embedding model) | Instant; no API, no model |

**Auto-fallback:** if the selected provider isn't actually available at runtime
(missing API key, or missing GGUF file), the chatbot silently falls back to
**extractive** so it never breaks. Figures/tables and source downloads work in **all
three** modes (Qwen may omit inline citation markers, but its source download buttons
still appear).

> `config.json` is **tracked** (it's non-secret config); `.env` holds only the secret
> `ANTHROPIC_API_KEY`. A fresh clone defaults to `none` (extractive) and runs with no
> key and no model download.

### Setting up the local Qwen model (only for the "Qwen" option)

Download a Qwen2.5-1.5B-Instruct **GGUF** (q4_k_m is a good size/quality balance) and
place it where the app expects it:

```bash
mkdir -p models
# ~1 GB download from the official Hugging Face GGUF repo:
wget -c -O models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf?download=true"
```

Default path: `models/qwen2.5-1.5b-instruct-q4_k_m.gguf` (override with
`QWEN_GGUF_PATH` in `.env`). The `.gguf` file is **not** shipped in the repo (it's
git-ignored). It runs in-process on CPU via `llama-cpp-python` — no server, no Ollama.

**Performance & quality (be realistic):**
- **Slow on CPU** — roughly **1–2 minutes per answer**, plus a one-time model load
  (~a few seconds) on the first question of a session. The chatbot shows a warning
  banner when the Qwen engine is active so users know to expect the wait. Answers
  stream in as they generate.
- **Weaker citations** — the 1.5B model answers correctly and grounded, but may
  **omit the inline `[file p.N]` markers** that Claude emits reliably. The source
  **download buttons** still appear (driven by retrieval, not the answer text), so
  attribution isn't lost. For strict inline citations, use Claude.

---
## Using the chatbot

- Type a question in the chat box at the bottom and press Enter.
- The answer **streams in** as a single synthesized response, grounded in your
  documents, with inline **`[document p.N]` citations** for each claim.
- In the **sidebar**, tick **Include figures from cited pages** / **Include tables
  from cited pages** to render the figures/tables from the pages the answer cited.
- A **download button** appears under each answer for every source document it cited.
- **Follow-up questions are understood in context.** Ask "What's the Pico's VBUS
  current?" then "What about at 85°C?" — the chatbot rewrites the follow-up into a
  standalone query (resolving "that"/"it"/omitted subjects) using recent history
  **before** retrieving, so it finds the right page.
- **Suggested follow-ups** appear as clickable buttons under each answer — click one
  to ask it next. (Both features use the active LLM; in extractive mode they no-op.)
- **Clear chat** (sidebar) resets the conversation.
- **Prepare PDF of chat** (sidebar) builds a downloadable PDF of the whole conversation,
  including the images and tables.

### Behaviour notes

- The answer is **grounded**: if the retrieved context doesn't contain the answer,
  the assistant says it doesn't know rather than inventing facts. On such a grounded
  refusal, **no source downloads, figures/tables, or suggested follow-ups are shown**
  (they'd otherwise reflect off-topic retrieved chunks, not the user's question).
  Refusal detection now recognizes **paraphrased** refusals too (e.g. "I don't have
  any information about X"), so an out-of-context question no longer shows stray
  source-document links.
- **Suggested follow-ups are suppressed when a question is out of context** — on a
  grounded refusal **or** a low-confidence result (previously only on a refusal).
- **Low-confidence banner.** Before answering, the app reads the single best
  retrieval relevance score; if it falls below a configurable threshold
  (`low_confidence_threshold` in `config.json`, default **0.63**), a non-blocking
  "⚠️ Low confidence" banner appears above the answer. The answer still renders — the
  banner is advisory. Tune the threshold from the Embedding Manager GUI slider, or set
  it to `0.0` to disable. The default is **data-backed** by the evaluation harness
  (see below).
- **Citations drive attribution.** Rendered figures/tables and the source download
  buttons are scoped to the pages the answer actually **cited**, so a chunk that
  retrieval pulled in but the answer didn't use won't show a stray asset or download.
- Retrieval uses **MMR** (`k=8`, `fetch_k=24`, `lambda_mult=0.6`) to improve recall of
  the defining chunk and reduce near-duplicate noise.
- Images and tables are **de-duplicated per page**.
- Very long answers are capped at `max_tokens=2048`; if an answer is truncated, a
  clear "_…(response truncated)_" note is appended rather than leaving a broken line.
- Tables render as styled HTML (header row + bordered cells). Rendering quality depends
  on extraction quality — verify complex tables against the source via the page number.

---

## Security

**Prompt-injection defense (indirect injection via document content).** A RAG app
pastes retrieved document text into the LLM prompt, so a malicious or poisoned
document could carry instructions like *"ignore previous instructions and…"*. The
app defends against this:

- The retrieved context is **fenced as untrusted data** between
  `<<<BEGIN_UNTRUSTED_CONTEXT>>>` / `<<<END_UNTRUSTED_CONTEXT>>>` markers, and the
  **system prompt establishes an instruction hierarchy**: everything inside the fence
  is reference *data* to quote and cite, **never** instructions to obey. Any
  directive-looking text in a document is treated as ordinary content.
- The fence helper **strips those delimiter tokens if they appear inside a document**,
  so a crafted document can't forge the closing marker to "break out" of the data
  region.
- The defense applies to **both** generative providers (Claude and Qwen); the
  extractive (`none`) mode has no LLM and is inherently immune.
- It's **regression-tested** in `tests/eval/test_prompt_injection.py` — structural
  tests always run (CI-safe, no LLM), and an opt-in behavioural test (runs when a
  provider is available) asserts the model doesn't obey an injected document, doesn't
  leak its prompt, and still answers from the legitimate content.

**Output rendering is injection-safe.** The widget inserts the answer and source
names via `textContent` (HTML-escaped), and the Streamlit app renders the answer as
plain markdown — `unsafe_allow_html` is used only for the app's **own** figure/table
HTML built from local files, never for model output. So an injected answer can't
inject active markup into the page.

> **Not yet addressed (tracked as wishlist #2).** The Streamlit app and the widget
> API are currently **open and unauthenticated**, with open CORS and no request-size
> limits. Add auth, an origin allowlist, rate limiting, and input caps before any
> public or multi-user deployment — especially for internal/confidential documents.

---

## Validation

| Action | Expected |
|---|---|
| Ask about PDF content | A single **streamed, synthesized** answer with inline `[file p.N]` citations |
| Ask something not in the docs | The assistant says it doesn't know (grounded, no hallucination); a **⚠️ Low confidence** banner appears; **no** source downloads / figures / follow-ups |
| Tick **Include figures**, ask about a cited page with figures | Page image(s) render inline below the answer |
| Tick **Include tables**, ask about a cited page with a table | Table renders as HTML below the answer |
| Query a value that lives inside a table | The table's page is retrieved (content-level search) |
| Check under an answer | A **⬇️ Download source** button per cited document |
| Click **Prepare PDF of chat** | Downloadable PDF transcript incl. images/tables |

Service health:

```bash
curl -s -o /dev/null -w "app:%{http_code}\n" http://localhost:8501/
```

---

## Evaluation harness

An offline, maintainer-facing harness (`tests/eval/`) measures retrieval quality so
you can catch regressions and tune settings with evidence. It is **embedding-only —
no LLM calls, no API cost** — and drives the same functions the chatbot uses.

```bash
.venv/bin/python -m tests.eval.run            # human-readable report + pass/fail
.venv/bin/python -m tests.eval.run --json report.json
```

It reports:

- **Retrieval recall@k** — for each ground-truth question, is the expected page in
  the top-k retrieved chunks? Exits non-zero if recall drops below `recall_floor`
  (CI-gating), so retrieval regressions are caught before users see them.
- **Confidence calibration** — the score distributions for in-corpus vs off-topic
  questions and a **suggested `low_confidence_threshold`** sitting in the gap between
  them. (The current **0.63** default was set this way: in-corpus min ≈0.65,
  off-topic max ≈0.61.)

The reference set lives in `tests/eval/dataset.json` (plain JSON, human-asserted
question→page pairs). It **adapts to a changing store**: a question whose document
isn't currently embedded is **skipped** (reported, not failed), so you keep questions
for documents that come and go. Add a document → add its question(s); remove one →
its questions auto-skip. See `tests/eval/README.md` for details.

> With a small corpus the recall number is **directional, not statistically strong**;
> it becomes more meaningful as documents and questions grow. The calibration half is
> useful immediately for setting the banner threshold.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Images not rendering | No image extracted for the cited page, or figures toggle off | Confirm `pdf/<doc>/<page>/*.png` exists; tick **Include figures**; re-embed if missing |
| Answer stops mid-sentence | Hit the token cap on a very long answer | A truncation note is appended; ask a more specific question |
| `ANTHROPIC_API_KEY` error | `.env` missing or key unset | Add the key to `.env` |
| Anthropic 404 / model not found | `ANTHROPIC_MODEL` unavailable to your account | Set a valid Claude model id in `.env` |
| `applymap` AttributeError | Newer pandas | Use `DataFrame.map` (already applied) |
| Empty answers / dim mismatch | Store built with a different embedding model | `rm -rf chroma_Name` and rebuild |
| Garbled table values | PDF table extraction imperfect on complex layouts | Verify against the source via the page number; `lines_strict` strategy reduces this |
| Content skipped | `SKIP_PAGES` too high | Lower it in `.env` (default `0`) |
| Just-uploaded doc "not found" / "I don't have any information" | A **different** process ingested it while a reader was running; stale per-process Chroma index | Fixed: the chatbot/API reload the index per query (`_open_vectorstore()`), so new docs are found with no restart. If you see this on an **older** build, restart the reader service |
| Chatbot ignores a provider change | Provider is read **once per session** | Refresh the chatbot browser tab to start a new session |
| Selected Qwen but answers are extractive | GGUF missing or `llama-cpp-python` not installed → auto-fallback | Download the GGUF to `models/` (or set `QWEN_GGUF_PATH`); `pip install -r requirements.txt` |
| Qwen answers very slow | 1.5B model on CPU | Expected (~1–2 min/answer); use Claude for speed, or run on a GPU build of `llama-cpp-python` |
| Qwen answers lack `[file p.N]` citations | Small-model limitation | Expected; source **download buttons** still appear. Use Claude for strict inline citations |

---

## Teardown

```bash
# Stop services with Ctrl+C
rm -rf chroma_Name table                 # generated artifacts (destructive)
# pdf/ holds BOTH source PDFs and extracted images — review before deleting
deactivate 2>/dev/null; rm -rf .venv
```

---

## Quick start (TL;DR)

```bash
cd doc-intelligence-rag
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
cp .env.example .env            # add ANTHROPIC_API_KEY
mkdir -p pdf && cp /path/to/*.pdf pdf/
python embeddingEngine.py       # build store + extract images/tables

# Run the app
streamlit run chatBOT_engine.py
# → http://localhost:8501

# Optional: manage documents
streamlit run embeddingGUI.py --server.port 8502
```

---

## License

Released under the [MIT License](LICENSE) — free to use, modify, and distribute
with attribution. © 2026 Milind Agarwal.
