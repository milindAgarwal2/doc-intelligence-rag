# 📚 Technical-Document RAG Chatbot (with Tables & Images)

Query technical PDFs with an LLM — and get back not just text, but the **images**
and **tables** from the source pages too, with **source attribution** (document
name + page number) for explainable (XAI) results.

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

## How it works (two parts)

- **Part 1 — Ingestion (`embeddingEngine.py`):** for each PDF page, extract text,
  tables, and images. Embed the text (and now the table content too) into a Chroma
  vector store with metadata; save tables and images to disk keyed by document and
  page number.
- **Part 2 — Chatbot (`chatBOT_engine.py`):** a Streamlit app that runs vector search
  over the text, summarizes hits with Claude, and — based on user toggles —
  reconstructs and renders the images/tables from the matched pages, with a source
  link and page markers.

## Toolset

- **Embeddings:** local & offline via [`fastembed`](https://github.com/qdrant/fastembed),
  model `BAAI/bge-small-en-v1.5` (ONNX, **no PyTorch**)
- **LLM:** Anthropic Claude (per-chunk summarization + "Ask Claude directly")
- **Vector store:** Chroma — collection `Project_NAME`, persisted at `./chroma_Name`
- **PDF processing:** `pymupdf4llm` (text), `pymupdf`/`fitz` (tables), `Spire.PDF` (images)
- **UI:** Streamlit (+ HTML/CSS)
- **Chat-to-PDF export:** `xhtml2pdf`

---

## Core strategy: "page-as-anchor" sidecar storage

The join key between embedded text and non-text assets is **`file_name` + `pagenumber`**.

| Asset  | Extracted with            | Stored as                               |
|--------|---------------------------|-----------------------------------------|
| Text   | `pymupdf4llm` (markdown)  | embedding in Chroma                     |
| Tables | `pymupdf` `find_tables`   | pickled DataFrame `table/<doc>/<page>/tableN.pkl` |
| Images | `Spire.PDF` ImageHelper   | PNG `pdf/<doc>/<page>/imageN.png`       |

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
| `embeddingEngine.py` | Ingestion: extract text/tables/images + embed. Reusable functions `get_vectorstore`, `ingest_pdf`, `list_embedded_documents`, `delete_document`. |
| `chatBOT_engine.py` | Streamlit chatbot (retrieval + Claude + rendering + chat-to-PDF). |
| `embeddingGUI.py` | Streamlit GUI to view / upload / delete embedded documents. |
| `htmlTemplates.py` | CSS + chat message HTML templates. |
| `requirements.txt` | Python dependencies. |
| `.env` | Holds `ANTHROPIC_API_KEY` (git-ignored). |

Generated at runtime: `chroma_Name/` (vector store), `table/<doc>/<page>/*.pkl`,
`pdf/<doc>/<page>/*.png`.

---

## Prerequisites

| Requirement | Details |
|---|---|
| Python | 3.10 – 3.12 |
| Anthropic API key | With access to the configured Claude model |
| Internet | pip installs, one-time ONNX model download (~130 MB), Claude API calls |

> **Note on the Anthropic SDK:** this project targets `anthropic` **1.11.0**, whose
> `messages.create()` does **not** accept a `temperature` argument. The code therefore
> omits `temperature` (Claude uses its server-side default). Do not re-add it.

---

## Installation

```bash
cd ChatBOT
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
```

Optional branding assets in the project root (inlined into the page as base64 at
load time, if present): `AI.png`, `users.png`, `ericssonlogo.png`. Missing assets
are skipped gracefully — no broken image icons.

---

## Building the vector store

Put PDFs in `pdf/`, then run the batch ingester (this **resets** the collection for a
clean full rebuild):

```bash
mkdir -p pdf && cp /path/to/*.pdf pdf/
python embeddingEngine.py
```

This creates `chroma_Name/`, extracts tables to `table/<doc>/<page>/` and images to
`pdf/<doc>/<page>/`, embeds page text, and embeds each table's content as a searchable
chunk. First run downloads the `bge-small-en-v1.5` ONNX model.

For **incremental** add/remove without a full rebuild, use the Embedding Manager GUI
(below) or the `ingest_pdf` / `delete_document` functions.

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

---

## Embedding Manager GUI

A separate Streamlit app to manage the document library without the CLI:

```bash
streamlit run embeddingGUI.py --server.port 8502
```

Features:
- Shows the number of embedded documents and total chunks, and lists each document.
- **Upload a PDF** → saved to `pdf/` → embedded **incrementally** (no collection reset)
  → shows a pages/tables/images summary.
- **Delete a document** → removes its Chroma chunks, its `table/<doc>/` and `pdf/<doc>/`
  assets, and the source PDF. Requires a confirmation checkbox (destructive).

> Re-embedding a PDF with the **same filename** appends duplicate chunks (there is no
> delete-before-readd). Delete it first for a clean re-embed. The GUI warns you.

---

## Using the chatbot

- Type a question and press Enter.
- Tick **Include Image** / **Include Tables** to render page figures/tables in the answer.
- Click a past question in the **sidebar** to jump straight to that Q&A (no scrolling);
  **Show all** returns to the full thread.
- **Ask Claude directly** returns one concise synthesized answer instead of per-chunk
  summaries.
- Append `RAW` to a question to get unsummarized source text.
- **Prepare PDF of chat** (sidebar) builds a downloadable PDF of the whole conversation,
  including the images and tables.

### Behaviour notes

- Images and tables are **de-duplicated per page**: if several retrieved chunks come
  from the same page, each image/table renders only once.
- Tables render as styled HTML (header row + bordered cells). Rendering quality depends
  on extraction quality — some complex tables may extract imperfectly; use the source
  link + page number to verify against the original.

---

## Validation

| Action | Expected |
|---|---|
| Ask about PDF content | Per-chunk Claude summaries with a **From: `<file>`** link and **Page Number** markers |
| Tick **Include Image**, ask about a page with figures | Page image(s) render inline |
| Tick **Include Tables**, ask about a page with a table | Table renders as HTML |
| Query a value that lives inside a table | The table's page is retrieved (content-level search) |
| **Ask Claude directly** | Concise synthesized answer |
| Append `RAW` | Raw (unsummarized) chunk text |

Service health:

```bash
curl -s -o /dev/null -w "app:%{http_code}\n" http://localhost:8501/
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Images not rendering | Image never extracted for that page, or asset deleted | Confirm `pdf/<doc>/<page>/*.png` exists; re-embed the PDF if missing |
| `ANTHROPIC_API_KEY` error | `.env` missing or key unset | Add the key to `.env` |
| `temperature` TypeError | `temperature` re-added to a Claude call | Remove it — SDK 1.11.0 doesn't support it |
| `get_relevant_documents` AttributeError | Old LangChain API | Use `retriever.invoke(...)` (already applied) |
| `applymap` AttributeError | Newer pandas | Use `DataFrame.map` (already applied) |
| Empty answers / dim mismatch | Store built with a different embedding model | `rm -rf chroma_Name` and rebuild |
| Garbled table values | PDF table extraction imperfect on complex layouts | Verify against the source via the page link; `lines_strict` strategy is used to reduce this |
| Content skipped | `SKIP_PAGES` too high | Lower it in `.env` (default `0`) |

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
cd ChatBOT
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
