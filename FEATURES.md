# Feature Document — Technical-Document RAG Chatbot

A capability reference for evaluating this chatbot against other document-Q&A /
RAG solutions on the market. Written for a technical buyer comparing options.

**Document version:** 2026-10-08
**Product in one line:** A self-hosted RAG chatbot that answers questions about your
technical **PDFs and PowerPoint decks** and returns not just text but the **images and
tables** from the source pages — with per-claim source attribution.

---

## 1. What makes this different (the one-paragraph pitch)

Most RAG chatbots embed **text only**. During ingestion the images (architecture
diagrams, formulas) and tables (specs, parameters) in a technical document are
discarded, so even a correct text answer loses the artifact that actually carries the
value. This product keeps them: it extracts and stores tables and images keyed by
*document + page/slide*, embeds table **content** as its own searchable unit, and at
query time reconstructs and renders those assets inline next to the answer — scoped to
the pages the answer actually **cited**. It runs **fully self-hosted**, can run
**100% offline** (local embeddings + optional local LLM), and ships both a Streamlit
app and a one-line embeddable JavaScript widget off the **same** backend.

---

## 2. Feature matrix

Legend: ✅ full · ◑ partial / configurable · ⬜ not yet (roadmap) · — n/a

### 2.1 Ingestion & content understanding

| Capability | Status | Notes |
|---|:--:|---|
| PDF ingestion | ✅ | `pymupdf4llm` text, `pymupdf` tables, `Spire.PDF` images |
| PowerPoint (PPTX/PPT) ingestion | ✅ | `python-pptx`; slides map to page numbers — treated like a PDF downstream |
| **Table extraction (structure preserved)** | ✅ | Tables saved as DataFrames, rendered as HTML — not flattened to prose |
| **Image / figure extraction** | ✅ | Page images saved to disk, rendered inline |
| **Table *content* is semantically searchable** | ✅ | Each table embedded as its own chunk (`contentType: table`) — a query matching cell values retrieves that page |
| Incremental add / delete (no full rebuild) | ✅ | Via Admin GUI or `ingest_file` / `delete_document` |
| **Content-hash upsert (no duplicate chunks)** | ✅ | SHA-256 `doc_hash` per chunk; re-embedding identical content is a no-op; changed content replaces old chunks |
| Charts / SmartArt / vector graphics | ⬜ | Not raster — not extracted; reference in text or paste as a picture |
| OCR of scanned/image-only PDFs | ⬜ | Not currently performed |

### 2.2 Retrieval & answer quality

| Capability | Status | Notes |
|---|:--:|---|
| Vector store | ✅ | Chroma, persisted locally |
| **MMR retrieval** (recall + de-dup) | ✅ | `k=8`, `fetch_k=24`, `lambda_mult=0.6` |
| **Inline per-claim citations** `[doc p.N]` | ✅ | Reliable with Claude; small local model may omit markers |
| **Grounded refusal** (no hallucination) | ✅ | Says "I don't know" when context lacks the answer; suppresses sources/figures on refusal |
| **Low-confidence banner** | ✅ | Advisory banner when best relevance < threshold (`config.json`, default 0.63, data-backed) |
| **Conversational follow-ups** (context rewrite) | ✅ | Rewrites "what about at 85°C?" into a standalone query before retrieval |
| **Suggested follow-up questions** | ✅ | Clickable, generated per answer |
| Answer streaming | ✅ | Streamed in the Streamlit app (widget is non-streaming today) |
| Attribution scoped to **cited** pages | ✅ | A chunk retrieval pulled but the answer didn't use won't show a stray asset |

### 2.3 Answer engine (pluggable LLM)

| Capability | Status | Notes |
|---|:--:|---|
| **Pluggable provider, single-select** | ✅ | Chosen in the Admin GUI, stored in `config.json` |
| Claude (Anthropic API) | ✅ | Best quality, reliable citations |
| **Local SLM — Qwen 1.5B (offline)** | ✅ | `llama-cpp-python` + GGUF; no API, no network; slow on CPU (~1–2 min/answer) |
| **None (extractive, zero-cost)** | ✅ | Shows retrieved passages verbatim — no LLM, instant, no key |
| **Auto-fallback to extractive** | ✅ | If the selected provider isn't available at runtime, never breaks |
| **100% offline operation possible** | ✅ | Local embeddings + Qwen or extractive = no outbound calls |

### 2.4 Deployment & integration

| Capability | Status | Notes |
|---|:--:|---|
| **Self-hosted** | ✅ | Runs entirely on your infrastructure |
| Single-command local run | ✅ | `streamlit run chatBOT_engine.py` |
| **Docker Compose (4 services)** | ✅ | chatbot :8501, admin GUI :8502, widget API :8000, demo site :9000 |
| Health endpoints | ✅ | `/_stcore/health` (Streamlit), `/api/health` (API) |
| **Embeddable JS widget** (one `<script>` tag) | ✅ | Floating bubble on any site; shares the same backend |
| HTTP API (FastAPI) | ✅ | `POST /api/chat`, `GET /api/health`, `GET /api/source/{name}` |
| Cross-origin embed | ✅ | Demonstrated (widget :9000 → API :8000) |
| **Local, offline embeddings** | ✅ | `fastembed` ONNX (`bge-small-en-v1.5`), no PyTorch, no cloud embedding bill |

### 2.5 Administration & configuration

| Capability | Status | Notes |
|---|:--:|---|
| **Web admin console** (tabbed) | ✅ | Documents · Answer engine · Chatbot · Chat widget · Logging |
| Upload / delete documents from the UI | ✅ | With a pages/tables/images summary; delete is confirmation-gated |
| **All user-facing text configurable** | ✅ | Chatbot + widget titles, subtitles, placeholders, labels — via `config.json`, GUI-editable |
| **Show/hide figure & table options** | ✅ | Hide them entirely for clients whose docs have no images/tables |
| Configurable widget theme (accent color) | ✅ | Single CSS variable drives the theme; validated server-side |
| **No redeploy for config changes** | ✅ | `config.json` is a mounted volume; applies live |
| Chat-to-PDF export | ✅ | Full transcript incl. images/tables |

### 2.6 Observability

| Capability | Status | Notes |
|---|:--:|---|
| **Structured per-query logging** | ✅ | Provider, retrieval breadth, best score, low-confidence/refusal flags, source count, latency |
| Configurable log level / file | ✅ | `config.json`; host-visible via mounted `logs/` volume |
| **Evaluation harness** (recall@k + calibration) | ✅ | Offline, no LLM cost; CI-gating recall floor; suggests the confidence threshold |
| Full distributed tracing / cost dashboard | ⬜ | Deferred (Langfuse/OTel) — current log line suffices at current scale |

### 2.7 Security & access control (be honest here)

| Capability | Status | Notes |
|---|:--:|---|
| Secrets isolated to `.env` | ✅ | Only `ANTHROPIC_API_KEY`; non-secret config is separate |
| Path-traversal protection on source serving | ✅ | `/api/source` serves only whitelisted extensions by basename |
| Server-side input validation (accent color etc.) | ✅ | Blocks style/script injection |
| **Indirect prompt-injection defense** | ✅ | Retrieved document content is fenced as untrusted data with an instruction-hierarchy system prompt; the model treats embedded "instructions" in documents as data to quote, not obey. Covers Claude + Qwen. Regression-tested |
| Output rendering is injection-safe (XSS) | ✅ | Widget escapes answer/source text (`textContent`); Streamlit renders the answer as plain markdown (no `unsafe_allow_html` on model output) |
| **Authentication / per-user access** | ⬜ | **Not yet — the Streamlit app and widget API are currently OPEN.** Highest-priority roadmap item |
| API key / rate limiting on the widget API | ⬜ | Roadmap; required before any public embed |
| Input-size caps / history validation on the widget API | ⬜ | Roadmap; pairs with auth |
| CORS origin allowlist | ◑ | Currently `*` (demo-friendly); must be restricted for production |

---

## 3. How it compares to common alternatives

High-level positioning against the categories a buyer typically evaluates. (General
category characteristics — verify specifics against each vendor's current docs.)

### vs. hosted SaaS doc-chat (e.g. upload-and-chat cloud tools)
- **This wins on:** data residency (fully self-hosted; can be fully offline),
  **table/figure fidelity** (most SaaS flatten to text), no per-seat/per-token SaaS
  bill, pluggable local LLM, white-labelable embeddable widget.
- **They win on:** turnkey auth/SSO, multi-tenant user management, polished dashboards,
  and zero ops — all out of the box. This product's **auth is still a roadmap item**.

### vs. framework/DIY stacks (LangChain / LlamaIndex you assemble yourself)
- **This wins on:** it's a finished, runnable product (ingestion + chat + admin GUI +
  widget + Docker + eval harness), not a toolkit. The **non-text asset handling and
  content-hash upsert are built and tested**, not left as an exercise.
- **They win on:** unlimited flexibility and a huge connector/integration ecosystem.
  This product is focused on PDF/PPTX technical docs, not every source type.

### vs. enterprise RAG platforms (large managed knowledge-assistant suites)
- **This wins on:** footprint and cost (two Streamlit processes + an API, local
  embeddings, optional local LLM — no heavy infra), transparency (open code,
  inspectable prompts/retrieval), and offline capability.
- **They win on:** enterprise auth/RBAC, connectors to SharePoint/Confluence/etc.,
  compliance certifications, SLAs, and scale/HA. This product targets a focused,
  self-hosted deployment, not a company-wide managed platform.

### vs. general LLM chat with file upload (e.g. paste a PDF into a chat UI)
- **This wins on:** persistent indexed corpus (not re-uploading per chat), **source
  attribution to exact pages**, rendered tables/figures, grounded refusal, and no
  context-window limit on corpus size.
- **They win on:** general reasoning breadth and zero setup for a one-off question.

---

## 4. Signature differentiators (lead with these in a comparison)

1. **Tables and figures survive ingestion and render inline** — the core value of a
   technical document is preserved, not discarded.
2. **Table content is semantically searchable** as its own unit, not just nearby prose.
3. **Three answer engines incl. a fully-offline local SLM and a zero-cost extractive
   mode**, hot-swappable from a GUI, with automatic safe fallback.
4. **Grounded by design** — per-claim page citations, a data-backed low-confidence
   banner, and an "I don't know" refusal instead of confident hallucination.
5. **One backend, two front-ends** — a Streamlit app *and* a one-line embeddable
   widget share the exact same retrieval/answer core.
6. **Self-hosted and config-driven** — all user-facing text/appearance tunable from a
   web admin console with no redeploy; can run with no outbound network calls.
7. **Provability built in** — an offline evaluation harness measures retrieval recall
   and calibrates the confidence threshold, CI-gating regressions.

---

## 5. Known limitations & honest gaps

State these plainly in any comparison — they're the questions a sharp evaluator asks.

- **No authentication yet.** The app and widget API are currently open; this is the
  top-priority roadmap item and **must** be addressed before any public/multi-user
  deployment, especially for confidential material.
- **Widget is text + citations only** today — no streaming and no inline
  figures/tables in the embedded widget (the Streamlit app has all three).
- **No OCR** — scanned/image-only PDFs aren't text-extracted.
- **Non-raster graphics** (charts, SmartArt, vector art) aren't extracted as images.
- **Local Qwen is slow on CPU** (~1–2 min/answer) and may omit inline citation markers
  — Claude is the choice when speed and strict citations matter.
- **Observability is lightweight** — a structured per-query log line, not full
  distributed tracing or a cost dashboard (deferred by design at current scale).

---

## 6. Roadmap (what's next, in priority order)

| # | Item | Priority |
|---|---|---|
| 2 | **Authentication / per-user access** + API key + rate limiting on the widget API | **Urgent** |
| 7 | Modern Anthropic SDK + low temperature for determinism | Medium (needs sign-off) |
| 8 | Persistent named chat sessions; Markdown/DOCX export alongside PDF | Low (polish) |
| 9 | Full tracing (Langfuse/OTel) + admin cost/usage dashboard | Deferred |

Delivered to date: content-hash upsert, low-confidence banner, evaluation harness,
Docker Compose, structured logging, embeddable widget, and full configurability +
admin-console polish.

---

## 7. Technical specs (quick reference)

| Dimension | Choice |
|---|---|
| Embeddings | `fastembed` ONNX, `BAAI/bge-small-en-v1.5` (local, offline, no PyTorch) |
| Vector store | Chroma (local persistence) |
| Retrieval | MMR — `k=8`, `fetch_k=24`, `lambda_mult=0.6` |
| LLM options | Claude (API) · Qwen 1.5B GGUF (local) · None (extractive) |
| PDF processing | `pymupdf4llm` (text), `pymupdf` (tables), `Spire.PDF` (images) |
| PPTX processing | `python-pptx` (text, native tables, raster pictures) |
| Front-ends | Streamlit chat app · embeddable vanilla-JS widget |
| API | FastAPI — `/api/chat`, `/api/health`, `/api/source/{name}` |
| Deployment | Local (venv) or Docker Compose (4 services) |
| Config | `config.json` (18 keys, GUI-editable); `.env` for the one secret |
| Python | 3.10 – 3.12 |
