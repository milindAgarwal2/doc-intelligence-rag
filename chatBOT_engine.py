import base64
import glob
import html as _html
import io
import os
import re
from datetime import datetime

import anthropic
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from config import load_config, VALID_PROVIDERS

# Reads ANTHROPIC_API_KEY from the environment (.env) - never hardcode it here.
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")

# Local Qwen SLM GGUF path (used only when provider == "qwen"). Override via env.
QWEN_GGUF_PATH = os.getenv(
    "QWEN_GGUF_PATH",
    os.path.join("models", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
)


# ---------------------------------------------------------------------------
# Retrieval / embedding engine (PRESERVED — unchanged behaviour)
# ---------------------------------------------------------------------------
def getdirname(filepath):
    # Strip directory + extension so the asset folder name matches ingestion
    # (foo.pdf and foo.pptx both -> "foo"), keeping assets resolvable per format.
    base = os.path.basename(filepath)
    base = os.path.splitext(base)[0]
    base = base.replace(" ", "")
    return base


def retrieve_docs(user_question):
    """Vector-search the Chroma store for chunks relevant to the question."""
    embedding_function = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma(
        "Project_NAME",
        persist_directory="./chroma_Name",
        embedding_function=embedding_function,
        collection_metadata={"hnsw:space": "cosine"},
    )
    # MMR (max-marginal-relevance) retrieval: pull a wider candidate set
    # (fetch_k) then select k results that are both relevant AND diverse. This
    # improves recall of a defining chunk that pure top-k similarity can miss,
    # and reduces near-duplicate / low-value chunks (e.g. reference lists)
    # crowding out substantive content. lambda_mult balances relevance (1.0)
    # vs. diversity (0.0).
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 24, "lambda_mult": 0.6},
    )
    return retriever.invoke(user_question)


# ---------------------------------------------------------------------------
# Asset rendering (PRESERVED — image/table reconstruction from page metadata)
# ---------------------------------------------------------------------------
def _image_to_data_uri(filepath, max_dim=1200, jpeg_quality=80):
    """Return a compact base64 data: URI for an image, downscaling if large.

    Large images are resized/recompressed so the inlined markup stays small;
    inlining multi-MB images overwhelms Streamlit's renderer. JPEG is used unless
    the image has transparency (then PNG). Returns "" on any error.
    """
    try:
        from PIL import Image

        with Image.open(filepath) as img:
            has_alpha = img.mode in ("RGBA", "LA") or (
                img.mode == "P" and "transparency" in img.info
            )
            w, h = img.size
            if max(w, h) > max_dim:
                scale = max_dim / float(max(w, h))
                img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))

            buf = io.BytesIO()
            if has_alpha:
                img.save(buf, format="PNG", optimize=True)
                mime = "image/png"
            else:
                img.convert("RGB").save(
                    buf, format="JPEG", quality=jpeg_quality, optimize=True
                )
                mime = "image/jpeg"
            b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:{mime};base64,{b64}"
    except Exception as ex:
        print("Could not process image", filepath, ex)
        return ""


def appendImage(responsetoquestion, newfilename, newpageNumber):
    """Append all images for a (file, page) as inline base64 <img> tags."""
    imageDir = getdirname(newfilename)
    pathToStoreImage = os.path.join("pdf", imageDir, str(newpageNumber))
    for filepath in glob.glob(os.path.join(pathToStoreImage, "*.png")):
        data_uri = _image_to_data_uri(filepath)
        if not data_uri:
            continue
        responsetoquestion += (
            f"<img src='{data_uri}' style='max-width:100%;height:auto;'><br>"
        )
    return responsetoquestion


def _render_table_html(df):
    """Render a DataFrame as a clean, self-contained inline-styled HTML table."""
    def fmt(cell):
        text = "" if cell is None else str(cell)
        return _html.escape(text).replace("\n", "<br>")

    th_style = (
        "border:1px solid #2e7d32;padding:6px 10px;background:#2e7d32;"
        "color:#fff;text-align:left;vertical-align:top;font-weight:600;"
    )
    td_style = (
        "border:1px solid #2e7d32;padding:6px 10px;text-align:left;"
        "vertical-align:top;color:#111;background:#fff;"
    )
    table_style = (
        "border-collapse:collapse;border:2px solid #2e7d32;margin:8px 0;"
        "font-size:0.85rem;width:100%;"
    )

    rows = df.values.tolist()
    if not rows:
        return ""
    header, data = rows[0], rows[1:]
    thead = "<tr>" + "".join(f"<th style='{th_style}'>{fmt(c)}</th>" for c in header) + "</tr>"
    tbody = ""
    for row in data:
        tbody += "<tr>" + "".join(f"<td style='{td_style}'>{fmt(c)}</td>" for c in row) + "</tr>"
    return (
        f"<table style='{table_style}'>"
        f"<thead>{thead}</thead><tbody>{tbody}</tbody></table>"
    )


def appendTable(responsetoquestion, newfilename, newpageNumber):
    """Append all tables for a (file, page) as inline HTML tables."""
    tableDir = getdirname(newfilename)
    pathToStoreTable = os.path.join("table", tableDir, str(newpageNumber))
    for filepath in glob.glob(os.path.join(pathToStoreTable, "*.pkl")):
        df = pd.read_pickle(filepath)
        responsetoquestion += _render_table_html(df)
    return responsetoquestion


def _resolve_source_path(filename):
    """Resolve a stored source file_name to an on-disk path, or None."""
    candidates = [filename, os.path.join("pdf", os.path.basename(filename))]
    return next((p for p in candidates if os.path.isfile(p)), None)


# ---------------------------------------------------------------------------
# Synthesized, streamed answer with inline citations (NEW default behaviour)
# ---------------------------------------------------------------------------
def _build_context(docs):
    """Build a context block for the LLM and the citation metadata.

    Each chunk is labelled with its source + page so the model can cite inline as
    ``[<file> p.<N>]``. Returns (context_text, pages) where pages is an ordered,
    de-duplicated list of (file_name, pagenumber) tuples that were offered.
    """
    parts = []
    pages = []
    seen = set()
    for d in docs:
        fname = os.path.basename(d.metadata.get("file_name", "unknown"))
        page = d.metadata.get("pagenumber", "?")
        key = (d.metadata.get("file_name"), page)
        if key not in seen:
            seen.add(key)
            pages.append((d.metadata.get("file_name"), page))
        parts.append(f"[Source: {fname} p.{page}]\n{d.page_content}")
    return "\n\n".join(parts), pages


_SYNTH_SYSTEM = (
    "You are a precise technical documentation assistant. Answer the user's "
    "question using ONLY the provided context from their documents. "
    "Cite the specific source and page inline for each claim using the exact "
    "format [<filename> p.<page>] drawn from the context's [Source: ...] markers. "
    "If the context does not contain the answer, say you don't know based on the "
    "available documents — do not invent facts. Be clear and concise. "
    "Do NOT add disclaimers about being unable to show images, figures, tables, "
    "or diagrams: the application renders any available figures and tables from "
    "the cited pages directly below your answer. When a figure or table is "
    "referenced, describe its content from the text and cite its page; the app "
    "handles the visual rendering."
)

# Small models (Qwen 1.5B) need a stronger, example-driven nudge to reliably emit
# the [filename p.N] citation markers that Claude produces naturally. Appended
# ONLY for the Qwen path (Claude follows the base prompt fine).
_QWEN_CITATION_REINFORCE = (
    "\n\nCITATION RULES (follow exactly):\n"
    "- After EVERY sentence or bullet point, append the source marker in square "
    "brackets copied verbatim from the matching [Source: <filename> p.<page>] line "
    "in the context, written as [<filename> p.<page>].\n"
    "- Every bullet must end with at least one such marker.\n"
    "Example of the required style:\n"
    "  The board uses the RP2040 chip [datasheet.pdf p.3].\n"
    "  - It has 2 MB of flash memory [datasheet.pdf p.3].\n"
)


def stream_synthesized_answer(client, user_question, docs):
    """Yield the synthesized answer token-by-token using Claude streaming.

    The caller is responsible for retrieval; this just streams the LLM response.
    Yields text deltas; on error yields a single error string so the UI never
    hangs or blanks.
    """
    context, _pages = _build_context(docs)
    system_prompt = f"{_SYNTH_SYSTEM}\n\nContext:\n{context}"
    try:
        with client.messages.stream(
            model=ANTHROPIC_MODEL,
            # Generous budget so full answers aren't cut off mid-sentence.
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_question}],
        ) as stream:
            for text in stream.text_stream:
                yield text
            # If the model hit the token ceiling, the last sentence is likely
            # truncated. Surface this clearly instead of leaving a dangling
            # fragment that looks like a bug.
            try:
                final = stream.get_final_message()
                if getattr(final, "stop_reason", None) == "max_tokens":
                    yield "\n\n_…(response truncated — ask a more specific "
                    yield "question or request the rest to continue)_"
            except Exception:
                pass
    except Exception as ex:  # pragma: no cover - network/credential errors
        yield f"\n\n_Error generating answer: {type(ex).__name__}: {ex}_"


# ---------------------------------------------------------------------------
# Provider abstraction: claude | qwen (local SLM) | none (extractive)
# ---------------------------------------------------------------------------
def provider_available(name):
    """Return True if the given provider can actually run right now.

    - claude:  an ANTHROPIC_API_KEY is set
    - qwen:    llama-cpp-python importable AND the GGUF model file exists
    - none:    always available (extractive needs no generation)
    """
    if name == "claude":
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    if name == "qwen":
        if not os.path.isfile(QWEN_GGUF_PATH):
            return False
        try:
            import llama_cpp  # noqa: F401
            return True
        except ImportError:
            return False
    if name == "none":
        return True
    return False


def resolve_provider(requested):
    """Resolve the effective provider with auto-fallback so the app never breaks.

    If the requested provider is unavailable, fall back to extractive ("none").
    Returns the effective provider string. Logs a server-side note on fallback so
    it stays transparent to the chat user.
    """
    if requested not in VALID_PROVIDERS:
        requested = "none"
    if provider_available(requested):
        return requested
    if requested != "none":
        print(f"[provider] '{requested}' unavailable; falling back to extractive.")
    return "none"


@st.cache_resource(show_spinner="Loading local Qwen model…")
def _qwen_llm():
    """Load the Qwen GGUF once per process (cached). Returns a llama_cpp.Llama."""
    from llama_cpp import Llama

    return Llama(
        model_path=QWEN_GGUF_PATH,
        n_ctx=4096,
        n_threads=max(1, (os.cpu_count() or 2) - 1),
        verbose=False,
        chat_format="qwen",
    )


def stream_qwen_answer(user_question, docs):
    """Stream a synthesized answer from the local Qwen SLM.

    Uses the shared grounding prompt PLUS an explicit, example-driven citation
    reinforcement (small models otherwise tend to drop the [file p.N] markers).
    Yields text deltas; on any error, yields an error string.
    """
    context, _pages = _build_context(docs)
    system_prompt = f"{_SYNTH_SYSTEM}{_QWEN_CITATION_REINFORCE}\n\nContext:\n{context}"
    # Restating the citation requirement next to the question helps a small model
    # keep it in mind while generating.
    user_msg = (
        f"{user_question}\n\n"
        "Remember: end every sentence and bullet with its [<filename> p.<page>] "
        "citation copied from the context's [Source: ...] markers."
    )
    try:
        llm = _qwen_llm()
        stream = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=1024,
            temperature=0.2,
            stream=True,
        )
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            piece = delta.get("content")
            if piece:
                yield piece
    except Exception as ex:  # pragma: no cover
        yield f"\n\n_Error generating answer (Qwen): {type(ex).__name__}: {ex}_"


def render_extractive(docs):
    """Build an extractive answer: the retrieved passages verbatim, each under a
    [<file> p.<N>] source header. No LLM/generation involved. Returns the full
    answer text (markdown). Citations in the text drive the same asset/download
    scoping used for the generative providers.
    """
    if not docs:
        return "I couldn't find anything relevant in the indexed documents."
    lines = [
        "_No answer-generation model is configured, so here are the most "
        "relevant passages from your documents:_\n"
    ]
    seen = set()
    for d in docs:
        fname = os.path.basename(d.metadata.get("file_name", "unknown"))
        page = d.metadata.get("pagenumber", "?")
        key = (fname, page, d.page_content[:60])
        if key in seen:
            continue
        seen.add(key)
        passage = d.page_content.strip()
        lines.append(f"**[{fname} p.{page}]**\n\n{passage}\n")
    return "\n".join(lines)


def _cited_pages(answer_text, offered_pages):
    """Return the subset of offered (file, page) tuples actually cited in the
    answer text, matched by the [<file> p.<N>] markers the model emitted.

    Falls back to all offered pages if no explicit citation is detected, so
    relevant assets are still shown.
    """
    cited = []
    for fname, page in offered_pages:
        base = os.path.basename(fname) if fname else ""
        # Match "[<base> p.<page>]" allowing minor spacing variations.
        pattern = re.escape(base) + r"\s*p\.?\s*" + re.escape(str(page))
        if base and re.search(pattern, answer_text, flags=re.IGNORECASE):
            cited.append((fname, page))
    return cited or offered_pages


def render_assets_html(answer_text, pages, include_image, include_table):
    """Build an HTML fragment of images/tables for the given pages.

    ``pages`` is the list of (file_name, pagenumber) the answer cited (filtered by
    the caller), so only assets the answer actually drew on are shown. Images and
    tables are de-duplicated per page and labelled with their source. Returns ""
    when there is nothing to show (either no asset-bearing page or both toggles
    off).
    """
    if not (include_image or include_table):
        return ""
    html_out = ""
    seen_img = set()
    seen_tbl = set()
    for fname, page in pages:
        label = f"{os.path.basename(fname)} p.{page}"
        if include_image and (fname, page) not in seen_img:
            seen_img.add((fname, page))
            imgs = appendImage("", fname, page)
            if imgs:
                html_out += (
                    f"<p style='color:#2e7d32;font-size:0.8rem;margin:8px 0 2px;'>"
                    f"Figure(s) from {label}</p>" + imgs
                )
        if include_table and (fname, page) not in seen_tbl:
            seen_tbl.add((fname, page))
            tbls = appendTable("", fname, page)
            if tbls:
                html_out += (
                    f"<p style='color:#2e7d32;font-size:0.8rem;margin:8px 0 2px;'>"
                    f"Table(s) from {label}</p>" + tbls
                )
    return html_out


# ---------------------------------------------------------------------------
# Source downloads + chat-to-PDF export (PRESERVED, adapted to turn model)
# ---------------------------------------------------------------------------
def _render_source_downloads(sources, turn_id):
    """Render a download button for each cited source PDF (bytes served natively)."""
    for i, filename in enumerate(sources):
        path = _resolve_source_path(filename)
        label = os.path.basename(filename)
        if not path:
            st.caption(f"Source unavailable on disk: {label}")
            continue
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            st.caption(f"Could not read source: {label}")
            continue
        st.download_button(
            label=f"⬇️ Download source: {label}",
            data=data,
            file_name=label,
            mime="application/pdf",
            key=f"dl_{turn_id}_{i}",
        )


def _pdf_link_callback(uri, rel):
    """Resolve URIs in chat HTML to local paths for xhtml2pdf (data: pass through)."""
    if uri.startswith("data:"):
        return uri
    project_root = os.path.abspath(os.path.dirname(__file__))
    local = os.path.join(project_root, uri)
    return local if os.path.isfile(local) else uri


def chat_history_to_pdf(messages):
    """Render the chat turns (questions + answers + assets) to a PDF (bytes)."""
    from xhtml2pdf import pisa

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections = []
    for m in messages:
        if m["role"] == "user":
            sections.append(
                f"<div class='qa'><p class='q'><b>Q:</b> {_html.escape(m['content'])}</p>"
            )
        else:
            answer_html = m["content"] + m.get("assets_html", "")
            sections.append(
                f"<div class='a'><b>A:</b><br/>{answer_html}</div></div>"
            )
    body = "".join(sections) if sections else "<p>No conversation yet.</p>"

    html_doc = f"""
    <html><head><style>
        @page {{ size: a4 portrait; margin: 1.5cm; }}
        body {{ font-family: Helvetica, Arial, sans-serif; font-size: 10pt; color: #111; }}
        h1 {{ font-size: 16pt; color: #2e7d32; }}
        .meta {{ color: #666; font-size: 8pt; margin-bottom: 12px; }}
        .qa {{ border-bottom: 1px solid #ccc; padding: 8px 0; }}
        .q {{ background: #eef3ee; padding: 6px 8px; border-radius: 4px; }}
        .a {{ padding: 4px 8px; }}
        table {{ border-collapse: collapse; width: 100%; font-size: 8pt; }}
        th, td {{ border: 1px solid #2e7d32; padding: 4px 6px; }}
        th {{ background: #2e7d32; color: #fff; }}
        img {{ max-width: 480px; }}
    </style></head><body>
        <h1>Chat Transcript</h1><p class='meta'>Generated: {generated}</p>{body}
    </body></html>
    """
    buffer = io.BytesIO()
    result = pisa.CreatePDF(src=html_doc, dest=buffer, link_callback=_pdf_link_callback)
    return None if result.err else buffer.getvalue()


# ---------------------------------------------------------------------------
# Core answer flow (retrieve → stream synthesized answer → render assets)
# ---------------------------------------------------------------------------
def answer_question(user_question, include_image, include_table, provider):
    """Run one Q&A turn and append it to st.session_state.messages.

    ``provider`` is the effective backend ("claude" | "qwen" | "none"). Claude and
    Qwen stream a synthesized answer; "none" renders retrieved passages verbatim
    (extractive). Then cited images/tables and source downloads are rendered.
    Returns the assistant turn dict. The provider is transparent to the chat user.
    """
    with st.spinner("Searching your documents…"):
        docs = retrieve_docs(user_question)

    if not docs:
        answer_text = "I couldn't find anything relevant in the indexed documents."
        offered_pages = []
    else:
        _, offered_pages = _build_context(docs)

        if provider == "none":
            # Extractive: no generation, show the passages verbatim.
            answer_text = render_extractive(docs)
            st.markdown(answer_text)
        else:
            # Generative (claude or qwen): stream into the live chat bubble.
            placeholder = st.empty()
            acc = []
            if provider == "qwen":
                gen = stream_qwen_answer(user_question, docs)
            else:  # claude
                client = anthropic.Anthropic(timeout=60.0, max_retries=2)
                gen = stream_synthesized_answer(client, user_question, docs)
            for delta in gen:
                acc.append(delta)
                placeholder.markdown("".join(acc))
            answer_text = "".join(acc)
            placeholder.markdown(answer_text)

    # Scope downloads AND rendered assets to what the answer actually cited, so
    # a cross-document chunk that retrieval pulled in but the answer never used
    # doesn't show a stray download button or figure. _cited_pages falls back to
    # all retrieved pages when no citation is detected (true for extractive).
    cited_pages = _cited_pages(answer_text, offered_pages) if offered_pages else []

    sources = []
    for fname, _page in cited_pages:
        if fname and fname not in sources:
            sources.append(fname)

    # Build asset HTML for the cited pages per the toggles.
    assets_html = render_assets_html(
        answer_text, cited_pages, include_image, include_table
    )
    if assets_html:
        st.markdown(assets_html, unsafe_allow_html=True)

    turn = {
        "role": "assistant",
        "content": answer_text,
        "assets_html": assets_html,
        "sources": sources,
    }
    return turn


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def _init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of turn dicts
    # Resolve the answer provider ONCE per session from config.json (written by
    # the upload GUI). Auto-fallback to extractive if it's not actually usable.
    # This is transparent to the chat user — there is no provider control here.
    if "provider" not in st.session_state:
        requested = load_config().get("llm_provider", "none")
        st.session_state.provider = resolve_provider(requested)


def _render_turn(turn, idx):
    """Render one chat turn using Streamlit chat primitives."""
    if turn["role"] == "user":
        with st.chat_message("user"):
            st.markdown(turn["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(turn["content"])
            if turn.get("assets_html"):
                st.markdown(turn["assets_html"], unsafe_allow_html=True)
            if turn.get("sources"):
                _render_source_downloads(turn["sources"], idx)


def main():
    load_dotenv()
    st.set_page_config(page_title="Document Chat", page_icon="📚", layout="wide")
    _init_state()

    st.title("📚 Technical Document Assistant")
    st.caption(
        "Ask questions about your indexed PDFs. Answers are synthesized from the "
        "documents with inline source citations."
    )

    # One-time heads-up when the local Qwen SLM is the active engine: it runs on
    # CPU and is noticeably slower than the cloud model, so set expectations.
    if st.session_state.get("provider") == "qwen":
        st.warning(
            "⏳ This session uses the local **Qwen** model (offline, on CPU). "
            "Answers are slower — expect roughly **1–2 minutes per answer**, and a "
            "longer wait on the first question while the model loads. The answer "
            "streams in as it's generated."
        )

    # Sidebar: answer options + export + reset.
    with st.sidebar:
        st.subheader("Answer options")
        include_image = st.checkbox("Include figures from cited pages", value=False)
        include_table = st.checkbox("Include tables from cited pages", value=False)

        st.divider()
        st.subheader("Conversation")
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.session_state.messages:
            if st.button("Prepare PDF of chat", use_container_width=True):
                with st.spinner("Building PDF…"):
                    st.session_state.pdf_bytes = chat_history_to_pdf(
                        st.session_state.messages
                    )
            pdf_bytes = st.session_state.get("pdf_bytes")
            if pdf_bytes:
                st.download_button(
                    "⬇️ Download chat as PDF",
                    data=pdf_bytes,
                    file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

    # Replay existing conversation.
    for idx, turn in enumerate(st.session_state.messages):
        _render_turn(turn, idx)

    # Chat input (bottom-docked) — this is the default entry point.
    prompt = st.chat_input("Ask a question about your documents…")
    if prompt:
        # Show the user's message immediately.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate + stream the assistant answer inside its bubble.
        with st.chat_message("assistant"):
            turn = answer_question(
                prompt, include_image, include_table, st.session_state.provider
            )
            if turn.get("sources"):
                _render_source_downloads(turn["sources"], len(st.session_state.messages))
        st.session_state.messages.append(turn)


if __name__ == "__main__":
    main()
