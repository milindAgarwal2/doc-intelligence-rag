"""Streamlit GUI to manage the document embeddings.

Features
--------
- Shows how many documents are already embedded and lists them.
- Lets you upload a new PDF; it is saved to ./pdf and embedded incrementally
  (tables -> table/<doc>/<page>/, images -> pdf/<doc>/<page>/, text -> Chroma).
- Does NOT reset the collection, so existing embeddings are preserved.

Run with:
    streamlit run embeddingGUI.py
"""

import logging
import os

import streamlit as st
from dotenv import load_dotenv

from embeddingEngine import (
    get_vectorstore,
    ingest_file,
    list_embedded_documents,
    delete_document,
)
from config import load_config, save_config, VALID_PROVIDERS, VALID_LOG_LEVELS
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger("embeddingGUI")

PDF_DIR = "pdf"
PPTX_DIR = "pptx"

# Human-readable labels for the provider radio.
_PROVIDER_LABELS = {
    "claude": "Claude (Anthropic API) — best quality, needs ANTHROPIC_API_KEY",
    "qwen": "Qwen 1.5B (local, offline) — needs the GGUF model file",
    "none": "None (extractive) — show source passages, no generation",
}


def _safe_hex(value, fallback="#2e7d32"):
    """st.color_picker needs a #hex value. If the configured color is a named
    color or rgb()/hsl(), fall back to a hex default for the picker (the actual
    config value is still whatever the user set)."""
    import re
    if isinstance(value, str) and re.match(r"^#[0-9a-fA-F]{6}$", value.strip()):
        return value.strip()
    return fallback


def _dest_dir_for(filename):
    """Return the source folder for an uploaded file based on its extension."""
    ext = os.path.splitext(filename)[1].lower()
    return PPTX_DIR if ext in (".pptx", ".ppt") else PDF_DIR


@st.cache_resource(show_spinner="Loading vector store…")
def _load_vectorstore():
    """Load the Chroma store once per session (cached across reruns)."""
    return get_vectorstore()


def _skip_pages():
    try:
        n = int(os.getenv("SKIP_PAGES", "0"))
    except ValueError:
        n = 0
    return max(0, n)


def _count_chunks(vectorstore):
    try:
        return vectorstore._collection.count()
    except Exception:
        return None


def _tab_documents(vectorstore, docs, chunk_count):
    """Documents tab: metrics, list, upload, delete."""
    c1, c2 = st.columns(2)
    c1.metric("Embedded documents", len(docs))
    c2.metric("Total text chunks", chunk_count if chunk_count is not None else "—")
    st.divider()

    st.subheader("Embedded documents")
    if docs:
        rows = [{"Document": os.path.basename(d), "Stored path": d} for d in docs]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No documents embedded yet. Upload a PDF or PPTX below to get started.")

    st.divider()

    col_add, col_del = st.columns(2)

    # ---- Add ----
    with col_add:
        st.markdown("#### ➕ Add a document")
        uploaded = st.file_uploader(
            "Upload a PDF or PPTX to embed",
            type=["pdf", "pptx", "ppt"],
            accept_multiple_files=False,
        )
        if uploaded is not None:
            dest_dir = _dest_dir_for(uploaded.name)
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, uploaded.name)
            if dest in docs:
                st.info(
                    f"'{uploaded.name}' is already embedded. Re-embedding is a safe "
                    "**upsert**: identical content is skipped; changed content "
                    "replaces the old chunks (no duplicates)."
                )
            if st.button(f"Embed '{uploaded.name}'", type="primary"):
                with open(dest, "wb") as f:
                    f.write(uploaded.getbuffer())
                try:
                    with st.spinner(f"Embedding '{uploaded.name}'…"):
                        summary = ingest_file(dest, vectorstore, skip_pages=_skip_pages())
                    logger.info(
                        "GUI embed '%s' | pages=%s tables=%s images=%s skipped=%s",
                        uploaded.name, summary.get("pages"), summary.get("tables"),
                        summary.get("images"), summary.get("skipped"),
                    )
                    if summary.get("skipped"):
                        st.info(f"'{uploaded.name}' already embedded with identical "
                                "content — nothing to do.")
                    else:
                        st.success(
                            f"Embedded '{uploaded.name}': {summary['pages']} pages/slides, "
                            f"{summary['tables']} tables, {summary['images']} images."
                        )
                    st.rerun()
                except Exception as ex:
                    logger.exception("failed to embed '%s'", uploaded.name)
                    st.error(f"Failed to embed '{uploaded.name}': {type(ex).__name__}: {ex}")

    # ---- Delete ----
    with col_del:
        st.markdown("#### 🗑️ Delete a document")
        if not docs:
            st.caption("No documents to delete.")
        else:
            name_to_path = {os.path.basename(d): d for d in docs}
            choice = st.selectbox("Select a document", list(name_to_path.keys()))
            target_path = name_to_path[choice]
            st.warning(
                f"Permanently removes **{choice}**: its chunks, extracted "
                f"tables/images, and the source file. Cannot be undone."
            )
            confirm = st.checkbox(f"Yes, delete '{choice}' and all its data")
            if st.button("Delete document", disabled=not confirm):
                try:
                    with st.spinner(f"Deleting '{choice}'…"):
                        result = delete_document(target_path, vectorstore)
                    msg = f"Deleted '{choice}': {result['chunks_deleted']} chunks removed"
                    if result.get("source_removed"):
                        msg += ", source file removed"
                    st.success(msg)
                    if result.get("error"):
                        st.error(result["error"])
                    st.rerun()
                except Exception as ex:
                    logger.exception("failed to delete '%s'", choice)
                    st.error(f"Failed to delete '{choice}': {type(ex).__name__}: {ex}")


def _tab_engine(cfg):
    """Answer-engine + low-confidence settings."""
    st.subheader("Answer engine")
    st.caption(
        "Which backend the chatbot/widget uses to answer. One at a time; if the "
        "selected engine isn't available at runtime it falls back to extractive."
    )
    current = cfg.get("llm_provider", "none")
    options = list(VALID_PROVIDERS)
    choice = st.radio(
        "Model",
        options,
        index=options.index(current) if current in options else options.index("none"),
        format_func=lambda p: _PROVIDER_LABELS.get(p, p),
    )
    if choice != current:
        save_config({"llm_provider": choice})
        st.success(f"Answer engine set to: {choice}. New chat sessions will use it.")

    st.divider()
    st.subheader("Low-confidence banner")
    st.caption(
        "Advisory banner shown above an answer when the best retrieval relevance "
        "is below this floor. Never blocks an answer. 0.00 disables it."
    )
    current_thr = float(cfg.get("low_confidence_threshold", 0.63))
    new_thr = st.slider(
        "Relevance threshold", 0.0, 1.0, value=current_thr, step=0.05,
    )
    if abs(new_thr - current_thr) > 1e-9:
        save_config({"low_confidence_threshold": new_thr})
        st.success("Low-confidence banner disabled." if new_thr == 0.0
                   else f"Threshold set to {new_thr:.2f}.")


def _tab_widget(cfg):
    """Embeddable widget appearance/content settings."""
    st.subheader("Chat widget")
    st.caption(
        "Appearance and greeting for the embeddable widget. Saved to config.json; "
        "the widget reads them when it loads (no redeploy needed)."
    )
    with st.form("widget_form"):
        new_welcome = st.text_area(
            "Welcome message",
            value=cfg.get("widget_welcome_message", ""),
            height=80,
            help="Shown when a visitor opens the widget. Blank = no greeting.",
        )
        c1, c2 = st.columns(2)
        with c1:
            new_title = st.text_input(
                "Title", value=cfg.get("widget_title", "📚 Document Assistant"))
        with c2:
            new_ph = st.text_input(
                "Input placeholder",
                value=cfg.get("widget_input_placeholder", "Ask about the documents…"))
        new_color = st.color_picker(
            "Accent color", value=_safe_hex(cfg.get("widget_accent_color", "#2e7d32")))
        submitted = st.form_submit_button("Save widget settings", type="primary")

    if submitted:
        save_config({
            "widget_welcome_message": new_welcome,
            "widget_title": new_title,
            "widget_input_placeholder": new_ph,
            "widget_accent_color": new_color,
        })
        st.success("Widget settings saved. Reload the widget to see changes.")


def _tab_logging(cfg):
    """Logging settings."""
    st.subheader("Logging")
    st.caption("Application log verbosity and optional file. Saved to config.json.")
    _levels = list(VALID_LOG_LEVELS)
    current_level = cfg.get("log_level", "INFO")
    current_file = cfg.get("log_file", "")
    with st.form("logging_form"):
        new_level = st.selectbox(
            "Log level", _levels,
            index=_levels.index(current_level) if current_level in _levels else _levels.index("INFO"),
        )
        new_file = st.text_input(
            "Log file (optional)", value=current_file,
            placeholder="e.g. logs/app.log — blank = console only",
            help="In Docker use a path under logs/ so it lands on the host at "
                 "./logs/…  Takes effect on the next app restart.",
        )
        submitted = st.form_submit_button("Save logging settings", type="primary")
    if submitted and (new_level != current_level or new_file != current_file):
        save_config({"log_level": new_level, "log_file": new_file})
        st.success(f"Logging set to {new_level}"
                   + (f", file '{new_file}'" if new_file else " (console only)") + ".")


def main():
    load_dotenv()
    st.set_page_config(page_title="Document Assistant Admin",
                       page_icon="📚", layout="wide")
    # Hide Streamlit's top-right toolbar (Deploy button + dev menu).
    st.markdown(
        "<style>[data-testid='stToolbar']{display:none!important;}"
        "[data-testid='stDeployButton']{display:none!important;}"
        "#MainMenu{visibility:hidden;}</style>",
        unsafe_allow_html=True,
    )
    st.title("📚 Document Assistant — Admin Console")
    st.caption(
        "Manage documents and configure the answer engine, chat widget, and "
        "logging. Settings are saved to config.json and used by the chatbot and "
        "the embeddable widget."
    )

    cfg = load_config()
    vectorstore = _load_vectorstore()
    docs = list_embedded_documents(vectorstore)
    chunk_count = _count_chunks(vectorstore)

    tab_docs, tab_engine, tab_widget, tab_logs = st.tabs(
        ["📚 Documents", "⚙️ Answer engine", "💬 Chat widget", "🔧 Logging"]
    )
    with tab_docs:
        _tab_documents(vectorstore, docs, chunk_count)
    with tab_engine:
        _tab_engine(cfg)
    with tab_widget:
        _tab_widget(cfg)
    with tab_logs:
        _tab_logging(cfg)


if __name__ == "__main__":
    main()
