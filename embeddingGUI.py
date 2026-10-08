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

import os

import streamlit as st
from dotenv import load_dotenv

from embeddingEngine import (
    get_vectorstore,
    ingest_file,
    list_embedded_documents,
    delete_document,
    getdirname,
)
from config import load_config, save_config, VALID_PROVIDERS

PDF_DIR = "pdf"
PPTX_DIR = "pptx"

# Human-readable labels for the provider radio.
_PROVIDER_LABELS = {
    "claude": "Claude (Anthropic API) — best quality, needs ANTHROPIC_API_KEY",
    "qwen": "Qwen 1.5B (local, offline) — needs the GGUF model file",
    "none": "None (extractive) — show source passages, no generation",
}


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


def main():
    load_dotenv()
    st.set_page_config(page_title="Embedding Manager", page_icon="📥", layout="wide")
    st.title("📥 Document Embedding Manager")

    # ---- Answer provider (single-select, writes config.json) ----
    # This is an admin setting: it controls which backend the CHATBOT uses to
    # answer. Only one can be active at a time. The chat app reads this once per
    # session, transparently — chat users don't see or choose the provider.
    with st.sidebar:
        st.subheader("Answer engine (chatbot)")
        cfg = load_config()
        current = cfg.get("llm_provider", "none")
        options = list(VALID_PROVIDERS)  # ("claude", "qwen", "none")
        choice = st.radio(
            "Which model should the chatbot use to answer?",
            options,
            index=options.index(current) if current in options else options.index("none"),
            format_func=lambda p: _PROVIDER_LABELS.get(p, p),
        )
        if choice != current:
            save_config({"llm_provider": choice})
            st.success(f"Answer engine set to: {choice}. New chat sessions will use it.")
        st.caption(
            "Only one engine is active at a time. If the selected engine isn't "
            "available at runtime (missing API key or model file), the chatbot "
            "automatically falls back to extractive mode."
        )
        st.divider()

    vectorstore = _load_vectorstore()

    # ---- Status: how many documents are embedded ----
    docs = list_embedded_documents(vectorstore)
    chunk_count = _count_chunks(vectorstore)

    col1, col2 = st.columns(2)
    col1.metric("Embedded documents", len(docs))
    col2.metric("Total text chunks", chunk_count if chunk_count is not None else "—")

    st.subheader("Embedded documents")
    if docs:
        rows = []
        for d in docs:
            name = os.path.basename(d)
            rows.append({"Document": name, "Stored path": d})
        st.table(rows)
    else:
        st.info("No documents embedded yet. Upload a PDF or PPTX below to get started.")

    st.divider()

    # ---- Upload a new document ----
    st.subheader("Add a new document")
    uploaded = st.file_uploader("Upload a PDF or PPTX to embed",
                                type=["pdf", "pptx", "ppt"],
                                accept_multiple_files=False)

    if uploaded is not None:
        dest_dir = _dest_dir_for(uploaded.name)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, uploaded.name)
        already = dest in docs

        if already:
            st.info(
                f"'{uploaded.name}' is already embedded. Re-embedding is now a "
                "safe **upsert**: identical content is skipped, and changed "
                "content replaces the old chunks (no duplicates)."
            )

        if st.button(f"Embed '{uploaded.name}'", type="primary"):
            # Save the uploaded file to its source folder (pdf/ or pptx/).
            with open(dest, "wb") as f:
                f.write(uploaded.getbuffer())

            try:
                with st.spinner(f"Embedding '{uploaded.name}'… this may take a while."):
                    summary = ingest_file(dest, vectorstore, skip_pages=_skip_pages())
                if summary.get("skipped"):
                    st.info(
                        f"'{uploaded.name}' already embedded with identical content "
                        "— nothing to do (no duplicate chunks added)."
                    )
                else:
                    st.success(
                        f"Embedded '{uploaded.name}': {summary['pages']} pages/slides, "
                        f"{summary['tables']} tables, {summary['images']} images extracted."
                    )
                # Refresh the cached store/list so the new doc appears.
                st.rerun()
            except Exception as ex:
                import traceback
                traceback.print_exc()
                st.error(f"Failed to embed '{uploaded.name}': {type(ex).__name__}: {ex}")

    st.divider()

    # ---- Delete a document (destructive) ----
    st.subheader("Delete a document")
    if not docs:
        st.caption("No documents to delete.")
    else:
        # Map display name -> stored path for selection.
        name_to_path = {os.path.basename(d): d for d in docs}
        choice = st.selectbox("Select a document to delete", list(name_to_path.keys()))
        target_path = name_to_path[choice]

        st.warning(
            f"This will permanently remove **{choice}**: its embedded chunks, "
            f"its extracted tables (`table/{getdirname(target_path)}/`), its extracted "
            f"images (`pdf/{getdirname(target_path)}/`), and the source file. "
            "This cannot be undone."
        )
        confirm = st.checkbox(f"Yes, delete '{choice}' and all its data")

        if st.button("🗑️ Delete document", disabled=not confirm):
            try:
                with st.spinner(f"Deleting '{choice}'…"):
                    result = delete_document(target_path, vectorstore)
                msg = (
                    f"Deleted '{choice}': {result['chunks_deleted']} chunks removed"
                )
                if result.get("assets_removed"):
                    msg += f", assets: {', '.join(result['assets_removed'])}"
                if result.get("source_removed"):
                    msg += ", source file removed"
                st.success(msg)
                if result.get("error"):
                    st.error(result["error"])
                st.rerun()
            except Exception as ex:
                import traceback
                traceback.print_exc()
                st.error(f"Failed to delete '{choice}': {type(ex).__name__}: {ex}")


if __name__ == "__main__":
    main()
