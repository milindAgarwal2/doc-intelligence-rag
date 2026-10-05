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
    ingest_pdf,
    list_embedded_documents,
    delete_document,
    getdirname,
)

PDF_DIR = "pdf"


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
        st.info("No documents embedded yet. Upload a PDF below to get started.")

    st.divider()

    # ---- Upload a new document ----
    st.subheader("Add a new document")
    uploaded = st.file_uploader("Upload a PDF to embed", type=["pdf"],
                                accept_multiple_files=False)

    if uploaded is not None:
        os.makedirs(PDF_DIR, exist_ok=True)
        dest = os.path.join(PDF_DIR, uploaded.name)
        already = os.path.join(PDF_DIR, uploaded.name) in docs or dest in docs

        if already:
            st.warning(
                f"'{uploaded.name}' appears to be already embedded. "
                "Embedding again will add its chunks a second time. "
                "Delete it first if you want a clean re-embed."
            )

        if st.button(f"Embed '{uploaded.name}'", type="primary"):
            # Save the uploaded file to the pdf/ folder.
            with open(dest, "wb") as f:
                f.write(uploaded.getbuffer())

            try:
                with st.spinner(f"Embedding '{uploaded.name}'… this may take a while."):
                    summary = ingest_pdf(dest, vectorstore, skip_pages=_skip_pages())
                st.success(
                    f"Embedded '{uploaded.name}': {summary['pages']} pages, "
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
            f"images (`pdf/{getdirname(target_path)}/`), and the source PDF. "
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
                    msg += ", source PDF removed"
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
