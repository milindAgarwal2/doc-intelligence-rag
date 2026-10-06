import base64
import glob
import io
import os
from datetime import datetime
from pathlib import Path

import anthropic
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
import pandas as pd
import requests
import streamlit as st
from PyPDF2 import PdfReader
from attr import validators
from dotenv import load_dotenv

from htmlTemplates import css, bot_template, user_template

load_dotenv()

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

imageAgree = False
showTable = False

# Reads ANTHROPIC_API_KEY from the environment (.env) - never hardcode it here.
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")


# Chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def add_logo(logo_url: str, height: int = 120):
    """Add a logo (from logo_url) on the top of the navigation page of a multipage app.
    Taken from [the Streamlit forum](https://discuss.streamlit.io/t/put-logo-and-title-above-on-top-of-page-navigation-in-sidebar-of-multipage-app/28213/6)
    The url can either be a url to the image, or a local path to the image.

    Args:
        logo_url (str): URL/local path of the logo
    """

    if validators.url(logo_url) is True:
        logo = f"url({logo_url})"
    else:
        logo = f"url(data:image/png;base64,{base64.b64encode(Path(logo_url).read_bytes()).decode()})"

    st.markdown(
        f"""
        <style>
            [data-testid="stSidebarNav"] {{
                background-image: {logo};
                background-repeat: no-repeat;
                padding-top: {height - 40}px;
                background-position: 20px 20px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def logo():
    add_logo("ericssonlogo.png", height=300)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(
            pdf)  #initialize the pdf object with PdfReader function, it created pdf object that has pages
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,  #This means thousand characters
        chunk_overlap=200,
        length_function=len,
        separator="\n"
    )
    chunks = text_splitter.split_text(text)
    return chunks


def getdirname(filepath):
    imageDir = os.path.basename(filepath)
    imageDir = imageDir.replace(".pdf", "")
    imageDir = imageDir.replace(" ", "")
    return imageDir


def getitem(newresp, checkitem):
    for i, airesponse in enumerate(newresp):
        index = airesponse.find(checkitem)

def _clean_question_label(q):
    """Strip internal suffixes (asset tags / RAW) used in chat_history keys."""
    # Remove a trailing asset tag like "  ::img" / "  ::tbl" / "  ::img+tbl".
    base = q.split("  ::", 1)[0]
    return base.replace("_IMG", "").replace("_TAB", "").replace("RAW", "").strip()


def _asset_tag(include_image, include_table, raw):
    """Short human-readable tag describing which assets an answer includes."""
    parts = []
    if include_image:
        parts.append("image")
    if include_table:
        parts.append("table")
    if raw:
        parts.append("raw")
    return " + ".join(parts) if parts else "text"


def _history_key(question, include_image, include_table, raw):
    """Build a stable chat_history key that is UNIQUE per (question, toggles).

    Each distinct combination of question + Include Image/Tables + RAW becomes its
    own history entry, so the user can keep and revisit every variant instead of
    one entry being overwritten. The tag after "  ::" is stripped for display by
    _clean_question_label.
    """
    tag = _asset_tag(include_image, include_table, raw)
    return f"{question}  ::{tag}"


def _resolve_source_path(filename):
    """Resolve a stored source file_name to an on-disk path, or None.

    The stored name may be a path like "pdf/foo.pdf" or a bare "foo.pdf".
    """
    candidates = [filename, os.path.join("pdf", os.path.basename(filename))]
    return next((p for p in candidates if os.path.isfile(p)), None)


def _render_source_downloads(hist_key):
    """Render a real download button for each source PDF cited by an answer.

    Uses st.download_button (serves bytes natively) instead of inlining the PDF
    into the answer HTML, which would garble Streamlit's renderer for large files.
    """
    sources = st.session_state.get("answer_sources", {}).get(hist_key, [])
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
            key=f"dl_{hist_key}_{i}",
        )


def _variant_tag(q):
    """Return the asset-variant tag embedded in a composite history key, or ''."""
    if "  ::" in q:
        return q.split("  ::", 1)[1]
    return ""


def displaychathistory():
    history = st.session_state.chat_history
    selected = st.session_state.get("selected_question")

    def _q_label(k):
        tag = _variant_tag(k)
        base = _clean_question_label(k)
        return f"{base}  <span style='font-size:0.75rem;opacity:0.8;'>[{tag}]</span>" if tag else base

    # If a specific question was picked from the sidebar, show only that Q&A.
    if selected and selected in history:
        st.markdown(f"#### Showing selected question")
        st.markdown(user_template.replace("{{MSG}}", _q_label(selected)),
                    unsafe_allow_html=True)
        st.markdown(bot_template.replace("{{MSG}}", history[selected]),
                    unsafe_allow_html=True)
        _render_source_downloads(selected)
        return

    # Otherwise show the whole conversation, newest first.
    reversechat = dict(reversed(list(history.items())))
    for i, (k, v) in enumerate(reversechat.items()):
        st.markdown(user_template.replace("{{MSG}}", _q_label(k)),
                    unsafe_allow_html=True)
        st.markdown(bot_template.replace("{{MSG}}", v), unsafe_allow_html=True)
        _render_source_downloads(k)

def _source_link_html(filename):
    """Return a lightweight 'From:' source label for a document.

    IMPORTANT: we deliberately do NOT inline the source PDF as a base64 data: URI.
    A multi-MB PDF base64-encodes to many millions of characters, and dumping that
    inline into st.write(..., unsafe_allow_html=True) overwhelms Streamlit's
    renderer and garbles the whole answer. Images (small PNGs) are still inlined
    in appendImage; only the bulky source PDF is reduced to a plain label here.

    For a downloadable source PDF, use a separate st.download_button in the UI
    layer (it serves bytes natively without bloating the answer HTML).
    """
    label = os.path.basename(filename)
    return f"From: <b>{label}</b> | "


def appendImage(responsetoquestion,newfilename,newpageNumber):

    imageDir = getdirname(newfilename)
    pathToStoreImage = os.path.join("pdf", imageDir, str(newpageNumber))

    imageNames = list(glob.glob(os.path.join(pathToStoreImage, "*.png")))

    for filepath in imageNames:
        print("Image file path", filepath)
        # Inline the image as a base64 data URI so the browser renders it without
        # a separate HTTP fetch (no static server needed). Large images are
        # downscaled/recompressed first: inlining a multi-MB image produces
        # millions of characters of markup that overwhelm Streamlit's renderer
        # and garble the answer. Target a sane on-screen size instead.
        data_uri = _image_to_data_uri(filepath)
        if not data_uri:
            continue
        imageTag = f"<img src='{data_uri}' style='max-width:100%;height:auto;'><br>"
        responsetoquestion = responsetoquestion + imageTag

    return responsetoquestion


def _image_to_data_uri(filepath, max_dim=1200, jpeg_quality=80):
    """Return a compact base64 data: URI for an image, downscaling if large.

    Images wider/taller than ``max_dim`` are resized preserving aspect ratio.
    Encoded as JPEG (much smaller than PNG for photos/diagrams) unless the image
    has transparency, in which case PNG is kept. Returns an empty string on any
    read/decode error so a single bad image never blanks the answer.
    """
    try:
        from PIL import Image

        with Image.open(filepath) as img:
            has_alpha = img.mode in ("RGBA", "LA") or (
                img.mode == "P" and "transparency" in img.info
            )

            # Downscale if either dimension exceeds the cap.
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

def _render_table_html(df):
    """Render a pandas DataFrame as a clean, self-contained HTML table.

    Streamlit's st.write(..., unsafe_allow_html=True) strips <style> blocks, so
    all styling is applied inline per element. In-cell newlines (common in
    extracted PDF tables) are converted to <br> for readability.
    """
    import html

    def fmt(cell):
        text = "" if cell is None else str(cell)
        # escape HTML, then restore intentional line breaks as <br>
        return html.escape(text).replace("\n", "<br>")

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

    # First extracted row is the table header; the rest are data rows.
    header, data = rows[0], rows[1:]

    thead = "<tr>" + "".join(
        f"<th style='{th_style}'>{fmt(c)}</th>" for c in header
    ) + "</tr>"

    tbody = ""
    for row in data:
        tbody += "<tr>" + "".join(
            f"<td style='{td_style}'>{fmt(c)}</td>" for c in row
        ) + "</tr>"

    return (
        f"<table style='{table_style}'>"
        f"<thead>{thead}</thead><tbody>{tbody}</tbody></table>"
    )


def appendTable(responsetoquestion, newfilename, newpageNumber):

    tableDir = getdirname(newfilename)
    pathToStoreTable = os.path.join("table", tableDir, str(newpageNumber))

    tableNames = list(glob.glob(os.path.join(pathToStoreTable, "*.pkl")))

    for filepath in tableNames:
        print("Table file path", filepath)
        df = pd.read_pickle(filepath)
        responsetoquestion = responsetoquestion + _render_table_html(df)

    return responsetoquestion

def retrieve_docs(user_question):
    embedding_function = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma("Project_NAME", persist_directory="./chroma_Name",
                         embedding_function=embedding_function, collection_metadata={"hnsw:space": "cosine"})
    retriever = vectorstore.as_retriever()
    return retriever.invoke(user_question)

def summarize_with_claude(client, text):
    """Summarize a chunk of retrieved text using Claude.

    Returns a short summary string. Falls back to the original text on error
    so the chat response is never empty.
    """
    if not text or not text.strip():
        return ""

    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=300,
            system=(
                "You are a summarization assistant. Summarize the provided text "
                "concisely in a few sentences, preserving key technical details. "
                "Return only the summary with no preamble."
            ),
            messages=[{"role": "user", "content": text}],
        )
        return response.content[0].text
    except Exception as ex:
        # Catch broadly: a single chunk's failure must never blank the whole
        # response. Fall back to the raw text so the user still sees content.
        return f"[summary unavailable: {type(ex).__name__}: {ex}]\n{text}"


def handle_userinput_getdocuments(user_question):
    responsetoquestion = ""
    try:
        with st.spinner("Searching documents and summarizing…"):
            # Single Claude client reused for summarizing each retrieved chunk.
            # timeout prevents the UI from hanging indefinitely on a slow network.
            client = anthropic.Anthropic(timeout=30.0, max_retries=2)

            # Retrieve relevant chunks from the vector store
            docs = retrieve_docs(user_question)

            filename = ""
            pageNumber = ""
            # Track which pages already had their image/table rendered so we
            # don't append the same asset multiple times when several chunks
            # from the same page are retrieved.
            rendered_images = set()
            rendered_tables = set()
            # Source documents referenced by this answer, in first-seen order.
            # Used to render real st.download_button widgets below the answer.
            source_files = []

            for text in docs:
                newfilename = text.metadata['file_name']
                newpageNumber = text.metadata['pagenumber']
                imagePresent = text.metadata['imagePresent']
                tablePresent = text.metadata['tablePresent']
                page_key = (newfilename, newpageNumber)

                if filename != newfilename:
                    filename = newfilename
                    print("Filename:", filename)
                    if filename not in source_files:
                        source_files.append(filename)
                    # Lightweight text label in the answer HTML. The actual
                    # downloadable PDF is offered via a separate st.download_button
                    # (see displaychathistory) because inlining a multi-MB PDF as
                    # a base64 data: URI garbles Streamlit's renderer.
                    responsetoquestion += _source_link_html(filename)

                if pageNumber != newpageNumber:
                    pageNumber = newpageNumber
                    responsetoquestion += f"<p style='color:Tomato;'>Page Number: {pageNumber} </p>"

                if "RAW" not in user_question:
                    summary = summarize_with_claude(client, text.page_content)
                    responsetoquestion = responsetoquestion + f"<p>{summary}</p>"
                else:
                    import html as _html
                    raw = _html.escape(text.page_content)
                    responsetoquestion = responsetoquestion + (
                        f"<pre style='white-space:pre-wrap;word-wrap:break-word;'>{raw}</pre>"
                    )

                if (imagePresent == 1 and st.session_state.get('include_image', False)
                        and page_key not in rendered_images):
                    rendered_images.add(page_key)
                    responsetoquestion = appendImage(responsetoquestion, newfilename, newpageNumber)

                if (tablePresent == 1 and st.session_state.get('include_table', False)
                        and page_key not in rendered_tables):
                    rendered_tables.add(page_key)
                    responsetoquestion = appendTable(responsetoquestion, newfilename, newpageNumber)
    except Exception as ex:
        import traceback
        traceback.print_exc()
        st.error(f"Error while answering: {type(ex).__name__}: {ex}")
        if not responsetoquestion:
            responsetoquestion = f"<p style='color:red;'>Error: {ex}</p>"

    # Key history uniquely per (question + Include Image/Tables + RAW) so every
    # toggle combination is kept as its own entry instead of overwriting the
    # single question. The display label strips the tag via _clean_question_label.
    include_image = st.session_state.get("include_image", False)
    include_table = st.session_state.get("include_table", False)
    raw = "RAW" in user_question
    key = _history_key(user_question, include_image, include_table, raw)
    st.session_state.chat_history[key] = responsetoquestion
    # Remember which source PDFs this answer cited, so displaychathistory can
    # render a real download button per source.
    if "answer_sources" not in st.session_state:
        st.session_state.answer_sources = {}
    try:
        st.session_state.answer_sources[key] = source_files
    except NameError:
        st.session_state.answer_sources[key] = []
    # Focus the main window on the variant just produced.
    st.session_state.selected_question = key


def ask_llm(user_question):
    context = format_docs(retrieve_docs(user_question))

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        f"\n\nContext:\n{context}"
    )

    client = anthropic.Anthropic()
    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": user_question}],
        )
        result = response.content[0].text
    except anthropic.APIError as ex:
        result = f"An error occurred: {ex}"

    st.session_state.chat_history[user_question] = result
    st.session_state.selected_question = user_question

def text_input_handler():
    st.session_state['text_input'] = st.session_state['text_input_widget']


def select_question(question):
    """Sidebar callback: focus the main window on a specific past question."""
    st.session_state.selected_question = question


def show_all_questions():
    """Sidebar callback: clear selection and show the full conversation."""
    st.session_state.selected_question = None

def click_button():
    st.session_state.clicked = True
    print("Click status", st.session_state.clicked)

def click_image():
    st.session_state.clicked = False
    # No need to clear the rerun guard: the history key includes the toggle
    # state, so flipping this checkbox produces a new composite key and the
    # main loop generates a fresh entry for the (question + image) variant.
    print("Click status include_image =", st.session_state.get('include_image'))

def click_table():
    st.session_state.clicked = False
    print("Click status include_table =", st.session_state.get('include_table'))


def _pdf_link_callback(uri, rel):
    """Resolve URIs referenced in the chat HTML to local file paths for xhtml2pdf.

    Images and source links in answers are now inlined as base64 ``data:`` URIs,
    which xhtml2pdf renders natively — those are passed through unchanged. The
    legacy ``http://localhost:8000/<path>`` handling is kept as a safety net so
    older chat histories (built when the static server was in use) still export
    correctly by mapping the host prefix to the on-disk file.
    """
    # Base64 data URIs are self-contained; let xhtml2pdf handle them directly.
    if uri.startswith("data:"):
        return uri

    project_root = os.path.abspath(os.path.dirname(__file__))

    for prefix in ("http://localhost:8000/", "http://127.0.0.1:8000/"):
        if uri.startswith(prefix):
            local = os.path.join(project_root, uri[len(prefix):])
            return local if os.path.isfile(local) else uri

    # Already a local/relative path
    local = os.path.join(project_root, uri)
    if os.path.isfile(local):
        return local
    return uri


def chat_history_to_pdf(chat_history):
    """Render the full chat history (questions + HTML answers) to a PDF.

    Returns the PDF as bytes, or None if generation fails. Images and tables that
    appear in answers are included via the link callback.
    """
    from xhtml2pdf import pisa

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections = []
    # Preserve chronological order (oldest first) for a readable transcript.
    for question, answer in chat_history.items():
        # The question may carry _IMG/_TAB suffixes used internally; show it cleanly.
        clean_q = question.replace("_IMG", "").replace("_TAB", "").replace("RAW", "").strip()
        sections.append(
            "<div class='qa'>"
            f"<p class='q'><b>Q:</b> {clean_q}</p>"
            f"<div class='a'><b>A:</b><br/>{answer}</div>"
            "</div>"
        )

    body = "".join(sections) if sections else "<p>No conversation yet.</p>"

    html_doc = f"""
    <html>
    <head>
    <style>
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
        a {{ color: #1565c0; }}
    </style>
    </head>
    <body>
        <h1>Chat Transcript</h1>
        <p class='meta'>Generated: {generated}</p>
        {body}
    </body>
    </html>
    """

    buffer = io.BytesIO()
    result = pisa.CreatePDF(src=html_doc, dest=buffer, link_callback=_pdf_link_callback)
    if result.err:
        return None
    return buffer.getvalue()


def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with project solution",
                       page_icon=":books:")

    st.markdown(css, unsafe_allow_html=True)

    if 'clicked' not in st.session_state:
        st.session_state.clicked = False

    user_question = st.text_input("Ask a question about your project solution:",key='text_input_widget', on_change=text_input_handler)

    st.button("Concise answer", on_click=click_button)
    st.checkbox("Include Image", key="include_image", on_change=click_image)
    st.checkbox("Include Tables", key="include_table", on_change=click_table)

    if st.session_state.clicked:
       # The message and nested widget will remain on the page
       st.session_state.clicked = False
       ask_llm(user_question)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}
    if "selected_question" not in st.session_state:
        st.session_state.selected_question = None
    if "last_processed_question" not in st.session_state:
        st.session_state.last_processed_question = None

    # Process a submission once per unique (question + toggle state). Keying the
    # guard on the composite history key means toggling Include Image/Tables for
    # the same question is treated as a NEW variant and generates its own entry,
    # instead of being suppressed as a duplicate.
    if user_question:
        _ii = st.session_state.get("include_image", False)
        _it = st.session_state.get("include_table", False)
        _raw = "RAW" in user_question
        current_key = _history_key(user_question, _ii, _it, _raw)
        if current_key != st.session_state.last_processed_question:
            st.session_state.last_processed_question = current_key
            handle_userinput_getdocuments(user_question)

    # Render the main window FIRST so the answer always appears, independent of
    # anything that happens in the sidebar (PDF generation, etc.).
    if st.session_state.chat_history:
        displaychathistory()

    # Sidebar: clickable question index + PDF export. Wrapped so a sidebar
    # error can never blank the main answer area.
    try:
        with st.sidebar:
            st.subheader("Questions")
            if st.session_state.chat_history:
                # "Show all" clears the selection and shows the full conversation.
                st.button("📜 Show all", on_click=show_all_questions,
                          use_container_width=True)
                # Newest question first. Each is a clickable button that focuses
                # the main window on that single Q&A (no scrolling needed).
                questions = list(reversed(list(st.session_state.chat_history.keys())))
                for idx, q in enumerate(questions):
                    label = _clean_question_label(q)
                    label = (label[:40] + "…") if len(label) > 40 else label
                    tag = _variant_tag(q)
                    display = f"{label} [{tag}]" if tag else label
                    is_selected = (q == st.session_state.selected_question)
                    st.button(
                        ("▶ " if is_selected else "") + display,
                        key=f"q_{idx}",
                        on_click=select_question,
                        args=(q,),
                        use_container_width=True,
                    )
            else:
                st.caption("Ask a question to build your history.")

            st.divider()
            st.subheader("Export")
            if st.session_state.chat_history:
                # Lazy PDF generation: only build the PDF when the user asks,
                # so it never slows down or stalls normal question answering.
                if st.button("Prepare PDF of chat", use_container_width=True):
                    with st.spinner("Building PDF…"):
                        st.session_state.pdf_bytes = chat_history_to_pdf(
                            st.session_state.chat_history)
                pdf_bytes = st.session_state.get("pdf_bytes")
                if pdf_bytes:
                    st.download_button(
                        label="⬇️ Download chat as PDF",
                        data=pdf_bytes,
                        file_name=f"chat_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
            else:
                st.caption("Ask a question to enable PDF export.")
    except Exception as ex:
        import traceback
        traceback.print_exc()
        st.sidebar.error(f"Sidebar error: {type(ex).__name__}: {ex}")


if __name__ == '__main__':
    main()