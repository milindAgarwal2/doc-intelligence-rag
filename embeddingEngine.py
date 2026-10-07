import gc
import glob
import os

from dotenv import load_dotenv

from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document

from spire.pdf import *
import pymupdf4llm
import fitz
import pandas as pd


def get_pdf_text(pdf_docs):
    text = ""
    for page in pdf_docs.pages:
        text += page.extract_text()
    return text

def getdirname(filepath):
    # Strip the directory and the file extension so the asset folder name is
    # stable across formats (foo.pdf and foo.pptx both -> "foo"). The chatbot
    # computes the same name, so extracted assets resolve regardless of format.
    base = os.path.basename(filepath)
    base = os.path.splitext(base)[0]
    base = base.replace(" ", "")
    return base


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        chunk_size=500,  #This means thousand characters
        chunk_overlap=200,
        length_function=len,
        separator="\n"
    )
    chunks = text_splitter.split_text(text)
    return chunks


def clean_table_df(df):
    """Tidy a raw extracted table DataFrame for display and embedding.

    - replaces None/NaN with empty strings
    - collapses stray newlines inside cells
    - drops rows and columns that are entirely empty
    """
    import pandas as _pd

    if df is None or df.empty:
        return df

    # Normalize every cell to a clean string.
    # Normalize every cell to a clean string. Keep it MINIMAL: only handle
    # None/NaN and collapse literal newlines. Do NOT re-split on whitespace,
    # which would mangle already-clean values (e.g. decimals, "@5V (mA)").
    def _norm(v):
        if v is None:
            return ""
        s = str(v)
        if s.strip().lower() == "nan":
            return ""
        # Collapse newlines; join a digit/decimal split like "135\n." -> "135."
        s = s.replace("\n", "").strip()
        return s

    # DataFrame.map replaced the removed .applymap in newer pandas.
    try:
        df = df.map(_norm)
    except AttributeError:
        df = df.applymap(_norm)
    # Drop fully-empty columns, then fully-empty rows.
    df = df.loc[:, ~(df == "").all(axis=0)]
    df = df.loc[~(df == "").all(axis=1)]
    df = df.reset_index(drop=True)
    return df


def table_to_text(df):
    """Convert an extracted table DataFrame to a flat text representation that is
    useful for semantic embedding (cell values matter more than exact layout).

    The first extracted row is treated as the header. Produces lines like:
    "<header1>: <value1> | <header2>: <value2>" so queries matching cell values
    retrieve the table's page.
    """
    rows = df.values.tolist()
    if not rows:
        return ""

    header = [("" if c is None else str(c).replace("\n", " ").strip()) for c in rows[0]]
    lines = ["Table columns: " + " | ".join(h for h in header if h)]

    for row in rows[1:]:
        cells = []
        for idx, val in enumerate(row):
            text = "" if val is None else str(val).replace("\n", " ").strip()
            if not text:
                continue
            label = header[idx] if idx < len(header) and header[idx] else f"col{idx}"
            cells.append(f"{label}: {text}")
        if cells:
            lines.append(" | ".join(cells))

    return "\n".join(lines)


def storetableembedding(table_text, vectorstore, file, pagenumber):
    """Embed a single table's text content as a searchable chunk.

    The chunk is tagged tablePresent=1 so the retriever's table-rendering path
    (which reads the pickled DataFrame for display) still fires on a hit.
    """
    if not table_text or not table_text.strip():
        return
    document = Document(
        page_content=table_text,
        metadata={
            "file_name": file,
            "pagenumber": pagenumber,
            "imagePresent": 0,
            "tablePresent": 1,
            "contentType": "table",
        },
    )
    vectorstore.add_documents(documents=[document])


def storetextembedding(text_chunks, vectorstore, file, pagenumber, hasImage, hasTable):

    loadingdocuments =  []

    for text in text_chunks:
        document_1 = Document(page_content=text, metadata={"file_name": file, "pagenumber": pagenumber, "imagePresent": hasImage, "tablePresent":hasTable})
        loadingdocuments.append(document_1)

    if len(loadingdocuments) != 0:
        vectorstore.add_documents(documents=loadingdocuments)

    return file

def get_vectorstore():
    """Return the shared Chroma vector store (same config everywhere)."""
    embedding_function = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return Chroma(
        "Project_NAME",
        embedding_function,
        "./chroma_Name",
        collection_metadata={"hnsw:space": "cosine"},
    )


def ingest_pdf(filepath, vectorstore, skip_pages=0):
    """Embed a single PDF into the vector store and extract its tables/images.

    This does NOT reset the collection, so it can be used to incrementally add
    new documents. Returns a dict summary: pages processed, tables, images.
    Safe to call repeatedly; existing extracted assets are overwritten.
    """
    gc.collect()

    summary = {"pages": 0, "tables": 0, "images": 0}

    doc = PdfDocument()
    doc.LoadFromFile(filepath)
    tableDoc = fitz.open(filepath)
    imageDir = getdirname(filepath)
    image_helper = PdfImageHelper()
    output = pymupdf4llm.to_markdown(filepath, page_chunks=True)

    for i in range(skip_pages, doc.Pages.Count):
        summary["pages"] += 1

        hasTable = 0
        tablePage = tableDoc[i]
        # "lines_strict" uses ruled lines for cell boundaries and gives far
        # cleaner results on bordered spec tables (e.g. decimals stay intact).
        # Fall back to the default heuristic if it finds nothing.
        try:
            tables = tablePage.find_tables(strategy="lines_strict")
            if not tables.tables:
                tables = tablePage.find_tables()
        except Exception:
            tables = tablePage.find_tables()
        pathToStoreTable = os.path.join("table", imageDir, str(i))
        j = 1
        for table in tables:
            hasTable = 1
            if not os.path.exists(pathToStoreTable):
                os.makedirs(pathToStoreTable)
            pickleTableName = os.path.join(pathToStoreTable, "table" + str(j) + ".pkl")
            df = clean_table_df(pd.DataFrame(table.extract()))
            df.to_pickle(pickleTableName)
            # Also embed the table's cell values as searchable text so queries
            # matching table content retrieve this page.
            storetableembedding(table_to_text(df), vectorstore, filepath, i)
            summary["tables"] += 1
            j = j + 1

        images_info = image_helper.GetImagesInfo(doc.Pages[i])
        hasImage = 0
        for j in range(len(images_info)):
            hasImage = 1
            pathToStoreImage = os.path.join("pdf", imageDir, str(i))
            if not os.path.exists(pathToStoreImage):
                os.makedirs(pathToStoreImage)
            image_info = images_info[j]
            output_file = os.path.join(pathToStoreImage, "image" + str(j) + ".png")
            image_info.Image.Save(output_file)
            summary["images"] += 1

        text = output[i].get("text")
        text_chunks = get_text_chunks(text)
        storetextembedding(text_chunks, vectorstore, filepath, i, hasImage, hasTable)

    doc.Close()
    tableDoc.close()
    return summary


def _pptx_slide_text(slide):
    """Collect all readable text from a slide's shapes (titles, bullets, text
    boxes). Table text is handled separately (as structured tables), so it is
    skipped here to avoid duplicating it into the plain-text chunks.
    """
    parts = []
    for shape in slide.shapes:
        if getattr(shape, "has_table", False):
            continue
        if shape.has_text_frame:
            for para in shape.text_frame.paragraphs:
                line = "".join(run.text for run in para.runs).strip()
                if line:
                    parts.append(line)
    # Include speaker notes if present — often rich, query-relevant prose.
    try:
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                parts.append("Notes: " + notes)
    except Exception:
        pass
    return "\n".join(parts)


def _pptx_table_to_df(table):
    """Build a DataFrame from a python-pptx table (first row treated as header
    downstream, same as the PDF path)."""
    rows = []
    for row in table.rows:
        rows.append([cell.text for cell in row.cells])
    return clean_table_df(pd.DataFrame(rows))


def ingest_pptx(filepath, vectorstore, skip_pages=0):
    """Embed a single PPTX into the vector store and extract its tables/images.

    Mirrors ingest_pdf exactly, mapping SLIDE INDEX -> pagenumber so the chatbot's
    retrieval, citation, and asset-rendering code works unchanged. Picture shapes
    are saved to pdf/<doc>/<slide>/imageN.png and table shapes to
    table/<doc>/<slide>/tableN.pkl, matching the PDF on-disk layout.

    Does NOT reset the collection (safe for incremental add). Returns a dict
    summary: slides processed, tables, images.
    """
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    gc.collect()
    summary = {"pages": 0, "tables": 0, "images": 0}

    prs = Presentation(filepath)
    imageDir = getdirname(filepath)

    for i, slide in enumerate(prs.slides):
        if i < skip_pages:
            continue
        summary["pages"] += 1

        hasTable = 0
        hasImage = 0

        # Tables: pickle each + embed its cell values as a searchable chunk.
        t_idx = 1
        for shape in slide.shapes:
            if not getattr(shape, "has_table", False):
                continue
            df = _pptx_table_to_df(shape.table)
            if df is None or df.empty:
                continue
            hasTable = 1
            pathToStoreTable = os.path.join("table", imageDir, str(i))
            if not os.path.exists(pathToStoreTable):
                os.makedirs(pathToStoreTable)
            df.to_pickle(os.path.join(pathToStoreTable, "table" + str(t_idx) + ".pkl"))
            storetableembedding(table_to_text(df), vectorstore, filepath, i)
            summary["tables"] += 1
            t_idx += 1

        # Pictures: save each picture shape's raw image bytes as a PNG.
        img_idx = 0
        for shape in slide.shapes:
            if shape.shape_type != MSO_SHAPE_TYPE.PICTURE:
                continue
            try:
                image = shape.image
                blob = image.blob
            except Exception:
                continue
            hasImage = 1
            pathToStoreImage = os.path.join("pdf", imageDir, str(i))
            if not os.path.exists(pathToStoreImage):
                os.makedirs(pathToStoreImage)
            # Normalize to PNG so the chatbot's glob("*.png") finds it regardless
            # of the embedded format (jpeg/emf/etc.).
            out_png = os.path.join(pathToStoreImage, "image" + str(img_idx) + ".png")
            try:
                from PIL import Image as _PILImage
                import io as _io
                _PILImage.open(_io.BytesIO(blob)).convert("RGBA").save(out_png, "PNG")
            except Exception:
                # Fallback: write raw bytes (works when the blob is already PNG).
                with open(out_png, "wb") as f:
                    f.write(blob)
            summary["images"] += 1
            img_idx += 1

        # Slide text (titles, bullets, text boxes, notes).
        text = _pptx_slide_text(slide)
        text_chunks = get_text_chunks(text) if text.strip() else []
        storetextembedding(text_chunks, vectorstore, filepath, i, hasImage, hasTable)

    return summary


def ingest_file(filepath, vectorstore, skip_pages=0):
    """Dispatch to the right ingester based on file extension.

    Supports PDF (ingest_pdf) and PPTX/PPT (ingest_pptx). Returns the ingester's
    summary dict, or raises ValueError for an unsupported extension.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return ingest_pdf(filepath, vectorstore, skip_pages=skip_pages)
    if ext in (".pptx", ".ppt"):
        return ingest_pptx(filepath, vectorstore, skip_pages=skip_pages)
    raise ValueError(f"Unsupported file type for ingestion: {ext}")


def list_embedded_documents(vectorstore):
    """Return a sorted list of distinct file_name values already embedded."""
    try:
        data = vectorstore.get(include=["metadatas"])
    except Exception:
        return []
    names = set()
    for md in data.get("metadatas", []) or []:
        if md and md.get("file_name"):
            names.add(md["file_name"])
    return sorted(names)


def delete_document(filepath, vectorstore):
    """Remove a document from the system: its embedded chunks and extracted assets.

    Deletes:
      - all Chroma chunks whose metadata file_name == filepath
      - extracted tables under table/<doc>/
      - extracted images under pdf/<doc>/
      - the source file itself (PDF or PPTX)

    Returns a dict summary of what was removed.
    """
    import shutil

    summary = {"chunks_deleted": 0, "assets_removed": [], "source_removed": False}

    # 1) Delete embedded chunks by metadata filter.
    try:
        before = vectorstore._collection.count()
        vectorstore._collection.delete(where={"file_name": filepath})
        after = vectorstore._collection.count()
        summary["chunks_deleted"] = max(0, before - after)
    except Exception as ex:
        summary["error"] = f"chunk delete failed: {ex}"

    docdir = getdirname(filepath)

    # 2) Remove extracted tables and images directories.
    for base in ("table", "pdf"):
        assetdir = os.path.join(base, docdir)
        if os.path.isdir(assetdir):
            shutil.rmtree(assetdir, ignore_errors=True)
            summary["assets_removed"].append(assetdir)

    # 3) Remove the source PDF file.
    if os.path.isfile(filepath):
        try:
            os.remove(filepath)
            summary["source_removed"] = True
        except OSError:
            pass

    return summary


def main():
    load_dotenv()

    # Number of leading pages to skip per PDF (e.g. cover/table-of-contents).
    # Default 0 = process every page. Override via the SKIP_PAGES env var or .env.
    try:
        skip_pages = int(os.getenv("SKIP_PAGES", "0"))
    except ValueError:
        skip_pages = 0
    if skip_pages < 0:
        skip_pages = 0

    vectorstore = get_vectorstore()
    # Batch mode resets the collection for a clean full rebuild.
    vectorstore.reset_collection()

    # Source documents: PDFs in pdf/, PPTX decks in pptx/.
    source_files = (
        glob.glob(os.path.join("pdf", "*.pdf"))
        + glob.glob(os.path.join("pptx", "*.pptx"))
        + glob.glob(os.path.join("pptx", "*.ppt"))
    )
    for filepath in source_files:
        print(filepath)
        try:
            ingest_file(filepath, vectorstore, skip_pages=skip_pages)
        except ValueError as ex:
            print("  skipped:", ex)


if __name__ == '__main__':
    main()