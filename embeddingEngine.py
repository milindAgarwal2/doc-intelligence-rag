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
    imageDir = os.path.basename(filepath)
    imageDir = imageDir.replace(".pdf", "")
    imageDir = imageDir.replace(" ", "")
    return imageDir


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
      - the source PDF file itself

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

    pdf_names = list(glob.glob(os.path.join("pdf", "*.pdf")))
    for filepath in pdf_names:
        print(filepath)
        ingest_pdf(filepath, vectorstore, skip_pages=skip_pages)


if __name__ == '__main__':
    main()