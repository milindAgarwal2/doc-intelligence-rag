import gc
import glob
import os

from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
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

def storetextembedding(text_chunks, vectorstore, file, pagenumber, hasImage, hasTable):

    loadingdocuments =  []

    for text in text_chunks:
        document_1 = Document(page_content=text, metadata={"file_name": file, "pagenumber": pagenumber, "imagePresent": hasImage, "tablePresent":hasTable})
        loadingdocuments.append(document_1)

    if len(loadingdocuments) != 0:
        vectorstore.add_documents(documents=loadingdocuments)

    return file

def main():
    load_dotenv()

    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

    vectorstore = Chroma("Project_NAME", embedding_function, "./chroma_Name",collection_metadata={"hnsw:space": "cosine"})
    vectorstore.reset_collection()
    image_count = 0

    pdf_names = list(glob.glob(os.path.join("pdf", "*.pdf")))

    for filepath in pdf_names:
        print(filepath)
        gc.collect()

        # Create a PdfDocument object
        doc = PdfDocument()

        # Load a PDF document
        doc.LoadFromFile(filepath)
        tableDoc = fitz.open(filepath)
        imageDir = getdirname(filepath)
        image_helper = PdfImageHelper()
        output = pymupdf4llm.to_markdown(filepath, page_chunks=True)

        # Loop through the pages in the document
        for i in range(4,doc.Pages.Count):
            print(i)

            hasTable = 0
            tablePage = tableDoc[i]
            tables = tablePage.find_tables()
            pathToStoreTable = os.path.join("table", imageDir, str(i))
            j = 1
            for table in tables:
                # Process each table here
                hasTable = 1
                if not os.path.exists(pathToStoreTable):
                    os.makedirs(pathToStoreTable)
                pickleTableName = os.path.join(pathToStoreTable, "table" + str(j) + ".pkl")
                df = pd.DataFrame(table.extract())
                df.to_pickle(pickleTableName)
                print(table.extract())
                j = j + 1

            images_info = image_helper.GetImagesInfo(doc.Pages[i])


            # Save images to specified location with specified format extension
            hasImage = 0

            for j in range(len(images_info)):
                hasImage = 1
                pathToStoreImage = os.path.join("pdf", imageDir, str(i))
                if not os.path.exists(pathToStoreImage):
                    os.makedirs(pathToStoreImage)
                image_info = images_info[j]
                output_file = os.path.join(pathToStoreImage, "image" + str(j) + ".png")
                image_info.Image.Save(output_file)
                image_count += 1

            text = output[i].get("text")

            text_chunks = get_text_chunks(text)
            storetextembedding(text_chunks, vectorstore, filepath, i, hasImage, hasTable)


if __name__ == '__main__':
    main()