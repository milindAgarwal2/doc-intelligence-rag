import base64
import glob
from pathlib import Path

from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
import pandas as pd
import requests
import streamlit as st
from PyPDF2 import PdfReader
from attr import validators
from dotenv import load_dotenv
from transformers import pipeline

from htmlTemplates import css, bot_template, user_template, eli_bot_template
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# Use the API Key as is (Reminder - do not hard-code in public notebooks/code)
ELI_API_KEY = "REDACTED_API_KEY"

# Set ELI Gateway URL for the EWS instance
ELI_API_URL = "https://gateway.eli.gaia.gic.ericsson.se"

imageAgree = False
showTable = False


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


def checkindex(response, checkres):
    index = response.find(checkres)
    if index != -1:
        st.write(
            bot_template.replace("{{MSG}}", "Do you want to connect to Ericsson AI for more information ? (Yes / No) "),
            unsafe_allow_html=True)

def getdirname(filepath):
    imageDir = filepath.replace("pdf\\", "")
    imageDir = imageDir.replace(".pdf", "")
    imageDir = imageDir.replace(" ", "")
    return imageDir


def getitem(newresp, checkitem):
    for i, airesponse in enumerate(newresp):
        index = airesponse.find(checkitem)

def displaychathistory():
    reversechat = dict(reversed(list(st.session_state.chat_history.items())))
    for i, (k, v) in enumerate(reversechat.items()):
        st.write(user_template.replace("{{MSG}}", k), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", v), unsafe_allow_html=True)

def appendImage(responsetoquestion,newfilename,newpageNumber):

    imageDir = getdirname(newfilename)
    pathToStoreImage = f"pdf\\" + imageDir + "\\" + str(newpageNumber)

    imageNames = list(glob.glob(pathToStoreImage+"\\*.png"))

    for filepath in imageNames:
        print("Image file path",filepath)
        file_url = f"http://localhost:8000/{filepath}"
        print("File URL",file_url)
        imageTag = f"<img src='{file_url}'><br>"
        responsetoquestion = responsetoquestion + imageTag

    return responsetoquestion

def appendTable(responsetoquestion,newfilename,newpageNumber):

    tableDir = getdirname(newfilename)
    pathToStoreTable = f"table\\" + tableDir + "\\" + str(newpageNumber)

    imageNames = list(glob.glob(pathToStoreTable+"\\*.pkl"))

    for filepath in imageNames:
        print("Table file path",filepath)
        df = pd.read_pickle(filepath)
        df = df.iloc[1:]
        #df.drop(columns=df.columns[0], axis=1, inplace=True)
        #print("File URL",file_url)
        #imageTag = f"<img src='{file_url}'><br>"
        #responsetoquestion = responsetoquestion + df.to_html(index=False, header=False)
        df.style.set_table_styles([{'selector': '', 'props': [('border', '2px solid green')]}])
        responsetoquestion = responsetoquestion + df.to_html(index=False, header=False, border=5)

    return responsetoquestion

def handle_userinput_getdocuments(user_question):
    embedding_function = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    vectorstore = Chroma("Project_MCPTT", persist_directory="./chroma_mcptt",
                         embedding_function=embedding_function, collection_metadata={"hnsw:space": "cosine"})
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Retrieve and generate using the relevant snippets of the documents
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(user_question)

    responsetoquestion = ""
    filename = ""
    pageNumber = ""

    for text in docs:
        newfilename = text.metadata['file_name']
        newpageNumber = text.metadata['pagenumber']
        imagePresent = text.metadata['imagePresent']
        tablePresent = text.metadata['tablePresent']
        if filename != newfilename:
           filename = newfilename
           print("Filename:",filename)
           file_url = f"http://localhost:8000/{filename}"
           responsetoquestion += f"From: <a href='{file_url}' download>{filename}</a> | "

        if pageNumber != newpageNumber:
           pageNumber = newpageNumber
           responsetoquestion += f"<p style='color:Tomato;'>Page Number: {pageNumber} </p>"

        if "RAW" not in user_question:
            summary = summarizer(text.page_content, min_length=5, max_length=300)
            responsetoquestion = responsetoquestion + summary[0]["summary_text"] + "\n <br>"
        else:
            responsetoquestion = responsetoquestion + text.page_content + "\n <br>"

        if imagePresent == 1 and st.session_state['include_image']:
            responsetoquestion = appendImage(responsetoquestion,newfilename,newpageNumber)

        if tablePresent == 1 and st.session_state['include_table']:
            responsetoquestion = appendTable(responsetoquestion, newfilename, newpageNumber)

    if st.session_state['include_image']:
        user_question = user_question + "_IMG"

    if st.session_state['include_table']:
        user_question = user_question + "_TAB"

    st.session_state.chat_history[user_question] = responsetoquestion
    displaychathistory()


def askeli(user_question):
    print("Button pressed")
    print("clicked state", st.session_state.clicked)

    # reversechat = list(st.session_state.chat_history.items())
    # k,v = reversechat[-1]
    # print("Question :", k)
    # print("Answer :", v)

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n"
    )

    result = eli_chat_stream(
        model="mistral",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question},
        ],
        temperature=0.3,
        max_new_tokens=512,
    )

    st.session_state.chat_history[user_question + "_ELI"] = result

def text_input_handler():
    st.session_state['text_input'] = st.session_state['text_input_widget']

def click_button():
    st.session_state.clicked = True
    print("Click status", st.session_state.clicked)

def click_image():
    st.session_state.clicked = False
    print("Click status", st.session_state['include_image'])

def click_table():
    st.session_state.clicked = False
    print("Click status", st.session_state['include_table'])


def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with project solution",
                       page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if 'clicked' not in st.session_state:
        st.session_state.clicked = False

    user_question = st.text_input("Ask a question about your project solution:",key='text_input_widget', on_change=text_input_handler)

    st.button("To ask ELI click here", on_click=click_button)
    st.checkbox("Include Image",key="include_image",on_change=click_image)
    st.checkbox("Include Tables", key="include_table", on_change=click_table)

    if st.session_state.clicked:
       # The message and nested widget will remain on the page
       st.session_state.clicked = False
       askeli(user_question)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}

    if user_question:
        if user_question.lower() == "yes":
            displaychathistory()
            askeli(user_question)
        else:
            #handle_userinput(user_question)
            handle_userinput_getdocuments(user_question)

    st.write(user_template.replace(
        "{{MSG}}", "hello bot"), unsafe_allow_html=True)

    st.write(bot_template.replace(
        "{{MSG}}", "hello human"), unsafe_allow_html=True)


def eli_chat_stream(
        messages=None,
        model: str = "mistral",
        max_new_tokens=128,
        temperature=0.3
):
    """
    Make API call to ELI LLM Chat endpoint

    The input messages field follows the format:
        [
            {"role": "system", "content" : "<message>"},
            {"role": "user", "content" : "<message>"}
            {"role": "assistant", "content" : "<message>"},
        ]
    """

    if messages is None:
        messages = []
    try:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stream": True,  # Set to false if you do not want streamed response
        }
        headers = {
            "Authorization": f"Bearer {ELI_API_KEY}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            f"{ELI_API_URL}/api/v1/llm/chat_stream",
            json=payload,
            headers=headers,
            timeout=(300, 300),
            verify=False,
        )
        output = []
        print("Prepare to print response")
        for part in response.iter_content():
            part = part.decode("utf8")
            output.append(part)
            print(part, end="", flush=True)  # <- print stream content to console
        print()
        return "".join(output)
    except Exception as ex:
        return f"An error occurred: {ex}"


if __name__ == '__main__':
    main()