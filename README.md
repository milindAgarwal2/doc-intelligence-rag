In a technical document there is not just text but images and tables too. These images could convey important information such as architecture diagram, formulas etc. Tables could have information such as specifications and more. In many cases these images and tables convey much more valuable information than text itself.

If we have to create a chatbot application to query these technical documents using a regular RAG approach, first we will have to we create a vector database which is nothing but embedding of chunks of text information in these documents. During this embedding process we loose the image and table information as regular embedding process simply ignores the information present here. Even if do pickup text from the table for embedding, we end up loosing the table structure. Hence when we query this vector table all we get back is the text information minus images and tables. Which many times is not useful to the end user. Also it does not instill confidence in the answer given by chatbot. What if we could also retrieve the table and images too ?? Below I am going to explain how we can do this using using regular LLM but being little smart when extracting information from technical document during embedding or vectorization process and also using little bit of ingenuity when displaying the results of the query. Since this is little bit longer process then regular RAG based chatbot application please bear with me till the end….

I have divided the below explanation in two parts :

Part 1 : on how to create embeddings on the chucks of documents while also extracting and saving tables and images separately with proper marker of page numbers and document name etc.

Part 2: This part will be focused on creating a chat bot which will query the vector database created in first part while also retrieving the relevant images and tables which were also saved while executing the first part.

Toolset used is Langchain, Chroma DB, Sentence Transformer, SpirePDF, pymupdf4llm

Part 1: Creating embedding and extracting images and tables:

Start with the following imports:

import gc
import glob
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
Define Main function which will be calling all the functions to embed chunks and also to extract tables and images.

Load environment variables
Define the vector store
Reset the vector store collection
def main():
    load_dotenv()

    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

    # Define your vectore store with the name "Project_NAME"
    vectorstore = Chroma("Project_NAME", embedding_function, "./chroma_name",collection_metadata={"hnsw:space": "cosine"})

    # I prefer to reset the collection every time I run this code, but you can impement
    # or ignore this option based on your requirement.
    # this helps in avoiding creating a duplicte embeddings
    vectorstore.reset_collection()
    
    # defining a counter for images
    image_count = 0
Section 1: Now we start loading the pdf documents from the local folder. You can also customize the code for docx or any other format as per your choice. I have taken PDF’s as its widely used. the detailed explanation is in the comments of the code.

    # pick all the pdfs from the local folder to be eembedded
    pdf_names = list(glob.glob("pdf\\*.pdf"))

    for filepath in pdf_names:
        # each pdf name will create a file path with its own name to store a
        # tables and images 
        print(filepath) 

        # clear the garbage collection
        gc.collect() 

        # Create a PdfDocument object
        doc = PdfDocument()

        # Load a PDF document
        # the below object will be used in below code to extract page 
        # level information
        doc.LoadFromFile(filepath) 
        
        #loading as this will plugin will help extarct the tables from pdf
        tableDoc = fitz.open(filepath) 

        # laoding as this will help extract the images from the pdf
        imageDir = getdirname(filepath)
        image_helper = PdfImageHelper()

        # extarct the text from the pdf for each page with page_chunks true
        output = pymupdf4llm.to_markdown(filepath, page_chunks=True)
Section 2: Below is continuation in the above for loop showing how to extract tables from each page of the PDF and storing it in the local directory. Please follow the comments in the below code.

     # Loop through the pages in the document
     # I am starting from page number 4 as initial page is just contents/tables
     # I want to ingnore those as it messes the embeddings and gives wrong info
        for i in range(4,doc.Pages.Count):

            # this is print is just for debugging and tracking the page number
            print(i)
            
            # setting a flag to just confirm if page has table
            hasTable = 0

            # this tableDoc object was defined in previous block using fitz 
            # plugin
            tablePage = tableDoc[i]

            # check if the page has any tabular information
            tables = tablePage.find_tables()

            # define the path where this table will be stored
            pathToStoreTable = f"table\\" + imageDir + "\\" + str(i)
            j = 1

            # extracting and store each table availabe on the page
            for table in tables:
                # Process each table here
                hasTable = 1

                # create the path if it does not exist already
                if not os.path.exists(pathToStoreTable):
                    os.makedirs(pathToStoreTable)

                # create a pickle file name under which the table information
                # will be stored
                pickleTableName = pathToStoreTable+"\\table"+str(j)+".pkl"

                # extract the table in dataframe format from the table.
                df = pd.DataFrame(table.extract())

                # convert the dataframe to pickle with pickleTableName created 
                # two steps above
                df.to_pickle(pickleTableName)
                
                # This print is just for the debugging purpose to print 
                # the content of the table, you can turn if off if not required
                print(table.extract())

                # move the counter
                j = j + 1
Section 3: Below is continuation in the for loop defined in section 1 showing how to extract images from each page of the PDF and storing it in the local directory. Please follow the comments in the below code.

            # extract image information using the image_helper object defined 
            # in Section 1 
            images_info = image_helper.GetImagesInfo(doc.Pages[i])

            # Save images to specified location with specified format extension
           
            # flag to check if the document page has image or not
            hasImage = 0

            for j in range(len(images_info)):
                hasImage = 1

                # define the path to store each image on the page
                # Format for directory is pdf\<PDF_FILE_NAME>\<PAGE_NUMBER>
                pathToStoreImage = f"pdf\\" + imageDir+"\\"+str(i)

                # create the path to store the image if does not exist already
                if not os.path.exists(pathToStoreImage):
                    os.makedirs(f"pdf\\" + imageDir+"\\"+str(i))

                # extract the image information from the page
                image_info = images_info[j]

                # covert the image information in file format
                output_file = pathToStoreImage+"\\image"+str(j)+".png"
                
                # Save the final image file in the image output directory  
                image_info.Image.Save(output_file)

                # increment the image counter by 1
                image_count += 1
Section 4: Below is continuation in the for loop defined in section 1 showing how to extract text from each page of the PDF and storing it in the embedded form in the vector database.

            # get the markdown text for given page number i from the ouput 
            # variable defined in section 1
            text = output[i].get("text")

            # divide the text on each page further into chunks of 500 characters
            text_chunks = get_text_chunks(text)

            # store the text chunks in vectore store database with the filepath
            # of where the image and tables are stored and also flat whether 
            # page has image or table information or not
            storetextembedding(text_chunks, vectorstore, filepath, i, hasImage, hasTable)

if __name__ == '__main__':
    main()
Section 4A: In section we will see the functions which were used in section 4.

# function defined to split the text in chunk size of 500 chars with 
# overlap of 200 chars
def get_text_chunks(text):
    
    # define the textsplitter function
    text_splitter = CharacterTextSplitter(
        chunk_size=500,  #This means thousand characters
        chunk_overlap=200,
        length_function=len,
        separator="\n"
    )
    
    # create chunks based on the above text splitter function    
    chunks = text_splitter.split_text(text)

    # return the collection of chunks
    return chunks

#funtion to store the text embedding of collection of chunks
def storetextembedding(text_chunks, vectorstore, file, pagenumber, hasImage, hasTable):

    loadingdocuments =  []

    # create a collection of documents to be loaded in vectore store
    for text in text_chunks:
        # define the doucment with text, and metadata with filename information
        # page number to which this chunk belongs
        # whether this page has image and table information in the form of flags
        # "imagePresent": hasImage, "tablePresent":hasTable        
        document_1 = Document(page_content=text, metadata={"file_name": file, "pagenumber": pagenumber, "imagePresent": hasImage, "tablePresent":hasTable})
        loadingdocuments.append(document_1)

    # load the entire collection of documents of collection of chunks of text 
    # on a page in the vectore store
    if len(loadingdocuments) != 0:
        vectorstore.add_documents(documents=loadingdocuments)

    # retur the file name
    return file
With this we have completed the part 1 of the coding. Now we will turn our focus on part 2 of the application where we create a chat bot functionality using stream lit. Hope I have not lost you ……….

Part 2: Chatbot implementation to retrieve text, images and tables for the query

For chatbot UI we are going to use Streamlit along with langchain, chromaDB. In support of GUI we will also be using some HTML and CSS.

There are two parts to GUI. One is HTML template file and other is the core AI engine file which supports the chatbot operations. Both are python files. I will not be explaining the HTML template file. As its nothing but consist of different elements of GUI created using HTML and CSS. In the end I will be pasting the HTML template PY file as it is. Same will also be available in my Git repo.

One thing to remember is that below core chatbot file is to be run as Streamlit file using streamlit run command and not a regular python file. Now lets look at the code for the chatbot engine.

Start with the following imports:

import base64
import glob
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PyPDF2 import PdfReader
from attr import validators
from dotenv import load_dotenv
from transformers import pipeline

#htmlTemplates file will be shared at the end
from htmlTemplates import css, bot_template, user_template
Define Main function which will be calling all the functions to embed chunks and also to extract tables and images.

Load environment variables
Define the primary chatbot GUI using Streamlit API
def main():
    # load env variable
    load_dotenv()

    # Define page title
    st.set_page_config(page_title="Chat with project solution",
                       page_icon=":books:")
    
    # define the streamlit parameters
    st.write(css, unsafe_allow_html=True)

    # check if the session is active
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False

    # define the section where user will post the question
    user_question = st.text_input("Ask a question about your project solution:",key='text_input_widget', on_change=text_input_handler)

    # define checkbox to get userinput whether user wants to include 
    # image or tables in the chatbot answer or not
    st.checkbox("Include Image",key="include_image",on_change=click_image)
    st.checkbox("Include Tables", key="include_table", on_change=click_table)

    # check the session state to ensure that there is no chat history or 
    # conversation state from previous run
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}

    # if user has posted the question than handle the user input and 
    # retrieve the required document from the vector database
    if user_question:
        handle_userinput_getdocuments(user_question)

if __name__ == '__main__':
    main()
Section 1: here we will define the function handle_userinput_getdocuments which will be used the retrieve the document and display the query and answer in chat format to the user.

# define the function to retrieve the answer for question posted by the user
def handle_userinput_getdocuments(user_question):
    # define the embedding function, this should be same as the 
    # one defined while creating the embedding
    embedding_function = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

    # define the verctore store. this should be also be the same in which 
    # the embedding was stored.
    vectorstore = Chroma("Project_MCPTT", persist_directory="./chroma_mcptt",
                         embedding_function=embedding_function, collection_metadata={"hnsw:space": "cosine"})

    # create a pipelin with summarization effect on each of the retrieved 
    # document  
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Retrieve and generate using the relevant snippets of the documents
    retriever = vectorstore.as_retriever()

    # retrieve the documets using the retriever of the vectore store.
    docs = retriever.get_relevant_documents(user_question)

    responsetoquestion = ""
    filename = ""
    pageNumber = ""
Section 2: Now we traverse through each of the retrieved document and extract text, image and table from it and append to the response. We also add the link to the document along with the page number so that if user wants he can open the document and check the answer itself. It helps improve the eXplainability of the AI (XAI) solution.

    # traverse through each document retrieved by the retriever.
    for text in docs:
        # for given doc extract the filename
        newfilename = text.metadata['file_name']
        # for given doc extract the page number
        newpageNumber = text.metadata['pagenumber']
        # for given doc extract the flag whether there is imagepresent
        imagePresent = text.metadata['imagePresent']
        # for given doc extract the flag whether there is any table present
        tablePresent = text.metadata['tablePresent']
        
        # if filename is not asame as the previous document, treat it as new 
        # filename and append a link to it to the answer response
        if filename != newfilename:
           filename = newfilename
           print("Filename:",filename)
           file_url = f"http://localhost:8000/{filename}"
           responsetoquestion += f"From: <a href='{file_url}' download>{filename}</a> | "
        
        # if page number is not asame as the previous pagenumber, 
        # treat it as new pagenumber
        # and append a link to it to the answer response
        if pageNumber != newpageNumber:
           pageNumber = newpageNumber
           responsetoquestion += f"<p style='color:Tomato;'>Page Number: {pageNumber} </p>"

        # create summary of response text retrieved from the page
        summary = summarizer(text.page_content, min_length=5, max_length=300)
        # append summary of text to the response
        responsetoquestion = responsetoquestion + summary[0]["summary_text"] + "\n <br>"

        # if the flag image present is true and user has asked to see relevant 
        # image in the response
        # call the function to append the image to the response
        if imagePresent == 1 and st.session_state['include_image']:
            responsetoquestion = appendImage(responsetoquestion,newfilename,newpageNumber)
        
        # if the flag table present is true and user has asked to see relevant 
        # table in the response
        # call the function to append the table the response
        if tablePresent == 1 and st.session_state['include_table']:
            responsetoquestion = appendTable(responsetoquestion, newfilename, newpageNumber)
Below lines will be outside the above for loop….

    # if the user opted to include image in the answer
    # we include the _IMG at the end of question in chat history 
    if st.session_state['include_image']:
        user_question = user_question + "_IMG"
    
    # if the user opted to include table in the answer
    # we include the _TAB at the end of question in chat history
    if st.session_state['include_table']:
        user_question = user_question + "_TAB"

    # add the question and the final response to the chat hisotry map in the 
    # chat session
    st.session_state.chat_history[user_question] = responsetoquestion

    # display the entire chat history
    displaychathistory()
Some of the functions which we used in the above 2 code blocks….

This method is to display the complete chat history….

def displaychathistory():
    reversechat = dict(reversed(list(st.session_state.chat_history.items())))
    for i, (k, v) in enumerate(reversechat.items()):
        st.write(user_template.replace("{{MSG}}", k), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", v), unsafe_allow_html=True)
This method is use to append the image to the response….

def appendImage(responsetoquestion,newfilename,newpageNumber):
    # from the page number, filename construct the directory 
    # where the images are stored
    imageDir = getdirname(newfilename)

    # covert it to path where images are stored
    pathToStoreImage = f"pdf\\" + imageDir + "\\" + str(newpageNumber)

    # get list of all the images in the above directory
    imageNames = list(glob.glob(pathToStoreImage+"\\*.png"))

    # converse throuch all the image names
    for filepath in imageNames:
        print("Image file path",filepath) # this just of debugging purpose

        # construct the url for image
        file_url = f"http://localhost:8000/{filepath}"
        print("File URL",file_url)
        imageTag = f"<img src='{file_url}'><br>"

        # append the image url to the response and move to next image or exit
        responsetoquestion = responsetoquestion + imageTag
    # return the response with the image names with url embedded 
    # in html format
    return responsetoquestion
This method is use to append the image to the response….

def appendTable(responsetoquestion,newfilename,newpageNumber):
    # from the page number, filename construct the directory 
    # where the tables are stored
    tableDir = getdirname(newfilename)

    # covert it to path where tables are stored
    pathToStoreTable = f"table\\" + tableDir + "\\" + str(newpageNumber)

    # get list of table filenames in pkl format from directory
    tableNames = list(glob.glob(pathToStoreTable+"\\*.pkl"))

    for filepath in tableNames:
        print("Table file path",filepath) # for debugging purpose
        # convert the pickle file to dataframe
        df = pd.read_pickle(filepath)

        # remove the very first column from the dataframe
        df = df.iloc[1:]

        #set the css style for the table
        df.style.set_table_styles([{'selector': '', 'props': [('border', '2px solid green')]}])
      
        # append the table content to response in html format       
        responsetoquestion = responsetoquestion + df.to_html(index=False, header=False, border=5)

    # return the final response after adding all the tables
    return responsetoquestion
In this chatbot solution I have not summarized the final response using a prompt. We can also do that. But when we summarise all the response sections using a prompt and present it as single answer, we loose track of the page and document number. As the final answer is summary of all the text received from vector search irrespective. Instead that we had opted for more detailed answer where we have shown summary of each or retrieved text chunk along with images/table on that page to which the chunk belongs to. The answer will not look clean but will be helpful to technical folks, who for the most part are looking for direction to the right document rather than some final answer (which many a times are very doubt full).

Code for htmltemplate file: Save it as htmlTemplates.py and put it in same directory as core chatbot engine py file.

css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="http://localhost:8000/AI.png">
    </div>
    <div class="message"><span style="white-space: pre-line">{{MSG}}</span></div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="http://localhost:8000/users.png">
    </div>    
    <div class="message"><span style="white-space: pre-line">{{MSG}}</span></div>
</div>
'''

file_list_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="http://localhost:8000/users.png">
    </div>    
    <div class="message"><span style="white-space: pre-line">{{MSG}}</span></div>
</div>
'''

check_box = '''
   <div class="checkbox-container">
        <div id="custom-checkbox" style="width: 20px; height: 20px; border: 1px solid black; display: inline-block; cursor: pointer;" onclick="toggleCheckbox()"></div>
        <label for="custom-checkbox">Custom Div Checkbox</label>
   </div>
   <br>
    <input type="submit" value="Submit" onclick="return sendDivCheckboxValue();">
'''
Folder structure will all the files will look like below. pdf folder will have all the pdf files for which you want to create the embeddings.


Folder structure for the solution.
