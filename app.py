import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain_core.prompts import PromptTemplate
import os
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.chat_models import ChatOpenAI


# zfuNNzjY10pWn6KKD3lfT3BlbkFJKZVniL8uhauKhYi6ahA8
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunk(text):
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len
        )
    docs = r_splitter.split_text(text)
    return docs


def get_vectorstore(text_chunk,model_name):
    db=''
    if model_name == 'OpenAI':
        embeddings=OpenAIEmbeddings()
        db=FAISS.from_texts(text_chunk,embeddings)
    return db

def get_conversation_chain(db,model_name):
    if model_name == 'OpenAI':
        llm=ChatOpenAI()
        memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                      retriever=db.as_retriever(),
                                                      memory=memory)
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def select_llm():
    """
    Read user selection of parameters in Streamlit sidebar.
    """
    model_name = "OpenAI"
    key = st.text_input("Enter API key", type="password")
    return model_name, key
    #return key


def model(pdf_doc,model_name):
        # convert pdf to text format
        raw_text = get_pdf_text(pdf_doc)
        # get chunk data
        text_chunk = get_text_chunk(raw_text)
        # create vector store
        db = get_vectorstore(text_chunk,model_name)
        # conversation
        st.session_state.conversation = get_conversation_chain(db,model_name)



def main():
    st.set_page_config(page_title='Chat with our PDF')
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None


    st.header("👋 Hello! I am your AI Chatbot!")
    user_question = st.text_input("Ask me any question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader('Your Document')
        pdf_doc=st.file_uploader("Upload your PDF Here", accept_multiple_files=True,type="pdf")
        model_name, key = select_llm()
        if st.button('Process'):
            with st.spinner("Processing"):
                if model_name == "OpenAI":
                    os.environ["OPENAI_API_KEY"] = key
                    model(pdf_doc,model_name)


if __name__== '__main__':
    main()
