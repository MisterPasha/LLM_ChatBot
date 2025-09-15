import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.messages import HumanMessage
import os, json
import warnings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


# Suppress specific warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# Load all PDFs from the 'rag-dataset' directory
pdfs = []
for root, dirs, files in os.walk('rag-dataset'):
    # print(root, dirs, files)
    for file in files:
        if file.endswith('.pdf'):
            pdfs.append(os.path.join(root, file))

# Load and split documents
docs = []
for pdf in pdfs:
    loader = PyMuPDFLoader(pdf)
    pages = loader.load()

    docs.extend(pages)

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)
print(len(docs), len(chunks))

# Create embeddings and vector store
embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")
index = faiss.IndexFlatL2(768)  # Dimension for 'nomic-embed-text' is 768
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

# Add documents to the vector store
ids = vector_store.add_documents(documents=chunks)

# Add documents to the vector store
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs = {'k': 3, 
                                                                          'fetch_k': 100,
                                                                          'lambda_mult': 1})

# Ollama base URL from environment variable (http://ollama:11434) 
# or default to localhost (http://localhost:11434)
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434") 

# Load the RAG  prompt from LangChain Hub
# prompt = hub.pull("rlm/rag-prompt")

# Custom prompt for customer support
PROMPT = """
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Answer in bullet points. Make sure your answer is relevant to the question 
    and it is answered from the context only.
    Question: {question} 
    Context: {context} 
    Answer:
"""
prompt = ChatPromptTemplate.from_template(PROMPT)

st.set_page_config(page_title="Customer Support Knowledge Agent", layout="centered")
st.markdown("<h1 style='text-align: center;'>⚙️DigiSoft⚙️</h1>", unsafe_allow_html=True)
st.caption("Chat with our virtual assistant about our products, services, policies, and procedures.")

# Load the Ollama model with streaming
model = ChatOllama(model="llama3.2:1b", base_url=OLLAMA_BASE_URL, streaming=True)
output_parser = StrOutputParser()

system_prompt = SystemMessagePromptTemplate.from_template(
    "You are a customer support agent for a software company called DigiSoft."
    "Your speaking style is friendly and professional."
    "your answers are short and concise."
    "You provide accurate information about the company's products, services, policies, and procedures."
)

# Define the save file for chat history
SAVE_FILE = "chat_history.json"

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# Function to load and save chat history
def load_chat_history():
    if os.path.exists(SAVE_FILE) and os.path.getsize(SAVE_FILE) > 0:
        with open(SAVE_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []  
    return []

def save_chat_history(history):
    with open(SAVE_FILE, "w") as f:
        json.dump(history, f)

# Session state for storing messages
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = load_chat_history()

# Chat input form
with st.container():
    for chat in st.session_state["chat_history"]:
        with st.chat_message("user", avatar="🥸"):
            st.markdown(chat["user"])
        with st.chat_message("ollama", avatar="⚙️"):
            st.markdown(chat["ollama"])

user_input = st.chat_input("Type here to ask me anything...")

# Streaming response handler
def stream_response(chat_history):
    chat_template = ChatPromptTemplate.from_messages(chat_history)
    #chain = chat_template | model | output_parser
    #response_stream = chain.stream({})  # streaming
    rag_chain = (
        {"context": retriever|format_docs, "question": RunnablePassthrough()}
        | chat_template
        | prompt
        | model
        | output_parser
        )
    response_stream = rag_chain.stream(user_input)
    return response_stream

def get_chat_history():
    chat_history = [system_prompt]
    for chat in st.session_state["chat_history"]:
        chat_history.append(HumanMessagePromptTemplate.from_template(chat["user"]))
        chat_history.append(AIMessagePromptTemplate.from_template(chat["ollama"]))
    return chat_history

# Handle user input
if user_input:
    st.chat_message("user", avatar="🥸").markdown(user_input)

    # Prepare prompt
    prompt = HumanMessagePromptTemplate.from_template(user_input)
    history = get_chat_history()
    history.append(prompt)

    with st.chat_message("ollama", avatar="⚙️"):
        full_response = ""
        response_container = st.empty()

        for token in stream_response(history):
            full_response += token
            response_container.markdown(full_response + "▌")  # Blinking cursor effect

        response_container.markdown(full_response)  # Final response

    st.session_state["chat_history"].append({
        "user": user_input,
        "ollama": full_response
    })

    # Save chat history to file
    save_chat_history(st.session_state["chat_history"])

