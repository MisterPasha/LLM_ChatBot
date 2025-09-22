import os
import json
import warnings
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from pdf_utils import collect_pdf_paths, load_documents, split_documents
from rag import build_rag_chain, format_docs  

# --- Setup ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
warnings.filterwarnings("ignore")
load_dotenv()

st.set_page_config(page_title="Customer Support Knowledge Agent", layout="centered")
st.markdown("<h1 style='text-align: center;'>⚙️DigiSoft⚙️</h1>", unsafe_allow_html=True)
st.caption("Chat with our virtual assistant about our products, services, policies, and procedures.")

SAVE_FILE = "chat_history.json"

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

def get_history_messages():
    msgs = []
    for chat in st.session_state["chat_history"]:
        msgs.append(HumanMessage(content=chat["user"]))
        msgs.append(AIMessage(content=chat["ollama"]))
    return msgs

# Session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = load_chat_history()

# --- Build RAG once at startup ---
# Collect PDFs → load → split
pdf_paths = collect_pdf_paths("rag-dataset")
docs = load_documents(pdf_paths)
chunks = split_documents(docs, chunk_size=1000, chunk_overlap=100)

# Build the chain (embeddings/retriever/prompt/model wired inside)
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://ollama:11434")
rag_chain = build_rag_chain(chunks, ollama_base_url=OLLAMA_BASE_URL)

# --- UI: render previous chat ---
with st.container():
    for chat in st.session_state["chat_history"]:
        with st.chat_message("user", avatar="🥸"):
            st.markdown(chat["user"])
        with st.chat_message("assistant", avatar="⚙️"):
            st.markdown(chat["ollama"])

user_input = st.chat_input("Type here to ask me anything...")

# --- Handle user input and stream output ---
if user_input:
    st.chat_message("user", avatar="🥸").markdown(user_input)
    history_msgs = get_history_messages()

    with st.chat_message("assistant", avatar="⚙️"):
        full_response = ""
        box = st.empty()
        # Stream tokens from the chain
        for token in rag_chain.stream({"question": user_input, "history": history_msgs}):
            full_response += token
            box.markdown(full_response + "▌")
        box.markdown(full_response)

    st.session_state["chat_history"].append({"user": user_input, "ollama": full_response})
    save_chat_history(st.session_state["chat_history"])
