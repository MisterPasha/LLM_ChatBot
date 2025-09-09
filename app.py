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

# Set the Ollama base URL from environment variable or default to local
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://ollama:11434") 

st.set_page_config(page_title="ComradGPT", layout="centered")
st.markdown("<h1 style='text-align: center;'>🥸 ComradeGPT</h1>", unsafe_allow_html=True)
st.caption("Your Grumpy Soviet Grandpa — now available online!")

# Load the Ollama model with streaming
model = ChatOllama(model="llama3.2:1b", base_url=OLLAMA_BASE_URL, streaming=True)
output_parser = StrOutputParser()

system_prompt = SystemMessagePromptTemplate.from_template(
    "You are ComradGPT, speaking in a USSR accent and style."
    "Your speaking style is an old grumpy man in his 70s."
    "your answers are short and sarcastic"
)

# Define the save file for chat history
SAVE_FILE = "chat_history.json"

# Function to load and save chat history
def load_chat_history():
    if os.path.exists(SAVE_FILE) and os.path.getsize(SAVE_FILE) > 0:
        with open(SAVE_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []  # fallback if file content is broken
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
        with st.chat_message("user", avatar="👽"):
            st.markdown(chat["user"])
        with st.chat_message("ollama", avatar="🥸"):
            st.markdown(chat["ollama"])

user_input = st.chat_input("Talk to good old ComradGPT here...")

# Streaming response handler
def stream_response(chat_history):
    chat_template = ChatPromptTemplate.from_messages(chat_history)
    chain = chat_template | model | output_parser
    response_stream = chain.stream({})  # streaming
    return response_stream

def get_chat_history():
    chat_history = [system_prompt]
    for chat in st.session_state["chat_history"]:
        chat_history.append(HumanMessagePromptTemplate.from_template(chat["user"]))
        chat_history.append(AIMessagePromptTemplate.from_template(chat["ollama"]))
    return chat_history

# Handle user input
if user_input:
    st.chat_message("user", avatar="👽").markdown(user_input)

    # Prepare prompt
    prompt = HumanMessagePromptTemplate.from_template(user_input)
    history = get_chat_history()
    history.append(prompt)

    with st.chat_message("ollama", avatar="🥸"):
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

