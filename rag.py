import os
from operator import itemgetter

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Single prompt that contains history + RAG variables
def _build_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "You are DigiSoft support for customers. Be concise and use ONLY the provided context. "
                   "If you don't know, say you don't know."),
        MessagesPlaceholder("history"),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer in bullet points.")
    ])

# Return FAISS retriever vector store
def _build_retriever(chunks, embeddings):
    if chunks:
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 100, "lambda_mult": 0.5},
        )

# full RAG pipeline (retriever + prompt + model + parser) and returns a streaming chain
def build_rag_chain(chunks, ollama_base_url=None, gen_model="llama3.2:1b", embed_model="nomic-embed-text"):
    base_url = ollama_base_url or os.getenv("OLLAMA_HOST", "http://ollama:11434")

    # Embeddings & retriever
    embeddings = OllamaEmbeddings(model=embed_model, base_url=base_url)
    retriever = _build_retriever(chunks, embeddings)

    # Prompt & model
    prompt = _build_prompt()
    model = ChatOllama(model=gen_model, base_url=base_url, streaming=True)
    parser = StrOutputParser()

    # Chain wiring:
    #  - route "question" to retriever -> format_docs for {context}
    #  - pass through "question" and "history" into the prompt
    chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "history": itemgetter("history"),
        }
        | prompt
        | model
        | parser
    )
    return chain
