import os
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Recursively collect all PDF paths under root_dir
def collect_pdf_paths(root_dir: str = "rag-dataset") -> List[str]:
    paths = []
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                paths.append(os.path.join(r, f))
    return paths

# Load pages from a list of PDF paths using PyMuPDFLoader
def load_documents(pdf_paths: List[str]):
    docs = []
    for path in pdf_paths:
        pages = PyMuPDFLoader(path).load()
        docs.extend(pages)
    return docs

# Split documents into chunks using RecursiveCharacterTextSplitter
def split_documents(docs, chunk_size: int = 1000, chunk_overlap: int = 100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)
