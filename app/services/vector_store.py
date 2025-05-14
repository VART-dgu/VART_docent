import os
from langchain_community.vectorstores.faiss import FAISS
from langchain_ollama import OllamaEmbeddings

def save_to_faiss(documents):
    embeddings = OllamaEmbeddings(model="llama3.3:70b")
    db = FAISS.from_documents(documents, embeddings)
    db.save_local("mmrag_faiss")

def load_faiss():
    embeddings = OllamaEmbeddings(model="llama3.3:70b")
    db = FAISS.load_local("mmrag_faiss", embeddings, allow_dangerous_deserialization=True)
    return db