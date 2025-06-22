import os
from langchain_community.vectorstores.faiss import FAISS
from langchain_ollama import OllamaEmbeddings

def save_to_faiss(documents, ollama_url="http://ollama-1:11434", index_path="mmrag_faiss"):
    embeddings = OllamaEmbeddings(model="llama3.3:70b", base_url=ollama_url)
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(index_path)

def load_faiss(ollama_url):
    embeddings = OllamaEmbeddings(model="llama3.3:70b", base_url=ollama_url)
    db = FAISS.load_local("mmrag_faiss", embeddings, allow_dangerous_deserialization=True)
    return db