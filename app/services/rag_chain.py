from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from app.services.vector_store import load_faiss

def build_rag_qa():
    retriever = load_faiss().as_retriever()
    llm = OllamaLLM(model="llama3.3:70b")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain