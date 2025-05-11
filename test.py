import os
import requests
import base64
from langchain_community.vectorstores.faiss import FAISS
from langchain.schema import Document
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains import RetrievalQA

# 1. LLaVA (GPU 1) 기반 이미지 캡셔닝 → 텍스트
def describe_image_with_llava(image_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU 1 사용
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    res = requests.post("http://localhost:11434/api/generate", json={
        "model": "llava:13b",
        "prompt": "Describe this image.",
        "images": [img_b64],
        "stream": False
    })
    return res.json()["response"]

# 2. FAISS 저장소 생성
def build_vector_db_from_images(image_paths):
    documents = []
    for path in image_paths:
        description = describe_image_with_llava(path)
        documents.append(Document(page_content=description, metadata={"source": path}))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 0 사용 (Embedding LLM)
    embeddings = OllamaEmbeddings(model="llama3.3:70b")
    db = FAISS.from_documents(documents, embeddings)
    db.save_local("mmrag_faiss")
    return db

# 3. RAG 질의 시스템 생성
def build_rag_qa_system():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 0 사용
    embeddings = OllamaEmbeddings(model="llama3.3:70b")
    db = FAISS.load_local("mmrag_faiss", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    llm = OllamaLLM(model="llama3.3:70b")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa_chain

# 4. 예시 실행
if __name__ == "__main__":
    image_list = ["image1.jpg", "image2.jpg"]
    build_vector_db_from_images(image_list)
    qa = build_rag_qa_system()
    query = "what is different between cat and dog?. 한국어로 대답해줘."
    result = qa.invoke(query)
    print("Answer:", result["result"])
    print("Source:", [doc.metadata["source"] for doc in result["source_documents"]])
