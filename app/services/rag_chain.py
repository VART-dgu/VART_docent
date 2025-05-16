from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from app.services.vector_store import load_faiss
from langchain.prompts import PromptTemplate

# Pre-initialize FAISS retriever, LLM, and RAG chain for reuse
_db = load_faiss()
_llm = OllamaLLM(model="llama3.3:70b")
_custom_prompt = PromptTemplate.from_template("""
당신은 전문적인 미술관 도슨트입니다.
제공된 문서를 바탕으로만 답변해야 합니다.
다음 정보를 참고하여 관람객의 질문에 성실하게 답해주세요.
모든 답변은 반드시 한국어로, 한글만 사용해 주세요.

{context}

질문: {question}
""")
_rag_chain = RetrievalQA.from_chain_type(
    llm=_llm,
    retriever=_db.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": _custom_prompt}
)

def make_answer(question, metadata):
    # 질문 주제 검증: 미술관, 작가, 작품 관련 질문이 아니면 안내 메시지 리턴
    if not any(keyword in question for keyword in ["작가", "미술관", "작품"]):
        return {"result": "이 시스템은 미술관, 작가, 작품에 대한 질문에만 답변할 수 있습니다. 해당 주제와 관련된 질문을 해주세요.", "source_documents": []}

    # artwork_id 검증
    target_artwork_id = metadata.get("artwork_id")
    if target_artwork_id is None:
        return {"result": "artwork_id가 제공되지 않았습니다.", "source_documents": []}

    db = load_faiss()
    all_docs = db.similarity_search(question, k=100)

    author_id = metadata.get("author_id")
    museum_id = metadata.get("museum_id")

    # target_artwork_id 기준 문서 찾기
    target_doc = next((doc for doc in all_docs if doc.metadata.get("artwork_id") == target_artwork_id), None)

    if not target_doc:
        return {"result": "현재 보고 있는 작품에 대한 정보를 찾을 수 없습니다.", "source_documents": []}

    # 같은 미술관 내 다른 작품들
    museum_docs = [doc for doc in all_docs if doc.metadata.get("museum_id") == museum_id and doc.metadata.get("artwork_id") != target_artwork_id]

    # 그 중 같은 작가 작품
    author_docs = [doc for doc in museum_docs if doc.metadata.get("author_id") == author_id]

    context_parts = []

    if author_docs:
        context_parts.append("같은 미술관에 전시 중인 이 작가의 다른 작품들에 대한 설명입니다:\n" + "\n\n".join(doc.page_content for doc in author_docs))
    if museum_docs:
        context_parts.append("현재 미술관에 전시 중인 다른 작품들입니다:\n" + "\n\n".join(doc.page_content for doc in museum_docs))
    context_parts.append("사용자가 현재 보고 있는 작품입니다:\n" + target_doc.page_content)

    context = "\n\n".join(context_parts)

    if not author_docs and not museum_docs:
        return {"result": "이 시스템은 미술관, 작가, 작품에 대한 질문에만 답변할 수 있습니다. 해당 주제와 관련된 질문을 해주세요.", "source_documents": [target_doc]}

    result = _rag_chain.invoke({"question": question, "context": context})
    return result