from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from app.services.vector_store import load_faiss
from langchain.vectorstores.base import VectorStore
from langchain.prompts import PromptTemplate
from openai import OpenAI

import os
import json
from app.services.vector_store import save_to_faiss
import requests

gpt_client = OpenAI()


_custom_prompt = PromptTemplate.from_template("""
You are a professional museum docent.
Use only the provided documents and respond in fluent English.

{context}

Question: {question}
""")

def make_answer(question, metadata, ollama_url="http://ollama-2:11434"):
    _db = load_faiss(ollama_url=ollama_url)
    # Classification with up to 3 retries
    question_en = ""
    for attempt in range(3):
        trans_en_response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": "llama3.3:70b",
                "prompt": (
                    f"Step 1: Classify whether the following Korean question is about a museum, artwork, or artist.\n"
                    f" If it is, set \"relevant\":1; otherwise, set \"relevant\":0.\n"
                    f"Step 2: If relevant, translate the question into English. If not relevant, leave \"question_en\" empty.\n"
                    f"Output as JSON with keys: relevant (0 or 1), question_en (string).\n\n"
                    f"한국어 질문:\n{question}"
                ),
                "stream": False,
                "temperature": 0.0,
                "top_p": 1.0
            }
        )
        classification = trans_en_response.json().get("response", "")
        try:
            print("Attempting to parse JSON...")
            # Find the index of the first '{' and the last '}'
            start_index = classification.find('{')
            end_index = classification.rfind('}')

            if start_index != -1 and end_index != -1 and start_index < end_index:
                # Extract the substring that contains only the JSON
                json_string = classification[start_index : end_index + 1]
                cls_obj = json.loads(json_string)
                print(f"Parsed JSON object: {cls_obj}")
                # If parsing is successful, you can break the loop or continue processing
            else:
                print("No valid JSON structure found in the output.")

            if cls_obj.get("relevant", 0) == 0:
                question_en = ""
                continue
            else:
                question_en = cls_obj.get("question_en", "").strip()
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            print("Retrying...")
            continue # Continue to the next attempt if parsing fails

        

    # If fewer than 2 relevant classifications, treat as irrelevant
    print(question_en)
    if not question_en:
        return {"result": "", "source_documents": []}
    print(question_en)

    # artwork_id 검증
    target_artwork_id = metadata.get("artwork_id")
    if target_artwork_id is None:
        return {"result": "artwork_id가 제공되지 않았습니다.", "source_documents": []}

    # 메타데이터 "field" 값을 기준으로 각 유형의 관련 문서를 검색합니다.
    fields = [
        ("Viewer Perspective", "viewer_description"),
        ("Visual Description", "visual_description"),
        ("Curatorial Commentary", "curatorial_commentary"),
        ("Emotional Impression", "emotional_impression")
    ]
    context_parts = []
    for title, field in fields:
        # as_retriever(): FAISS 인덱스를 검색기(retriever) 객체로 변환합니다.
        # filter: metadata["field"]가 해당 field인 문서만 조회, k=5로 상위 5개 문서 반환
        docs = _db.as_retriever(
            search_kwargs={"filter": {"field": field}, "k": 5}
        ).get_relevant_documents(question_en)  # get_relevant_documents(): 질의와 유사한 문서 리스트 반환
        if docs:
            # 검색된 문서들의 page_content를 합쳐서 컨텍스트 블록으로 생성
            context_parts.append(
                f"{title}:\n" +
                "\n\n".join(d.page_content for d in docs)
            )
    # 모든 필드별 컨텍스트 블록을 하나의 문자열로 결합
    context = "\n\n".join(context_parts)

    author_id = metadata.get("author_id")
    museum_id = metadata.get("museum_id")

    # RAG 체인을 초기화합니다.
    # OllamaLLM: ollama 서버의 llama3.3:70b 모델을 사용
    llm = OllamaLLM(model="llama3.3:70b", base_url=ollama_url)
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=_db.as_retriever(),  # 내부적으로도 같은 retriever 사용
        return_source_documents=True,  # 참조된 문서를 함께 반환
        chain_type_kwargs={"prompt": _custom_prompt}  # 사용자 정의 프롬프트 템플릿
    )
    # invoke(): question_en과 구축된 context를 넘겨 최종 답변 생성
    result = rag_chain.invoke({"query": question_en, "context": context})

    answer_en = result["result"]
    print(f"answer en: {answer_en}")

    # 번역을 OpenAI GPT-4o로 수행
    response = gpt_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that translates English to natural Korean."},
            {"role": "user", "content": f"Translate the following English sentence into natural Korean:\n\n{answer_en}"}
        ],
        temperature=0.7,
        top_p=0.9
    )
    answer_ko = response.choices[0].message.content.strip()
    result["result"] = answer_ko
    print(f"answer ko: {answer_ko}")

    return result