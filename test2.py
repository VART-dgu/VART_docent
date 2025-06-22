from app.services.rag_chain import make_answer

test_artworks = [
    {
        "artwork_id": "art1",
        "author_id": "author1",
        "author_name": "홍길동"
    },
    {
        "artwork_id": "art2",
        "author_id": "author2",
        "author_name": "이몽룡"
    },
]

for art in test_artworks:
    question = "이 작가의 다른 작품은 어떤 특징이 있나요?"
    metadata = {
        "artwork_id": art["artwork_id"],
        "author_id": art["author_id"],
        "museum_id": "museum1"
    }
    response = make_answer(question, metadata)
    print(f"\n작품 ID: {art['artwork_id']} / 작가: {art['author_name']}")
    print("질문:", question)
    print("답변:", response["result"])
