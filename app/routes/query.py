from flask import Blueprint, request, jsonify
from app.services.rag_chain import make_answer

query_bp = Blueprint("query", __name__)

@query_bp.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    question = data.get("question")
    metadata = {
        "source": data.get("source"),
        "artwork_id": data.get("artwork_id"),
        "author_id": data.get("author_id"),
        "author_name": data.get("author_name"),
        "museum_id": data.get("museum_id")
    }
    result = make_answer(question, metadata)
    return jsonify({
        "answer": result["result"],
        "sources": [doc.metadata["source"] for doc in result["source_documents"]]
    })