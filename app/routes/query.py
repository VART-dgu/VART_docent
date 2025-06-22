from flask import Blueprint, request, jsonify
from app.services.rag_chain import make_answer

query_bp = Blueprint("query", __name__)

@query_bp.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    question = data.get("question")
    metadata = {
        "title": data.get("title"),
        "artwork_id": data.get("artwork_id"),
        "author_id": data.get("author_id"),
        "author_name": data.get("author_name"),
        "museum_id": data.get("museum_id")
    }
    ollama_urls = ["http://ollama-1:11434", "http://ollama-2:11434", "http://ollama-3:11434"]
    result = make_answer(question, metadata, ollama_url=ollama_urls[1])
    return jsonify({
        "answer": result["result"],
        "sources": [doc.metadata["source"] for doc in result["source_documents"]]
    })