from flask import Blueprint, request, jsonify
from app.services.rag_chain import build_rag_qa

query_bp = Blueprint("query", __name__)

@query_bp.route("/query", methods=["POST"])
def query():
    question = request.get_json().get("question")
    qa = build_rag_qa()
    result = qa.invoke(question)
    return jsonify({
        "answer": result["result"],
        "sources": [doc.metadata["source"] for doc in result["source_documents"]]
    })