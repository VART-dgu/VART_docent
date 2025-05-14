from flask import Blueprint, request, jsonify
from app.services.image_processor import describe_images
from app.services.vector_store import save_to_faiss

generate_db_bp = Blueprint("generate_db", __name__)

@generate_db_bp.route("/generate-db", methods=["POST"])
def generate_db():
    query = request.get_json().get("query")
    # TODO: Replace with actual DB query
    image_urls = ["image1.jpg", "image2.jpg"]
    documents = describe_images(image_urls)
    save_to_faiss(documents)
    return jsonify({"status": "success", "count": len(documents)})