from flask import Blueprint, request, jsonify
from app.services.image_processor import describe_images
from app.services.vector_store import save_to_faiss

generate_db_bp = Blueprint("generate_db", __name__)

@generate_db_bp.route("/generate-db", methods=["POST"])
def generate_db():
    data = request.get_json()
    author_name = data.get("author_name")
    author_description = data.get("author_description")
    museum_id = data.get("museum_id")
    artworks = data.get("artworks", [])

    # Pass full artworks list to include title, description, and image

    documents = describe_images(artworks, museum_id)
    save_to_faiss(documents)
    return jsonify({"status": "success", "count": len(documents)})