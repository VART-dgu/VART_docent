import os
import base64
import requests
from langchain.schema import Document

def describe_image_with_llava(image_url, title, author_name, author_description, original_description):
    with open(image_url, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    res = requests.post("http://localhost:11434/api/generate", json={
        "model": "llava:13b",
        #"prompt": f"작가 이름: {author_name}\n작가 소개: {author_description}\n작품 제목: {title}\n작가의 작품 설명: {original_description}\n이 정보를 참고해서 이 이미지를 한국어로 설명해줘. 알파벳이나  쓰지 마.",
        "prompt": f"Please describe the following artwork in detail based on this information.\n\nAuthor Name: {author_name}\nAuthor Bio: {author_description}\nArtwork Title: {title}\nArtist's Description of the Artwork: {original_description}\n\nUse clear and descriptive English. Focus on visual details and the artistic intention behind the work.",
        "images": [img_b64],
        "stream": False
    })
    return res.json()["response"]

def describe_images(artworks, museum_id):
    documents = []
    for art in artworks:
        image_path = art["image_url"]
        title = art.get("title", "")
        original_description = art.get("description", "")
        author_id = art.get("author_id")
        author_name = art.get("author_name", "")
        author_description = art.get("author_description", "")
        description = describe_image_with_llava(image_path, title, author_name, author_description, original_description)
        full_content = f"작품 제목: {title}\n작가 설명: {original_description}\nLMM 설명: {description}"
        documents.append(Document(
            page_content=full_content,
            metadata={
                "source": image_path,
                "author_id": author_id,
                "author_name": author_name,
                "museum_id": museum_id
            }
        ))
    return documents