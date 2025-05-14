import os
import base64
import requests
from langchain.schema import Document

def describe_image_with_llava(image_path, title, original_description):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    res = requests.post("http://localhost:11434/api/generate", json={
        "model": "llava:13b",
        "prompt": f"작품 제목: {title}\n작가의 설명: {original_description}\n이 정보를 참고해서 이 이미지를 한국어로 설명해줘. 영어는 쓰지 마.",
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
        description = describe_image_with_llava(image_path, title, original_description)
        full_content = f"작품 제목: {title}\n작가 설명: {original_description}\nLMM 설명: {description}"
        artist = art.get("artist", "")
        documents.append(Document(
            page_content=full_content,
            metadata={
                "source": image_path,
                "artist": artist,
                "museum_id": museum_id
            }
        ))
    return documents