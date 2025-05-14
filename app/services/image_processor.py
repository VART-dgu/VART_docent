import os
import base64
import requests
from langchain.schema import Document

def describe_image_with_llava(image_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    res = requests.post("http://localhost:11434/api/generate", json={
        "model": "llava:13b",
        "prompt": "이 이미지를 설명해줘. 반드시 한국어로 설명하고 영어는 쓰지 마.",
        "images": [img_b64],
        "stream": False
    })
    return res.json()["response"]

def describe_images(image_paths):
    documents = []
    for path in image_paths:
        description = describe_image_with_llava(path)
        documents.append(Document(page_content=description, metadata={"source": path}))
    return documents