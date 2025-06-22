import os
import base64
import requests
import json
import re
from langchain.schema import Document


def describe_image_with_llava(image_url, title, author_name, author_description, original_description, llava_url, llama_url):
    with open(image_url, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    #img_b64 = image_url
    prompt_en = (
        f"You are a professional museum curator. Based on your expert knowledge, please write a description to help visitors better understand and appreciate the artwork.\n\n"
        f"Author Name: {author_name}\n"
        f"Author Introduction: {author_description}\n"
        f"Title of the Artwork: {title}\n"
        f"Author's Description: {original_description}\n\n"
        f"Look at the artwork image and write the following information in English, clearly and kindly, from a visitor's point of view:\n"
        f"1. What should a first-time viewer pay attention to? What kind of mood or emotions might they feel? Provide a detailed guide.\n"
        f"2. Describe the painting based on its visual elements such as composition, color, and shape.\n"
        f"3. Based on the image and author's description, explain what message or story the artist wants to convey to the audience.\n"
        f"4. Use poetic and soft language to describe the overall mood and emotional impression of the artwork. Help the viewer empathize with the feeling.\n"
        f"5. Summarize this artwork using 3 to 5 keywords. (e.g., peace, city, stillness)\n"
        f"6. Describe the art style in 1 to 2 words. (e.g., Impressionism, Cubism, Modern Abstract)\n"
        f"7. Provide 3 to 5 main subject tags depicted in the artwork. (e.g., bull, storm, festival)\n"
        f"Do not include empty or vague answers such as \"...\", \"unknown\", or \"none\". All fields must be fully and faithfully written.\n"
        f"The output should be in the following JSON format. All content must be in plain English text (no markdown or HTML tags):\n"
        f'{{\n'
        f'  "viewer_description": "...",\n'
        f'  "visual_description": "...",\n'
        f'  "curatorial_commentary": "...",\n'
        f'  "emotional_impression": "...",\n'
        f'  "tags": ["tag1", "tag2", "..."],\n'
        f'  "art_style": "...",\n'
        f'  "subject_tags": ["subject1", "subject2", "..."]\n'
        f'}}'
    )

    def clean_response_text(text):
        text = re.sub(r"<[^>]+>", "", text)  # remove HTML tags
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # remove markdown bold
        text = re.sub(r"\*(.*?)\*", r"\1", text)  # remove markdown italics
        text = text.encode('utf-8').decode('unicode_escape')  # decode escaped unicode
        return text.strip()

    def _should_retry(obj):
        if not isinstance(obj, dict):
            return True
        placeholders = {"", "...", "none", "None", "null", "Null"}
        for v in obj.values():
            if isinstance(v, str):
                val = v.strip()
                if val in placeholders or len(val) < 10:
                    return True
                non_alpha_ratio = len(re.findall(r"[^a-zA-Z\s]", val)) / max(len(val),1)
                if non_alpha_ratio > 0.5:
                    return True
            elif isinstance(v, list):
                if not v:
                    return True
                for item in v:
                    if not isinstance(item, str):
                        return True
                    val = item.strip()
                    if val in placeholders or len(val) < 3:
                        return True
        return False

    def try_generate():
        for attempt in range(6):
            try:
                response = requests.post(
                    f"{llava_url}/api/generate",
                    json={"model": "llava:13b", "images": [img_b64], "prompt": prompt_en, "stream": False, "temperature": 0.7, "top_p": 0.9}
                )
                resp_text = response.json().get("response", "")
                cleaned = clean_response_text(resp_text)
                return json.loads(cleaned)
            except (json.JSONDecodeError, requests.RequestException) as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")
                continue
        return {}

    json_obj = try_generate()
    if not json_obj:
        return {}
    retry_count = 0
    while _should_retry(json_obj) and retry_count < 3:
        json_obj = try_generate()
        retry_count += 1
        
    print(f"try gen: {json_obj}")

    return json_obj

def describe_images(artworks, museum_id, llava_url="http://ollama-1:11434", llama_url="http://ollama-2:11434"):
    documents_to_add = []
    for i, art in enumerate(artworks):
        image_path = art["image_url"]
        artwork_id = art.get("artwork_id")
        title = art.get("title", "")
        original_description = art.get("description", "")
        author_id = art.get("author_id")
        author_name = art.get("author_name", "")
        author_description = art.get("author_description", "")
        json_response = describe_image_with_llava(image_path, title, author_name, author_description, original_description, llava_url=llava_url, llama_url=llama_url)

        if not json_response:
            continue

        metadata = {
            "source": image_path,
            "artwork_id": artwork_id,
            "author_id": author_id,
            "author_name": author_name,
            "museum_id": museum_id,
            "title": title
        }
        if isinstance(json_response, dict):
            json_obj = json_response
        else:
            json_obj = {}

        content = json.dumps(json_obj, ensure_ascii=False, indent=2)
        full_doc = Document(page_content=content, metadata={**metadata, "field": "full"})
        # 전체 설명 하나로 저장 (전체 요약, 전시 설명 등에서 활용)
        documents_to_add.append(full_doc)

        for key in ["viewer_description", "visual_description", "curatorial_commentary", "emotional_impression", "art_style", "abstraction_score"]:
            text = json_obj.get(key, "")
            if text != "":
                documents_to_add.append(Document(
                    page_content=str(text).strip(),
                    metadata={**metadata, "field": key}
                ))

        subject_tags = json_obj.get("subject_tags", [])
        if subject_tags:
            documents_to_add.append(Document(
                page_content=" ".join(subject_tags),
                metadata={**metadata, "field": "subject_tags"}
            ))

        tags = json_obj.get("tags", [])
        if tags:
            documents_to_add.append(Document(
                page_content=" ".join(tags),
                metadata={**metadata, "field": "tags"}
            ))

    return documents_to_add