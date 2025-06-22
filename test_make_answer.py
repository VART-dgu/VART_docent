import csv
import pandas as pd
import os
from datetime import datetime
import requests
import sys
import threading
import queue
import time

from app.services.rag_chain import make_answer
from app.services.vector_store import load_faiss

csv_path = "/home/jy/dataset/classes.csv"
wclass_csv_path = "/home/jy/dataset/wclasses2.csv"
image_base_path = "/home/jy/dataset/"
llama_urls = ["http://ollama-1:11434", "http://ollama-2:11434", "http://210.94.179.18:9861", "http://210.94.179.18:9862", "http://210.94.179.18:9863"]
#llama_urls = []
#ollama_urls = ["http://ollama-2:11434", "http://ollama-3:11434", "http://210.94.179.18:9862", "http://210.94.179.18:9863"]
llava_urls = ["http://192.168.2.18:9760", "http://192.168.2.18:9761", "http://192.168.2.18:9762", "http://192.168.2.18:9763", "http://210.94.179.18:9661", "http://210.94.179.18:9662", "http://210.94.179.18:9663", "http://210.94.179.18:9664"]
#llava_urls = ["http://192.168.2.18:9760", "http://192.168.2.18:9761", "http://192.168.2.18:9762", "http://192.168.2.18:9763"]
ollama_urls = llama_urls + llava_urls
batch_size = 8
num_threads = len(ollama_urls)
base_dir = "mmrag_faiss"
log_path = os.path.join(base_dir, "processing.log")

# Ensure base directory exists
os.makedirs(base_dir, exist_ok=True)

def log(message: str):
    """Print to console and append to a log file with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    with open(log_path, "a") as lf:
        lf.write(full_message + "\n")
def prepare_test_artworks():
    test_artworks = []
    artwork_counter = 0

    # Load wclasses.csv
    wclass_df = pd.read_csv(wclass_csv_path)

    # Build a dictionary from csv_path mapping filename to (description, artist_name)
    desc_artist_map = {}
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            desc_artist_map[row["filename"]] = (row["description"], row["artist"].strip())

    for _, row in wclass_df.iterrows():
        filename = row["file"]
        image_url = os.path.join(image_base_path, filename)
        author_id = row["artist"]
        description, author_name = desc_artist_map.get(filename, ("", "unknown_artist"))
        description = description.replace("-", " ")

        test_artworks.append({
            "artwork_id": f"art{artwork_counter}",
            "image_url": image_url,
            "title": filename.split("/")[-1].split(".")[0].replace("-", " "),
            "description": description,
            "author_id": author_id,
            "author_name": author_name,
            "author_description": "No description provided."
        })
        artwork_counter += 1
    return test_artworks

def main():
    db = load_faiss(ollama_urls[0])
    all_test_artworks = prepare_test_artworks()

    target_artwork_id = {"art1": "이 작품의 아래와 위에 있는게 각각 어떤 것들을 의미해?", "art2": "이 작품은 작가가 무엇을 표현하고 싶었던 걸까?", "art3": "작품에서 초원에 앉은 소녀는 무엇을 보고 있는걸까?", "art8": "이 작품에는 어떤 의미가 담겨있어?"}
    for art in all_test_artworks:
        artwork_id = art["artwork_id"]
        if artwork_id not in target_artwork_id.keys():
            continue
        matched_docs = [
            doc for doc in db.docstore._dict.values()
            if doc.metadata.get("artwork_id") == artwork_id and doc.metadata.get("field") == "full"
        ]
        doc_metadata = matched_docs[0].metadata
        metadata = {
            "title": doc_metadata["title"],
            "artwork_id": doc_metadata["artwork_id"],
            "author_id": doc_metadata["author_id"],
            "author_name": doc_metadata["author_name"],
            "museum_id": doc_metadata["museum_id"]
        }

        print(make_answer(question=target_artwork_id[artwork_id], metadata=metadata, ollama_url=ollama_urls[0]))
 
        


if __name__ == "__main__":
    main()