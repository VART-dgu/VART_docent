import csv
import pandas as pd
import os
import glob
from math import ceil
import threading
import queue
import requests
import sys

from app.services.vector_store import save_to_faiss
from app.services.image_processor import describe_images
import time
from datetime import datetime
# Paths and constants
csv_path = "/home/jy/dataset/classes.csv"
wclass_csv_path = "/home/jy/dataset/wclasses.csv"
image_base_path = "/home/jy/dataset/"
museum_id = "museum100"
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

def describe_images_only(batch, museum_id, llama_url, llava_url, batch_idx):
    """
    Process exactly one batch of up to 30 artworks:
    - Call describe_images on that batch
    - Save returned Documents to a FAISS index named batch_{batch_idx}.faiss
    - Log start and completion
    Returns the list of Document objects for this batch.
    """
    # Log start of this batch
    log(f"[Batch {batch_idx+1}] Starting processing of {len(batch)} artworks using {llama_url}.")

    # Start timing
    start_time = time.time()

    # Call the model for this batch
    documents = describe_images(batch, museum_id, llama_url=llama_url, llava_url=llava_url)
    
    if not documents:
        log(f"[Batch {batch_idx+1}] Failed to generate documents. Skipping FAISS save.")
        return 

    # Save to FAISS for this batch
    index_path = os.path.join(base_dir, f"batch_{batch_idx + 1}.faiss")
    save_to_faiss(documents, llama_url, index_path=index_path)

    # Log completion
    log(f"[Batch {batch_idx+1}] Completed. Saved {len(documents)} documents to {index_path}.")
    duration = (time.time() - start_time) / 60  # duration in minutes
    log(f"[Batch {batch_idx+1}] Duration: {duration:.2f} minutes.")


def merge_all_indices():
    """
    After all batch_{i}.faiss files exist in base_dir,
    merge them into a single final_index.faiss under base_dir.
    """
    from langchain_community.vectorstores.faiss import FAISS
    from langchain_ollama import OllamaEmbeddings

    embeddings = OllamaEmbeddings(model="llama3.3:70b", base_url=llama_urls[0])
    index_files = sorted(
        glob.glob(os.path.join(base_dir, "batch_*.faiss")),
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    if not index_files:
        log("[WARN] No FAISS index files found to merge.")
        return

    # Load first index as base
    log(f"[Merge] Loading base index from {index_files[0]}")
    merged_db = FAISS.load_local(index_files[0], embeddings, allow_dangerous_deserialization=True)

    # Merge each subsequent index
    for idx_path in index_files[1:]:
        log(f"[Merge] Merging index {idx_path}")
        db = FAISS.load_local(idx_path, embeddings, allow_dangerous_deserialization=True)
        merged_db.merge_from(db)

    # Save merged index
    final_path = os.path.join(base_dir, "final_index.faiss")
    merged_db.save_local(final_path)
    log(f"[Merge] Merged all indices into {final_path}")

def main():
    # 1) Check Ollama servers before starting
    for i, llama_url in enumerate(ollama_urls):
        try:
            response = requests.get(f"{llama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                log(f"[INFO] Ollama-{i+1} is up at {llama_url}.")
            else:
                log(f"[ERROR] Ollama-{i+1} responded with {response.status_code}. Exiting.")
                sys.exit(1)
        except requests.RequestException as e:
            log(f"[ERROR] Ollama-{i+1} at {llama_url} is not reachable: {e}. Exiting.")
            sys.exit(1)

    # 2) Build full list of artworks based on whitelist
    all_test_artworks = prepare_test_artworks()

    # 3) Split all_test_artworks into batches of size 30
    batches = [
        all_test_artworks[i : i + batch_size]
        for i in range(0, len(all_test_artworks), batch_size)
    ]

    # 4) Process each batch using per-thread queues for pinned assignment
    queues = [queue.Queue() for _ in range(num_threads)]

    # Worker function: each thread handles its own queue
    def worker(thread_idx, task_queue):
        llama_url = ollama_urls[thread_idx % len(ollama_urls)]
        llava_url = llama_url
        while True:
            item = task_queue.get()
            if item is None:
                task_queue.task_done()
                break
            batch, idx = item
            index_path = os.path.join(base_dir, f"batch_{idx + 1}.faiss")
            # Skip if already exists
            if not (os.path.isdir(index_path) and
                    os.path.isfile(os.path.join(index_path, "index.faiss")) and
                    os.path.isfile(os.path.join(index_path, "index.pkl"))):
                describe_images_only(batch, museum_id, llama_url, llava_url, idx)
            task_queue.task_done()

    # Start worker threads
    threads = []
    for t_idx in range(num_threads):
        t = threading.Thread(target=worker, args=(t_idx, queues[t_idx]), daemon=True)
        t.start()
        threads.append(t)

    # Dispatch batches to queues in round-robin
    for idx, batch in enumerate(batches):
        q_idx = idx % num_threads
        queues[q_idx].put((batch, idx))

    # Send termination signal
    for q in queues:
        q.put(None)

    # Wait for all queues to be processed
    for q in queues:
        q.join()

    # 5) After all batches are done, merge batch-level FAISS indexes
    log("FAISS DB creation completed. Starting merge of batch indexes.")
    merge_all_indices()

if __name__ == "__main__":
    #merge_all_indices()
    main()