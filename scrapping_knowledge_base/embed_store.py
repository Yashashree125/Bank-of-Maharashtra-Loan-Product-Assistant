import os
import json
import faiss
import pickle
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ============ CONFIG ============
CHUNKS_FILE = "output/loan_chunks.jsonl"
FAISS_DIR = "output/faiss_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# =================================
def embed_texts(model, texts):
    """Generate embeddings for a list of texts."""
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

def build_faiss_index(chunks_file, faiss_dir):
    os.makedirs(faiss_dir, exist_ok=True)

    texts = []
    metadata = []

    print("[*] Loading chunks...")
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading chunks"):
            record = json.loads(line)
            texts.append(record["text"])
            metadata.append({
                "chunk_id": record["chunk_id"],
                "url": record["url"],
                "title": record["title"],
                "chunk_index": record["chunk_index"]
            })

    print("[*] Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = embed_texts(model, texts)

    dim = embeddings.shape[1]
    print(f"[*] Building FAISS index of dimension {dim}...")

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(faiss_dir, "index.faiss"))
    with open(os.path.join(faiss_dir, "metadata.pkl"), "wb") as f:
        pickle.dump({"texts": texts, "metadata": metadata}, f)

    print(f"[✔] FAISS index and metadata saved in {faiss_dir}")

if __name__ == "__main__":
    build_faiss_index(CHUNKS_FILE, FAISS_DIR)
