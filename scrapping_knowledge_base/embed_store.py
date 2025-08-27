import os
import json
from tqdm import tqdm
import pickle

from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings  # or HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings

# ============ CONFIG ============
CHUNKS_FILE = "output/loan_chunks.jsonl"
FAISS_DIR = "output/faiss_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"



def build_faiss_index(chunks_file, faiss_dir):
    os.makedirs(faiss_dir, exist_ok=True)

    texts = []
    metadatas = []

    print("[*] Loading chunks...")
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading chunks"):
            record = json.loads(line)
            texts.append(record["text"])
            metadatas.append({
                "chunk_id": record["chunk_id"],
                "url": record["url"],
                "title": record["title"],
                "chunk_index": record["chunk_index"]
            })

    print("[*] Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    print("[*] Building FAISS index with LangChain...")
    vectordb = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)

    #LangChain-compatible FAISS index (creates index.faiss + index.pkl)
    vectordb.save_local(faiss_dir)

    print(f"[✔] LangChain FAISS index saved in {faiss_dir}")


if __name__ == "__main__":
    build_faiss_index(CHUNKS_FILE, FAISS_DIR)
