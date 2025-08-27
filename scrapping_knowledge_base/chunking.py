import json
import os
import re
from datetime import datetime

INPUT_FILE = "output/loan_data.jsonl"    
OUTPUT_FILE = "output/loan_chunks.jsonl"

# ----------------- CONFIG -----------------
CHUNK_SIZE = 400  # approx number of words per chunk
OVERLAP = 50      # overlap between chunks (to avoid context loss)

def clean_text(text):
    """Clean up whitespace and weird characters."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=400, overlap=50):
    """Split text into overlapping chunks of ~chunk_size words."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # move forward with overlap
        if start >= len(words):
            break
    return chunks

def process_file(input_file, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    out_f = open(output_file, "w", encoding="utf-8")

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            text = clean_text(record.get("text", ""))
            if not text:
                continue

            # Split into chunks
            chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)

            for i, chunk in enumerate(chunks):
                chunk_record = {
                    "chunk_id": f"{record['id']}_chunk{i+1}",
                    "source_id": record["id"],
                    "url": record.get("url"),
                    "title": record.get("title"),
                    "scrape_date": record.get("scrape_date"),
                    "chunk_index": i+1,
                    "text": chunk
                }
                out_f.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")
    out_f.close()
    print(f"[✔] Chunking complete! Output saved to {output_file}")

if __name__ == "__main__":
    process_file(INPUT_FILE, OUTPUT_FILE)
