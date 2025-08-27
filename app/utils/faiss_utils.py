import os
import pickle
import faiss

def load_faiss_index(faiss_dir):
    """
    Loads a FAISS index and metadata from the given directory.

    Args:
        faiss_dir (str): Directory containing the FAISS index and metadata.

    Returns:
        tuple: (index, texts, metadata) where `index` is the FAISS index, 
               `texts` is a list of texts, and `metadata` is a list of metadata.

    """
    
    index_path = os.path.join(faiss_dir, "index.faiss")
    metadata_path = os.path.join(faiss_dir, "metadata.pkl")

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError("FAISS index or metadata file not found.")

    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        data = pickle.load(f)

    texts, metadata = data["texts"], data["metadata"]
    return index, texts, metadata
