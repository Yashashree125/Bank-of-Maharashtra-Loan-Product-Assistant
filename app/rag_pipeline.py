from sentence_transformers import SentenceTransformer
from .utils.faiss_utils import load_faiss_index
from .utils.llm_utils import generate_answer

def embed_query(model, query: str):
    """
    Searches the FAISS index for the most relevant documents based on the query.

    Args:
        query (str): The search query.
        index (faiss.Index): The FAISS index to search in.
        texts (list): List of texts corresponding to the index entries.
        metadata (list): List of metadata corresponding to the index entries.
        model (SentenceTransformer): Pre-trained model to embed the query.
        top_k (int): Number of top results to return (default is 5).

    Returns:
        list: A list of dictionaries containing the top results, including rank, text, metadata, and score.

    """
    return model.encode([query], convert_to_numpy=True)

def search_faiss(query, index, texts, metadata, model, top_k=5):
    query_vec = embed_query(model, query)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "rank": i+1,
            "text": texts[idx],
            "metadata": metadata[idx],
            "score": float(distances[0][i])
        })
    return results
