from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_faiss_index(faiss_dir: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load a FAISS index built with LangChain + HuggingFace embeddings.
    
    Args:
        faiss_dir (str): Path to the FAISS index directory.
        model_name (str): HuggingFace embedding model name.
    
    Returns:
        FAISS: A LangChain FAISS vectorstore object.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    vectordb = FAISS.load_local(
        faiss_dir, 
        embeddings, 
        allow_dangerous_deserialization=True
    )

    return vectordb
