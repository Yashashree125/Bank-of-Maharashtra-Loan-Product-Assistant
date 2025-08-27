from langchain_community.embeddings import HuggingFaceEmbeddings

def create_embeddings(model_name: str):
    """
    Initialize HuggingFace embeddings for LangChain.
    """
    return HuggingFaceEmbeddings(model_name=model_name)

def search_faiss(query, vectordb, top_k=5):
    """
    Retrieve top-k docs using LangChain retriever.
    """
    retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
    results = retriever.get_relevant_documents(query)
    return results
