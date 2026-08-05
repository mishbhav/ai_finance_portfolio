from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = "chroma_db"
# Loaded once when this module is first imported, reused on every call
_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
_vector_store = Chroma(persist_directory=PERSIST_DIR, embedding_function=_embeddings)

def get_retriever(k: int = 3):
    """Loads the persisted index and returns a retriever object
    that supports .invoke(query) -> list of relevant chunks."""
    return _vector_store.as_retriever(search_kwargs={"k": k})

def search(query: str, k: int = 3) -> list[str]:
    """Convenience wrapper: returns just the text content of the
    top-k relevant chunks for a given query string."""
    retriever = get_retriever(k=k)
    results = retriever.invoke(query)
    return [doc.page_content for doc in results]    