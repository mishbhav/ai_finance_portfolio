from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = "chroma_db"

# Loaded once when this module is first imported, reused on every call
_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
_vector_store = Chroma(persist_directory=PERSIST_DIR, embedding_function=_embeddings)


def get_retriever(k: int = 3):
    """Returns a retriever object over the persisted index,
    reusing the already-loaded embeddings/vector store."""
    return _vector_store.as_retriever(search_kwargs={"k": k})


def search(query: str, k: int = 3) -> list[str]:
    """Convenience wrapper: returns just the text content of the
    top-k relevant chunks for a given query string. Kept for backward
    compatibility with any caller that only needs text."""
    retriever = get_retriever(k=k)
    results = retriever.invoke(query)
    return [doc.page_content for doc in results]


def search_with_metadata(query: str, k: int = 3) -> list[dict]:
    """Like search(), but preserves each chunk's metadata (ticker,
    source, published date, type) — needed by anything that cares
    about evidence recency, e.g. citation freshness scoring in the judge."""
    retriever = get_retriever(k=k)
    results = retriever.invoke(query)
    return [
        {"text": doc.page_content, "metadata": doc.metadata}
        for doc in results
    ]