from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CORPUS_DIR = "retrieval/corpus"
PERSIST_DIR = "chroma_db"

def build_index():
    # 1. Load all .txt files from CORPUS_DIR into LangChain Document objects
    loader = DirectoryLoader(CORPUS_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")

    # 2. Split them with RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    ) 
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # 3. Embed with HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
    # 4. Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
    print(f"Index persisted to {PERSIST_DIR}")

if __name__ == "__main__":
    build_index()