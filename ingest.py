"""
ingest.py - One-time script to build the ChromaDB vector database from guidelines.txt
Run this ONCE before starting the Streamlit app:
    python ingest.py
"""
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Always resolve paths relative to this script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GUIDELINES_PATH = os.path.join(BASE_DIR, "guidelines.txt")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")

def build_vector_store():
    print("Loading guidelines...")
    loader = TextLoader(GUIDELINES_PATH, encoding="utf-8")
    documents = loader.load()

    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Loading embedding model (this may take a moment)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Building ChromaDB vector store...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    vectorstore.persist()
    print(f"Vector store built and saved to '{CHROMA_DB_PATH}'. Done!")

if __name__ == "__main__":
    build_vector_store()
