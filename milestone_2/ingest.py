"""
ingest.py - One-time script to build the ChromaDB vector database from knowledge PDFs
Run this ONCE before starting the Streamlit app:
    python ingest.py
"""
import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Always resolve paths relative to this script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")

def build_vector_store():
    print("Scanning knowledge directory for PDFs...")
    documents = []
    
    pdf_files = glob.glob(os.path.join(KNOWLEDGE_DIR, "*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {KNOWLEDGE_DIR}.")
        return

    for pdf_path in pdf_files:
        print(f"Loading {os.path.basename(pdf_path)}...")
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

    print(f"Successfully loaded {len(documents)} pages in total.")

    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,   # Increased chunk size for better semantic grouping from PDFs
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Loading embedding model (this may take a moment)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Building ChromaDB vector store...")
    # Clearing out old DB data by re-initializing and overwriting
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    print(f"Vector store built and saved to '{CHROMA_DB_PATH}'. Done!")

if __name__ == "__main__":
    build_vector_store()
