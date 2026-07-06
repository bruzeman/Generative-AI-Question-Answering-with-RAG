import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings 

# Load API keys from the .env file
load_dotenv()

def ingest_and_chunk_documents(data_directory: str):
    """
    Loads all PDF documents from a specified directory and splits them into text chunks.
    """
    print(f"Scanning directory '{data_directory}' for PDF documents...")
    loader = PyPDFDirectoryLoader(data_directory)
    documents = loader.load()
    
    if not documents:
        print(f"Error: No PDF documents found in {data_directory}.")
        return None
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    
    document_chunks = text_splitter.split_documents(documents)
    print(f"Successfully chunked documents into {len(document_chunks)} pieces.")
    return document_chunks

def setup_vector_database(chunks, index_name: str):
    """
    Initializes Pinecone, creates an index, and uploads HuggingFace embeddings.
    """
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Creating new Pinecone index: '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=384, # CHANGED: HuggingFace MiniLM uses 384 dimensions
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print("Pinecone index provisioned successfully.")
    else:
        print(f"Pinecone index '{index_name}' already exists. Proceeding to upload.")

    # CHANGED: Initialize free HuggingFace Embeddings instead of OpenAI
    print("Initializing HuggingFace Embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Converting text to embeddings and uploading to the vector database. This may take a moment...")
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        index_name=index_name,
        embedding=embeddings
    )
    
    print("Vector upload complete.")
    return vectorstore

if __name__ == "__main__":
    RAW_DATA_PATH = "data/raw"
    # CHANGED: We use a new index name so Pinecone doesn't conflict with your old 1536-dimension index
    INDEX_NAME = "rag-document-qa-hf" 
    
    if not os.environ.get("PINECONE_API_KEY"):
        print("Critical Error: Missing Pinecone API key.")
        exit(1)
    
    chunks = ingest_and_chunk_documents(RAW_DATA_PATH)
    
    if chunks:
        vectorstore = setup_vector_database(chunks, index_name=INDEX_NAME)
        print("\n--- Step 3 Complete ---")
        print("Your private data is now embedded using HuggingFace and securely stored in Pinecone.")