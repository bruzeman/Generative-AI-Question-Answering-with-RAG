import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_pinecone import PineconeVectorStore

# UPDATED IMPORTS:
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

def get_rag_chain():
    """
    Constructs the Retrieval-Augmented Generation pipeline connecting Pinecone and HuggingFace.
    """
    # 1. Initialize Embeddings (Must exactly match what we used in ingestion.py)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Connect to our HuggingFace-specific Pinecone index
    index_name = "rag-document-qa-hf"
    
    if not os.environ.get("PINECONE_API_KEY") or not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        raise ValueError("Missing API Keys. Please check your .env file.")

    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    
    # Configure the retriever to fetch the top 3 most relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. Initialize the free HuggingFace LLM
    # Qwen2.5-7B-Instruct is served by several providers (Together, Fireworks, etc.),
    # unlike the -1M variant which was only hosted by Featherless AI.
    # provider="auto" tells HF to route across ALL enabled providers on the account
    # instead of defaulting to just "hf-inference", which doesn't host every model.
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        provider="auto",
        max_new_tokens=512,
        temperature=0.3
    )
    llm = ChatHuggingFace(llm=llm_endpoint)

    # 4. Construct the Prompt Template
    # This strictly commands the LLM to only use the retrieved context.
    system_prompt = (
        "You are a professional assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, explicitly say that you don't know. "
        "Keep the answer concise and strictly based on the context.\n\n"
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 5. Build the LangChain RAG Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

if __name__ == "__main__":
    print("Initializing Open-Source RAG Pipeline...")
    chain = get_rag_chain()
    print("Pipeline Ready! Let's test it out.\n")
    
    # Change this query to something relevant to the PDF you uploaded!
    test_query = "What is the main topic of this document?"
    
    print(f"User Query: {test_query}")
    print("Fetching context and generating answer...\n")
    
    response = chain.invoke({"input": test_query})
    
    print("--- AI Response ---")
    print(response["answer"])
    print("-------------------")
    print("\nStep 4 complete. The backend is fully functional.")