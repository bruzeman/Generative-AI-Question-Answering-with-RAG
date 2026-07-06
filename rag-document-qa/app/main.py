import streamlit as st
import sys
import os

# Ensure the 'src' folder is recognizable by Python when running from the root directory
sys.path.append(os.path.abspath('.'))
from src.rag_pipeline import get_rag_chain

# 1. Configure the Streamlit Page
st.set_page_config(
    page_title="Enterprise Document QA", 
    page_icon="📚", 
    layout="centered"
)

st.title("📚 Enterprise Document QA")
st.markdown("""
Welcome to the private document retrieval system. 
This application uses **Retrieval-Augmented Generation (RAG)**, **Pinecone**, and open-source **HuggingFace** models to answer your questions based *strictly* on our internal knowledge base.
""")

# 2. Load and Cache the RAG Chain
@st.cache_resource(show_spinner=False)
def load_chain():
    return get_rag_chain()

with st.spinner("Initializing AI Models and connecting to Vector Database..."):
    try:
        chain = load_chain()
    except Exception as e:
        st.error(f"Critical Error initializing the RAG pipeline: {e}")
        st.stop()

# 3. Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I have reviewed your documents. What would you like to know?"}
    ]

# 4. Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Handle User Input
if user_query := st.chat_input("Ask a question about your documents..."):
    
    # Display user's question
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Generate and display the AI's response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base and generating answer..."):
            try:
                # Pass the query to our backend pipeline
                response = chain.invoke({"input": user_query})
                answer = response.get("answer", "Sorry, I could not generate an answer.")
                
                # Display the answer
                st.markdown(answer)
                
                # Save the answer to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"An error occurred during retrieval: {e}")