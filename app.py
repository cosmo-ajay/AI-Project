import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # Corrected Import
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI  # Corrected Import
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Set API Key for Google Generative AI
os.environ["GOOGLE_API_KEY"] = "AIzaSyCeWYwsdLuoue4tqdG-y3TyIKLloug0Q-s"  # Replace with your actual API key

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
            else:
                st.warning(f"Text extraction failed for a page in {pdf.name}. The PDF may be image-based or unreadable.")
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

# Function to create vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Fixed embedding model
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

# Function to set up conversational retrieval chain
def get_conversational_chain(vector_store):
    llm = GoogleGenerativeAI(model="gemini-pro")  # Updated model
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to handle user input and generate responses
def user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        st.write("Human: " if i % 2 == 0 else "Bot: ", message.content)

# Main function to set up Streamlit UI
def main():
    st.set_page_config(page_title="DocuQuery: AI-Powered PDF Knowledge Assistant")
    st.header("DocuQuery: AI-Powered PDF Knowledge Assistant")

    # Initialize session state variables
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar for document upload
    with st.sidebar:
        st.title("Settings")
        st.subheader("Upload your Documents")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Process Button", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Extract text from PDFs
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text:
                    st.error("No text could be extracted from the uploaded PDFs. Please upload valid PDFs.")
                else:
                    # Split text into chunks
                    text_chunks = get_text_chunks(raw_text)
                    # Create vector store
                    vector_store = get_vector_store(text_chunks)
                    # Set up conversational chain
                    st.session_state.conversation = get_conversational_chain(vector_store)
                    st.success("Processing Done!")

    # User input for questions
    user_question = st.text_input("Ask a question from the PDF files:")
    if user_question:
        if 'conversation' not in st.session_state or st.session_state.conversation is None:
            st.warning("Please upload and process documents first.")
        else:
            user_input(user_question)

# Run the application
if __name__ == "__main__":
    main()  # Removed the unnecessary period