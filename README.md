DocuQuery: AI-Powered PDF Knowledge Assistant
DocuQuery is an AI-powered application designed to help users extract and query information from PDF documents. It uses advanced Natural Language Processing (NLP) and Machine Learning (ML) techniques to provide accurate and context-aware responses to user queries. Built with Streamlit, Google Generative AI, and LangChain, DocuQuery simplifies the process of knowledge extraction from PDFs.

Features
PDF Upload: Upload multiple PDF files for processing.

Text Extraction: Extract text from uploaded PDFs.

AI-Powered Querying: Ask questions in natural language and get answers based on the PDF content.

Conversational Interface: Maintains chat history for each session.

Vector Embeddings: Uses Hugging Face Embeddings and FAISS for efficient text processing.

Technologies Used
Streamlit: For the front-end user interface.

PyPDF2: For extracting text from PDFs.

LangChain: For managing conversational chains and memory.

Google Generative AI: For generating responses using the Gemini-Pro model.

FAISS: For creating and managing vector stores.

Hugging Face Embeddings: For generating text embeddings.

Sentence Transformers: For text similarity and embeddings.

Installation
Prerequisites
Python 3.8 or higher.

A Google API key for using the Google Generative AI model.

Steps
Clone the repository:

bash
Copy
git clone https://github.com/your-username/DocuQuery.git
cd DocuQuery
Install the required dependencies:

bash
Copy
pip install -r requirements.txt
Set up your Google API key:

Replace the placeholder in app.py with your actual Google API key:

python
Copy
os.environ["GOOGLE_API_KEY"] = "your-api-key-here"
Run the application:

bash
Copy
streamlit run app.py
Open your browser and navigate to http://localhost:8501 to access the application.

Usage
Upload PDFs:

Click on the "Upload your PDF Files" button in the sidebar.

Select one or more PDF files to upload.

Process PDFs:

Click the "Process" button to extract text and create a vector store.

Ask Questions:

Enter your question in the text input box and press Enter.

The application will generate a response based on the content of the uploaded PDFs.
