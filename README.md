rag-chatbot
RAG Chatbot
This repository contains a Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF or Excel files and ask questions related to the content of the uploaded documents. The chatbot uses advanced natural language processing techniques to retrieve relevant information from the documents and generate comprehensive answers.

Features
Document Upload: Supports uploading PDF and Excel files.
Document Processing: Extracts text from uploaded documents and splits it into manageable chunks.
Vector Store: Uses ChromaDB to store and query document embeddings.
Question Answering: Utilizes the Ollama model to generate answers based on the retrieved document context.
Streamlit Interface: Provides a user-friendly interface for uploading documents and interacting with the chatbot.
Installation
To install the required dependencies, run:

pip install -r requirements.txt
