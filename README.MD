# AI PDF Reader

## Overview

Welcome to the AI PDF Reader! This application allows you to effortlessly extract text from PDF files and interact with it using advanced AI-powered question-answering capabilities. By leveraging state-of-the-art natural language processing (NLP) models, the AI PDF Reader provides quick and accurate answers to your questions based on the content of your uploaded PDFs.

## Features

- **Easy PDF Upload**: Upload your PDF files through a simple file uploader interface.
- **Text Extraction**: Extract text from every page of the uploaded PDF.
- **Text Splitting**: Split the extracted text into manageable chunks for better processing.
- **Embeddings Generation**: Create embeddings for text chunks using OpenAI Embeddings.
- **Vector Store**: Store and manage text embeddings using FAISS.
- **Question Answering**: Ask questions based on the extracted text and receive answers generated by an OpenAI language model.

## How It Works

1. **Environment Setup**: Load environment variables using `dotenv`.
2. **API Key Configuration**: Set your OpenAI API key from the environment variables.
3. **Streamlit Configuration**: Configure the Streamlit app settings.
4. **PDF Upload**: Use Streamlit's file uploader to select and upload a PDF file.
5. **Text Extraction**: Extract text from each page of the uploaded PDF using PyPDF2.
6. **Text Splitting**: Split the extracted text into chunks with specified chunk size and overlap using LangChain's `CharacterTextSplitter`.
7. **Embeddings Creation**: Generate embeddings for the text chunks using `OpenAIEmbeddings`.
8. **Vector Storage**: Store the text embeddings in a FAISS vector store.
9. **Question Input**: Input your question through a text input field.
10. **Answer Generation**: Use LangChain's question-answering chain to generate answers based on the similarity search results from the vector store.

## Installation

To run the AI PDF Reader locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ai-pdf-reader.git
   cd ai-pdf-reader
2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
4. Set up your OpenAI API key:
    **Create a .env file in the root directory of the project.**
    **Add your OpenAI API key to the .env file:**
    ```bash
    OPENAI_API_KEY=your_openai_api_key
5. Run the Streamlit app:
    ```bash
    streamlit run app.py

requirements.txt
Make sure to include the following dependencies in your requirements.txt file:
    ```bash
    python-dotenv
    streamlit
    PyPDF2
    langchain
    openai
    faiss-cpu

Usage
1. Upload a PDF: Click on the file uploader widget and select a PDF file from your local system.
2. Ask a Question: Enter a question related to the content of the PDF in the text input field.
3. Get Answers: Click the "Submit" button to get answers generated by the AI based on the extracted text from the PDF.


Contributing
We welcome contributions to enhance the functionality and performance of the AI PDF Reader. To contribute, please fork the repository and create a pull request with your changes. Make sure to follow the project's coding standards and include detailed commit messages.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Contact
For any inquiries or feedback, please reach out to your-email@example.com.