from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# Load environment variables
load_dotenv()

# Set your OpenAI API key here
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set page configuration
st.set_page_config(
    
    
    page_title="AI PDF Reader",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("📄 AI PDF Reader")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
extracted_text, splitted_text, text_splitter = "", [], None

if uploaded_file is not None:
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    extracted_text = text

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )  
    splitted_text = text_splitter.split_text(extracted_text)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_texts(splitted_text, embeddings)
    question = st.text_input("Ask a question based on the extracted text:")
    
    # Submit button
    if st.button("Submit"):
        if uploaded_file is None:
            st.error("Please upload a PDF file to proceed.")
        elif extracted_text == "":
            st.error("The uploaded file is not a valid PDF or it could not be processed.")
        elif question.strip() == "":
            st.error("Please enter a valid question.")
        else:
            try:
                docs = vector_store.similarity_search(question)
                llm = OpenAI(api_key=openai_api_key)
                chain = load_qa_chain(llm, chain_type="stuff")            
                response = chain.run(input_documents=docs, question=question)           
                st.success(response)
            except Exception as e:
                st.error(f"An error occurred while getting the answer: {e}")
