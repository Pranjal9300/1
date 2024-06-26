import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline

# Function to extract text from a PDF file
def extract_text_from_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Load the summarization model
@st.cache_resource
def load_summarizer():
    return pipeline("summarization")

# Function to split text into chunks
def split_text_into_chunks(text, chunk_size=1000):
    chunks = []
    while len(text) > chunk_size:
        split_point = text.rfind('.', 0, chunk_size)
        if split_point == -1:
            split_point = chunk_size
        chunks.append(text[:split_point+1])
        text = text[split_point+1:]
    chunks.append(text)
    return chunks

# Title of the app
st.title("PDF Summarizer AI")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Read the file into bytes
    file_bytes = uploaded_file.read()

    # Extract text from the PDF
    pdf_text = extract_text_from_pdf(file_bytes)

    # Display extracted text (optional)
    st.subheader("Extracted Text")
    st.write(pdf_text[:2000] + '...')  # Display first 2000 characters for preview

    # Summarize the extracted text
    summarizer = load_summarizer()
    chunks = split_text_into_chunks(pdf_text)
    summaries = [summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks]
    final_summary = ' '.join(summaries)

    # Display the summary
    st.subheader("Summary")
    st.write(final_summary)
