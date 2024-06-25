import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline

# Function to extract text from a PDF file
def extract_text_from_pdf(file_bytes):
    # Open the PDF from bytes
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Load the summarization model
@st.cache_resource
def load_summarizer():
    return pipeline("summarization")

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
    st.write(pdf_text[:20000] + '...')  # Display first 20000 characters for preview

    # Summarize the extracted text
    summarizer = load_summarizer()
    summary = summarizer(pdf_text, max_length=1500, min_length=30, do_sample=False)[0]['summary_text']

    # Display the summary
    st.subheader("Summary")
    st.write(summary)
