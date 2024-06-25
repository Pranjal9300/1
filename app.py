import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(pdf_file)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Load the summarization model
@st.cache(allow_output_mutation=True)
def load_summarizer():
    return pipeline("summarization")

# Title of the app
st.title("PDF Summarizer AI")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from the PDF
    pdf_text = extract_text_from_pdf(uploaded_file)

    # Display extracted text (optional)
    st.subheader("Extracted Text")
    st.write(pdf_text[:2000] + '...')  # Display first 2000 characters for preview

    # Summarize the extracted text
    summarizer = load_summarizer()
    summary = summarizer(pdf_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']

    # Display the summary
    st.subheader("Summary")
    st.write(summary)
