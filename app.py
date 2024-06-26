import streamlit as st
import fitz  # PyMuPDF
import re
from transformers import pipeline
from nltk.tokenize import sent_tokenize

# Download NLTK resources
import nltk
nltk.download('punkt')

# Function to extract text and headings from a PDF file
def extract_text_and_headings_from_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    headings = []
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block['type'] == 0:  # text block
                for line in block['lines']:
                    span = line['spans'][0]
                    if span['size'] > 12:  # assuming headings have larger font size
                        headings.append((span['text'], page_num))
                    text += span['text'] + " "
    return text, headings

# Load the summarization model
@st.cache_resource
def load_summarizer():
    return pipeline("summarization")

# Function to summarize text
def summarize_text(text, summarizer, max_chunk_size=1000):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)

    summaries = [summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks]
    return ' '.join(summaries)

# Title of the app
st.title("PDF Summarizer AI")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Read the file into bytes
    file_bytes = uploaded_file.read()

    # Extract text and headings from the PDF
    pdf_text, headings = extract_text_and_headings_from_pdf(file_bytes)

    # Display extracted headings with page numbers
    st.subheader("Extracted Headings")
    for heading, page_num in headings:
        st.write(f"{heading} (Page {page_num})")

    # Summarize the extracted text
    summarizer = load_summarizer()
    summary = summarize_text(pdf_text, summarizer)

    # Display the summary
    st.subheader("Summary")
    st.write(summary)
