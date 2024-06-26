import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer

# Directly include the punkt data
punkt_data = """
<...>  # Paste the content of the punkt data file here
"""

# Initialize the tokenizer
tokenizer = PunktSentenceTokenizer(punkt_data)

# Function to check if a font is bold
def is_bold(font):
    bold_indicators = ['bold', 'Bold', 'BOLD']
    return any(indicator in font for indicator in bold_indicators)

# Function to extract text and headings from a specific page of a PDF file
def extract_text_and_headings_from_pdf(file_bytes, page_number):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    if page_number < 1 or page_number > len(doc):
        st.error("Invalid page number")
        return "", []
    text = ""
    headings = []
    page = doc[page_number - 1]
    blocks = page.get_text("dict")["blocks"]
    for block in blocks:
        if block['type'] == 0:  # text block
            for line in block['lines']:
                for span in line['spans']:
                    font = span['font']
                    if is_bold(font) or span['size'] > 12:  # assuming headings have larger font size or are bold
                        headings.append((span['text'], page_number))
                    text += span['text'] + " "
    return text, headings

# Load the summarization model
@st.cache_resource
def load_summarizer():
    return pipeline("summarization")

# Function to summarize text
def summarize_text(text, summarizer, max_chunk_size=1000):
    sentences = tokenizer.tokenize(text)
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
    # Input for page number
    page_number = st.number_input("Enter the page number to process", min_value=1, value=1, step=1)

    # Read the file into bytes
    file_bytes = uploaded_file.read()

    # Extract text and headings from the specified page of the PDF
    pdf_text, headings = extract_text_and_headings_from_pdf(file_bytes, page_number)

    if pdf_text:
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
