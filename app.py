import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import faiss
import numpy as np

st.title("GraphRAG Demo (Cloud Version)")

@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()

def extract_pdf_text(file):
    reader = PdfReader(file)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text() or ""
        full_text += text + "\n"
    return full_text

def chunk_text(text, size=500, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap
    )
    return splitter.split_text(text)

def build_faiss(chunks):
    embeddings = model.encode(chunks, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index, embeddings

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    st.info("Extracting text...")
    text = extract_pdf_text(uploaded_file)

    st.info("Chunking text...")
    chunks = chunk_text(text)

    st.info("Creating embeddings + FAISS index...")
    index, embeddings = build_faiss(chunks)

    st.success("Document indexed! Ask a question below.")

    query = st.text_input("Your question:")

    if query:
        q_emb = model.encode([query])
        scores, ids = index.search(np.array(q_emb).astype("float32"), 3)
        results = [chunks[i] for i in ids[0]]

        st.subheader("Top Results:")
        for r in results:
            st.write(r)
            st.write("---")
