import streamlit as st
from sentence_transformers import SentenceTransformer

st.title("Test Deployment")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

st.write("Model loaded successfully!")
