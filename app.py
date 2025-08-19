# Import necessary libraries
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
from bert_score import BERTScorer

# ---- STREAMLIT UI ----
# Set the page configuration for a wide layout
st.set_page_config(page_title="GenAI MVP", page_icon="🤖", layout="wide")
st.title("🧠 GenAI Open-Source Demo - Three Agents on PDF")

# UI elements for file upload and user query
# ADDED INSTRUCTION: Specify that only text-based PDFs should be uploaded
uploaded_file = st.file_uploader("📄 Upload a text-only PDF file", type=["pdf"])
query = st.text_input("💬 Ask your question:")

# ---- Load models with caching ----
# Use st.cache_resource to load heavy models only once
@st.cache_resource
def load_models():
    """Loads and caches all necessary models."""
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    return embedder, qa_model, scorer

# Load all models at the start of the application
embedder, qa_model, scorer = load_models()

# ---- Functions for PDF processing and RAG ----
def load_pdf(file):
    """Extracts text from a PDF file."""
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

if uploaded_file:
    # Load and split the PDF text into chunks
    text = load_pdf(uploaded_file)
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)

    # Create embeddings and a FAISS index for efficient search
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    if query:
        # Encode the user query and search for the most relevant chunks
        q_emb = embedder.encode([query], convert_to_numpy=True)
        D, I = index.search(q_emb, k=2)
        retrieved_chunks = [chunks[i] for i in I[0]]

        # Create three columns for the three agents
        col1, col2, col3 = st.columns(3)

        # Agent 1: Simple Retriever
        with col1:
            st.subheader("🤖 Agent 1 (Retriever)")
            for c in retrieved_chunks:
                st.write(c)

        # Agent 2: Retriever + Question Answering Model
        with col2:
            st.subheader("🤖 Agent 2 (Retriever + QA)")
            # Combine retrieved chunks to form the context
            context = " ".join(retrieved_chunks)
            # Use the QA model to find the answer within the context
            result = qa_model(question=query, context=context)
            st.write(result["answer"])

        # Agent 3: Hallucination Checker
        with col3:
            st.subheader("🤖 Agent 3 (Hallucination Checker)")
            # Set the reference text as the retrieved context
            reference_text = " ".join(retrieved_chunks)
            # Calculate the BERTScore F1 metric for the answer vs. the reference text
            P, R, F1 = scorer.score([result["answer"]], [reference_text])
            
            f1_score = F1.mean().item()
            st.write(f"**Hallucination Score (F1):** {f1_score:.2f}")

            # Provide a simple interpretation of the F1 score
            if f1_score > 0.8:
                st.write("✅ The answer is likely grounded in the context.")
            else:
                st.write("⚠️ The answer may contain information not found in the context.")
