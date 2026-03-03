# =========================================================
# Contextual PDF Search Assistant (True Exact Sentence Search)
# =========================================================

import streamlit as st
from pypdf import PdfReader
import re

from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Contextual AI Assistant",
    page_icon="🤖",
    layout="wide",
)

st.markdown("""
<style>

/* ------------------ GLOBAL APP STYLE ------------------ */
.stApp {
    background-color: #ffffff;
    font-family: "Inter", "Segoe UI", sans-serif;
}

/* ------------------ MAIN TITLE ------------------ */
h1 {
    color: #0f172a;
    font-weight: 700;
}

h2, h3 {
    color: #1e293b;
}

/* ------------------ SIDEBAR ------------------ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#facc15,#d97706,#92400e);
    box-shadow: inset -5px 0 25px rgba(0,0,0,0.35);
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: #3b1f00 !important;
    font-weight: 500;
}

/* ------------------ INPUT FIELDS ------------------ */
input, textarea {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border-radius: 10px;
    border: 1px solid #cbd5e1 !important;
    padding: 10px !important;
}

/* Placeholder */
input::placeholder, textarea::placeholder {
    color: #94a3b8 !important;
}

/* Focus Effect */
input:focus, textarea:focus {
    border: 1px solid #2563eb !important;
    box-shadow: 0 0 0 2px rgba(37,99,235,0.15);
}

/* ------------------ BUTTONS ------------------ */
.stButton > button {
    background-color: #2563eb;
    color: white;
    font-weight: 600;
    border-radius: 8px;
    padding: 8px 18px;
    border: none;
}

.stButton > button:hover {
    background-color: #1d4ed8;
}

/* ------------------ FILE UPLOADER ------------------ */
[data-testid="stFileUploader"] {
    background-color: #fffaf0;
    border: 2px dashed #a3a3a3;
    padding: 18px;
    border-radius: 12px;
}

/* ------------------ CHAT BOX STYLE ------------------ */
.chat-box {
    background-color: #f8fafc;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #f0ece2;
    box-shadow: 0 4px 10px rgba(0,0,0,0.04);
    color: #0f172a;
    font-size: 17px;
}

/* ------------------ SUCCESS / INFO BOXES ------------------ */
.stSuccess, .stInfo, .stWarning {
    border-radius: 10px;
}

/* ------------------ FOOTER ------------------ */
footer {
    visibility: hidden;
}

</style>
""", 
unsafe_allow_html=True
)

# -----------------------------
# CENTER MAIN TITLE
# -----------------------------
st.markdown(
    """
<h1 style="
    text-align:center;
    font-size:52px;
    letter-spacing:2px;
">
 Contextual AI Assistant
</h1>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<center style="
    font-size:20px;
    font-weight:600;
    color:#0f172a;
    margin-bottom:50px;
    letter-spacing:1.5px;
">
    🤖 AI • Knowledge • Intelligence
</center>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("⚙️ Configuration")

top_k = st.sidebar.slider(
    "Number of Retrieved Chunks",
    min_value=2,
    max_value=10,
    value=4
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
### Features
- 🔐 Offline & Private  
- 📄 Multi-PDF Support  
- 🧠 Semantic Search  
- 🤖 Open-Source LLM  
"""
)

# -----------------------------
# Load Embeddings
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# -----------------------------
# Build Sentence-Level Vector Store
# -----------------------------
@st.cache_resource(show_spinner=True)
def build_vectorstore(files):

    sentence_docs = []

    for uploaded_file in files:
        reader = PdfReader(uploaded_file)

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()

            if text and text.strip():

                # Clean text
                text = text.replace("\n", " ")

                # Split into sentences
                sentences = re.split(r'(?<=[.!?])\s+', text)

                for sentence in sentences:
                    if len(sentence.strip()) > 20:  # ignore very small fragments
                        sentence_docs.append(
                            Document(
                                page_content=sentence.strip(),
                                metadata={
                                    "source": uploaded_file.name,
                                    "page": page_num + 1,
                                },
                            )
                        )

    if not sentence_docs:
        raise ValueError("No readable sentences found in PDFs.")

    embeddings = load_embeddings()

    return FAISS.from_documents(sentence_docs, embeddings)


# -----------------------------
# File Upload
# -----------------------------
uploaded_files = st.file_uploader(
    "📤 Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

# -----------------------------
# Main Logic
# -----------------------------
if uploaded_files:

    with st.spinner("🔮 Building knowledge base..."):
        vectorstore = build_vectorstore(uploaded_files)

    st.success("✅ Knowledge base ready!")
    
    query = st.text_input("💬 Ask a question")

    if query:

        with st.spinner("🤖 Thinking..."):
            results = vectorstore.similarity_search(query, k=top_k)

        st.markdown("###  Answer")
        
        if results:
            best_match = results[0]
            
            st.markdown( f"**📄 {best_match.metadata['source']} – Page {best_match.metadata['page']}**")
            
            st.markdown(
                f"<div class='chat-box'> {best_match.page_content}</div>",
                unsafe_allow_html=True
            )
        
        else:
            st.warning("No relevant content found in documents.")

else:
    st.info("⬆️ Upload one or more PDF files to begin")


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.markdown(
    "<center>⚡ Built with LangChain • FAISS • Transformers • Streamlit</center>",
    unsafe_allow_html=True,
)