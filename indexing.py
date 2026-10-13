import io
import os
from typing import Optional
from dotenv import load_dotenv
from pypdf import PdfReader
import docx

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv(override=True)

INDEXES_BASE_DIR = os.path.join("data", "indexes")


def get_embedding_model(api_key: Optional[str] = None) -> GoogleGenerativeAIEmbeddings:
    """Initializes and returns the Google Gemini embeddings model."""
    load_dotenv(override=True)
    key = (api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not key:
        raise ValueError(
            "Gemini API Key not found. Please set GEMINI_API_KEY in your .env file."
        )
    os.environ["GEMINI_API_KEY"] = key
    os.environ["GOOGLE_API_KEY"] = key
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=key,
    )


def extract_text_from_file(uploaded_file) -> str:
    """Extracts raw text from an uploaded PDF or DOC/DOCX file object."""
    filename = uploaded_file.name.lower()
    text = ""

    # Reset stream pointer
    uploaded_file.seek(0)

    if filename.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        pages_text = []
        for i, page in enumerate(reader.pages):
            content = page.extract_text()
            if content:
                pages_text.append(content)
        text = "\n\n".join(pages_text)

    elif filename.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n\n".join(paragraphs)

    elif filename.endswith(".doc"):
        # For legacy .doc, try reading as raw text or fallback
        try:
            raw_bytes = uploaded_file.read()
            text = raw_bytes.decode("utf-8", errors="ignore")
        except Exception:
            text = ""

    if not text.strip():
        raise ValueError(f"Could not extract readable text from '{uploaded_file.name}'.")

    return text


def create_vector_store(
    text: str,
    doc_name: str,
    api_key: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> FAISS:
    """Chunks text, computes embeddings, and builds a FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[{"source": doc_name}],
    )

    if not chunks:
        raise ValueError("Text splitting resulted in zero chunks.")

    embeddings = get_embedding_model(api_key=api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def save_vector_store(vector_store: FAISS, chat_id: str) -> str:
    """Saves the FAISS vector index to the local filesystem for persistent chat recovery."""
    target_dir = os.path.join(INDEXES_BASE_DIR, chat_id)
    os.makedirs(target_dir, exist_ok=True)
    vector_store.save_local(target_dir)
    return target_dir


def load_vector_store(chat_id: str, api_key: Optional[str] = None) -> Optional[FAISS]:
    """Loads a previously persisted FAISS index for a specific chat."""
    target_dir = os.path.join(INDEXES_BASE_DIR, chat_id)
    if not os.path.exists(target_dir):
        return None

    try:
        embeddings = get_embedding_model(api_key=api_key)
        return FAISS.load_local(
            target_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        print(f"Error loading index for chat {chat_id}: {e}")
        return None
