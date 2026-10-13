import json
import os
import time
import uuid
import streamlit as st
from dotenv import load_dotenv

from indexing import (
    extract_text_from_file,
    create_vector_store,
    save_vector_store,
    load_vector_store,
)
from retrieval import generate_answer_stream

# Load environment variables
load_dotenv(override=True)
active_api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
if active_api_key:
    os.environ["GEMINI_API_KEY"] = active_api_key
    os.environ["GOOGLE_API_KEY"] = active_api_key

# Directories for persistence
CHATS_DIR = os.path.join("data", "chats")
os.makedirs(CHATS_DIR, exist_ok=True)

# Streamlit Page Setup
st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for ChatGPT-like sleek aesthetic
st.markdown(
    """
    <style>
    /* Main container and typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Clean chat header */
    .chat-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.75rem 0rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1.5rem;
    }
    .doc-badge {
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.35);
        color: #818cf8;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 500;
    }

    /* Sidebar chat items */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        text-align: left;
        font-weight: 500;
        transition: all 0.15s ease-in-out;
    }
    
    /* Chat bubbles styling */
    .stChatMessage {
        border-radius: 12px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Helper Functions for Chat Persistence
def list_saved_chats():
    """Lists all saved chat sessions sorted by modification time (most recent first)."""
    if not os.path.exists(CHATS_DIR):
        return []
    chat_files = [f for f in os.listdir(CHATS_DIR) if f.endswith(".json")]
    chats = []
    for f in chat_files:
        path = os.path.join(CHATS_DIR, f)
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
                mtime = os.path.getmtime(path)
                chats.append({
                    "id": data.get("id", f.replace(".json", "")),
                    "title": data.get("title", "Untitled Chat"),
                    "doc_name": data.get("doc_name", "No document"),
                    "mtime": mtime,
                })
        except Exception:
            continue
    chats.sort(key=lambda x: x["mtime"], reverse=True)
    return chats


def save_chat_to_disk(chat_id: str, title: str, doc_name: str, messages: list):
    """Saves conversation data and messages to JSON file."""
    filepath = os.path.join(CHATS_DIR, f"{chat_id}.json")
    data = {
        "id": chat_id,
        "title": title,
        "doc_name": doc_name,
        "messages": messages,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_chat_from_disk(chat_id: str):
    """Loads chat conversation data from JSON file."""
    filepath = os.path.join(CHATS_DIR, f"{chat_id}.json")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def delete_chat(chat_id: str):
    """Deletes chat JSON and its persisted vector index."""
    filepath = os.path.join(CHATS_DIR, f"{chat_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
    # Also remove vector index folder if present
    index_path = os.path.join("data", "indexes", chat_id)
    if os.path.exists(index_path):
        import shutil
        shutil.rmtree(index_path, ignore_errors=True)


def init_new_chat():
    """Initializes a fresh chat state."""
    new_id = f"chat_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    st.session_state.chat_id = new_id
    st.session_state.chat_title = "New Conversation"
    st.session_state.doc_name = None
    st.session_state.messages = []
    st.session_state.vector_store = None


# Initialize Session State
if "chat_id" not in st.session_state:
    init_new_chat()

# Sidebar: ChatGPT-style Navigation & Document Uploader
with st.sidebar:
    st.markdown("### 💬 **RAG Chatbot**")


    # "+ New Chat" Button
    if st.button("➕ **New Chat**", use_container_width=True, type="primary"):
        init_new_chat()
        st.rerun()

    st.markdown("---")

    # Document Uploader
    st.markdown("#### 📄 **Document Upload**")
    uploaded_file = st.file_uploader(
        "Upload PDF, DOC, or DOCX",
        type=["pdf", "docx", "doc"],
        help="Upload a document to index for this conversation.",
        key=f"uploader_{st.session_state.chat_id}",
    )

    if uploaded_file is not None and (st.session_state.doc_name != uploaded_file.name):
        if not active_api_key:
            st.error("⚠️ `GEMINI_API_KEY` is empty in `.env`. Please paste your API key in `.env` and save (`Ctrl+S`).")
        else:
            with st.spinner("Extracting text and building vector index..."):
                try:
                    raw_text = extract_text_from_file(uploaded_file)
                    vector_store = create_vector_store(
                        text=raw_text,
                        doc_name=uploaded_file.name,
                        api_key=active_api_key,
                    )
                    # Persist index for this chat
                    save_vector_store(vector_store, st.session_state.chat_id)

                    st.session_state.vector_store = vector_store
                    st.session_state.doc_name = uploaded_file.name

                    # Auto title if untitled
                    if st.session_state.chat_title == "New Conversation":
                        st.session_state.chat_title = uploaded_file.name.rsplit(".", 1)[0]

                    # Save chat state
                    save_chat_to_disk(
                        chat_id=st.session_state.chat_id,
                        title=st.session_state.chat_title,
                        doc_name=st.session_state.doc_name,
                        messages=st.session_state.messages,
                    )
                    st.success(f"Indexed: **{uploaded_file.name}**")
                    st.rerun()
                except Exception as e:
                    err_msg = str(e)
                    if "API_KEY_INVALID" in err_msg or "API key not valid" in err_msg:
                        st.error("❌ **Invalid Gemini API Key**: Google Gemini keys must start with `AIzaSy...`. Get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey) and paste it into `.env`.")
                    else:
                        st.error(f"Indexing Error: {err_msg}")

    st.markdown("---")

    # Chat History List (Stored Chats)
    st.markdown("#### 🕒 **Chat History**")
    saved_chats = list_saved_chats()

    if not saved_chats:
        st.caption("No saved conversations yet.")
    else:
        for chat in saved_chats:
            col1, col2 = st.columns([0.82, 0.18])
            is_active = chat["id"] == st.session_state.chat_id
            btn_label = f"💬 {chat['title'][:20]}" + ("..." if len(chat['title']) > 20 else "")
            
            with col1:
                if st.button(
                    btn_label,
                    key=f"select_{chat['id']}",
                    type="secondary" if not is_active else "primary",
                    use_container_width=True,
                ):
                    if not is_active:
                        # Load selected chat
                        data = load_chat_from_disk(chat["id"])
                        if data:
                            st.session_state.chat_id = data["id"]
                            st.session_state.chat_title = data.get("title", "Conversation")
                            st.session_state.doc_name = data.get("doc_name")
                            st.session_state.messages = data.get("messages", [])
                            # Re-load associated vector store
                            st.session_state.vector_store = load_vector_store(
                                chat["id"], api_key=active_api_key
                            )
                            st.rerun()

            with col2:
                if st.button("🗑️", key=f"del_{chat['id']}", help="Delete chat"):
                    delete_chat(chat["id"])
                    if is_active:
                        init_new_chat()
                    st.rerun()


# Main Chat Interface
col_title, col_badge = st.columns([0.7, 0.3])
with col_title:
    st.subheader(st.session_state.chat_title)
with col_badge:
    if st.session_state.doc_name:
        st.markdown(
            f"<div style='text-align: right;'><span class='doc-badge'>📄 {st.session_state.doc_name}</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='text-align: right;'><span style='color: #94a3b8; font-size: 0.85rem;'>No document attached</span></div>",
            unsafe_allow_html=True,
        )

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Chat Input
user_query = st.chat_input("Ask a question about your document...")

if user_query:
    # Check prerequisites
    if not active_api_key:
        st.error("⚠️ `GEMINI_API_KEY` is empty in `.env`. Please add your key to `.env` and save (`Ctrl+S`).")
        st.stop()

    if not st.session_state.vector_store:
        # Check if index exists on disk to load
        loaded_vs = load_vector_store(st.session_state.chat_id, api_key=active_api_key)
        if loaded_vs:
            st.session_state.vector_store = loaded_vs
        else:
            st.warning("Please upload a PDF/DOCX document in the sidebar first to query it.")
            st.stop()

    # Display user's question immediately
    with st.chat_message("user"):
        st.markdown(user_query)

    st.session_state.messages.append({"role": "user", "content": user_query})

    # If first message, generate a suitable title
    if st.session_state.chat_title == "New Conversation":
        st.session_state.chat_title = user_query[:35] + ("..." if len(user_query) > 35 else "")

    # Generate and stream assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking & searching document..."):
            try:
                # chat_history excluding the current user query that was just appended
                prior_history = st.session_state.messages[:-1]
                stream, standalone_q = generate_answer_stream(
                    user_query=user_query,
                    vector_store=st.session_state.vector_store,
                    chat_history=prior_history,
                    api_key=active_api_key,
                )
                # Stream directly to UI
                response_text = st.write_stream(stream)
            except Exception as e:
                err_msg = str(e)
                if "API_KEY_INVALID" in err_msg or "API key not valid" in err_msg:
                    response_text = "❌ **Invalid Gemini API Key**: Google Gemini keys must start with `AIzaSy...`. Please generate a free key at https://aistudio.google.com/app/apikey and paste it into `.env`."
                else:
                    response_text = f"⚠️ An error occurred while generating answer: {err_msg}"
                st.error(response_text)

    # Append assistant's answer and persist
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    save_chat_to_disk(
        chat_id=st.session_state.chat_id,
        title=st.session_state.chat_title,
        doc_name=st.session_state.doc_name,
        messages=st.session_state.messages,
    )
    st.rerun()
