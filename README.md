# 📄 Minimal PDF/DOC RAG Chatbot (Streamlit + Gemini + FAISS)

A ChatGPT-inspired, minimal conversational RAG chatbot built with Python, Streamlit, LangChain, Google Gemini, and FAISS.

---

## 🚀 Features

* **ChatGPT-Style Interface**: Sidebar with **"+ New Chat"**, document upload, and saved conversations.
* **Multi-Format Ingestion**: Supports `.pdf`, `.docx`, and `.doc` files.
* **Persistent Conversations**: Reopening old chats restores full message history and the document's vector store index from disk (`data/chats/` & `data/indexes/`).
* **Conversational Follow-ups**: Intelligently resolves pronouns and context from previous messages (e.g., *"What about 2025?"*).
* **Streaming Responses**: Clean token-by-token streaming response in real-time.
* **Zero Overengineering**: Exactly 3 clean Python files.

---

## 📁 Project Structure

```text
RAG/
├── indexing.py         # Document parsing, chunking, Gemini embeddings, and FAISS persistence
├── retrieval.py        # Conversational query reformulator, FAISS search, streaming answer generation
├── app.py              # Streamlit interface, chat history management, session handling
├── requirements.txt    # Project dependencies
├── .env.example        # Environment variable template
└── .env                # Your local API key configuration
```

---

## 🛠️ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Your API Key
Get a free Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
Add it to your `.env` file:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```
*(Alternatively, you can paste the API key directly into the sidebar field inside the Streamlit app).*

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

---

## 💡 How Follow-up Questions Work

1. You upload a document (e.g. a company financial report).
2. **User**: *"What was the revenue in 2024?"*
   **Assistant**: *"₹50 crore."*
3. **User**: *"What about 2025?"*
   `retrieval.py` reformulates the query using recent chat history into *"What was the revenue in 2025?"*, retrieves the relevant chunks from FAISS, and streams the accurate answer!
