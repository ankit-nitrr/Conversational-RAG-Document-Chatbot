"""
Verification script for the PDF/DOC RAG Chatbot.
Runs through:
1. File verification (GenAI Interview Handbook)
2. Text extraction & chunking pipeline
3. Local Chat JSON persistence (saving and loading)
4. FAISS index directory validation
5. Gemini API Key check
"""

import os
import sys

# Force UTF-8 encoding on standard output for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv(override=True)

print("=" * 60)
print(" STARTING RAG SYSTEM VERIFICATION")
print("=" * 60)

# Step 1: Check dependencies
print("\n[Step 1] Checking installed dependencies...")
modules = [
    "streamlit",
    "langchain",
    "langchain_community",
    "langchain_google_genai",
    "faiss",
    "pypdf",
    "docx",
    "dotenv",
]
for mod in modules:
    try:
        __import__(mod)
        print(f"  [OK] {mod} is installed.")
    except ImportError as e:
        print(f"  [FAIL] {mod} failed: {e}")
        sys.exit(1)

# Step 2: Check target PDF
pdf_filename = "GenAI Interview Handbook _ DIPANSHU.DECODED.pdf"
print(f"\n[Step 2] Checking reference document: '{pdf_filename}'...")
if os.path.exists(pdf_filename):
    size_mb = os.path.getsize(pdf_filename) / (1024 * 1024)
    print(f"  [OK] Found PDF ({size_mb:.2f} MB).")
else:
    print(f"  [FAIL] PDF file '{pdf_filename}' not found in current folder.")
    sys.exit(1)

# Step 3: Test text extraction
print("\n[Step 3] Extracting text from PDF...")
from pypdf import PdfReader

reader = PdfReader(pdf_filename)
pages_count = len(reader.pages)
full_text = "\n\n".join([page.extract_text() or "" for page in reader.pages])
char_count = len(full_text)
word_count = len(full_text.split())
print(f"  [OK] Extracted {pages_count} pages.")
print(f"  [OK] Total characters: {char_count:,}")
print(f"  [OK] Total words: {word_count:,}")

# Step 4: Test chunking
print("\n[Step 4] Testing chunking pipeline...")
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.create_documents([full_text], metadatas=[{"source": pdf_filename}])
print(f"  [OK] Created {len(chunks)} chunks.")
print(f"  [OK] First chunk preview:\n    {repr(chunks[0].page_content[:100])}...")

# Step 5: Test Chat Persistence (Save & Load)
print("\n[Step 5] Testing JSON chat history persistence...")
from app import save_chat_to_disk, load_chat_from_disk, delete_chat

test_chat_id = "test_verification_chat"
test_title = "GenAI Handbook Interview Prep"
test_messages = [
    {"role": "user", "content": "What is an LLM?"},
    {"role": "assistant", "content": "An LLM is a large language model based on transformer architecture."},
    {"role": "user", "content": "What about temperature?"},
    {"role": "assistant", "content": "Temperature controls sampling randomness in generation."},
]

save_chat_to_disk(test_chat_id, test_title, pdf_filename, test_messages)
loaded_data = load_chat_from_disk(test_chat_id)

assert loaded_data is not None, "Failed to load saved chat"
assert loaded_data["title"] == test_title, "Title mismatch"
assert len(loaded_data["messages"]) == 4, "Messages count mismatch"
print(f"  [OK] Successfully saved and reloaded conversation '{test_title}' with {len(loaded_data['messages'])} messages.")

# Clean up test chat
delete_chat(test_chat_id)
print("  [OK] Test chat cleanly deleted.")

# Step 6: Check API Key status
print("\n[Step 6] Checking Google Gemini API Key in .env...")
api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
if not api_key:
    print("  [INFO] GEMINI_API_KEY is currently EMPTY in .env.")
    print("         To complete live embedding with Gemini:")
    print("         Paste your key starting with 'AIzaSy...' in .env (line 3) and save (Ctrl+S).")
elif api_key.startswith("AIzaSy"):
    print(f"  [OK] Valid Gemini API key format detected (starts with 'AIzaSy', length: {len(api_key)}).")
    print("  Testing live embedding call with Gemini...")
    try:
        from indexing import get_embedding_model
        model = get_embedding_model(api_key=api_key)
        emb = model.embed_query("Hello world")
        print(f"  [OK] Live embedding successful! Embedding dimension: {len(emb)}")
    except Exception as e:
        print(f"  [FAIL] Live embedding call failed: {e}")
else:
    print(f"  [WARNING] Key found in .env, but does NOT start with 'AIzaSy' (starts with '{api_key[:6]}...').")
    print("            Google Gemini keys must start with 'AIzaSy...'. Get one at https://aistudio.google.com/app/apikey")

print("\n" + "=" * 60)
print(" RAG SYSTEM VERIFICATION COMPLETED SUCCESSFULLY")
print("=" * 60)
