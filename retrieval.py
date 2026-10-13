import os
from typing import Generator, List, Dict, Optional, Tuple
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

load_dotenv(override=True)


def extract_text(content) -> str:
    """Helper to extract pure string content from Gemini response parts."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                texts.append(part["text"])
            elif isinstance(part, str):
                texts.append(part)
        return "".join(texts)
    return str(content)


def get_chat_model(api_key: Optional[str] = None, streaming: bool = True) -> ChatGoogleGenerativeAI:
    """Initializes and returns the Google Gemini chat model."""
    load_dotenv(override=True)
    key = (api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not key:
        raise ValueError(
            "Gemini API Key not found. Please set GEMINI_API_KEY in your .env file."
        )
    os.environ["GEMINI_API_KEY"] = key
    os.environ["GOOGLE_API_KEY"] = key
    return ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=key,
        temperature=0.2,
        streaming=streaming,
    )


def contextualize_question(
    user_query: str,
    chat_history: List[Dict[str, str]],
    api_key: Optional[str] = None,
) -> str:
    """
    Reformulates follow-up questions into standalone queries using recent chat history.
    Example:
      User: "What was the revenue in 2024?" -> Assistant: "50 crore."
      User: "What about 2025?" -> Reformulated: "What was the revenue in 2025?"
    """
    # If no prior history, the question is already standalone
    if not chat_history:
        return user_query

    # Format last 4 messages for context
    recent_history = chat_history[-4:]
    history_str = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_history]
    )

    rephrase_prompt = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that captures the user's full intent.
Do NOT answer the question. Only output the reformulated standalone question. If the question is already standalone, return it as is.

Chat History:
{history_str}

Follow-up Question: {user_query}
Standalone Question:"""

    try:
        # Fast non-streaming call to rephrase
        fast_llm = get_chat_model(api_key=api_key, streaming=False)
        response = fast_llm.invoke(rephrase_prompt)
        standalone = extract_text(response.content).strip()
        return standalone if standalone else user_query
    except Exception:
        # Fallback gracefully to original question if LLM call fails
        return user_query


def retrieve_relevant_chunks(
    query: str,
    vector_store: FAISS,
    top_k: int = 4,
) -> List[str]:
    """Retrieves top-k relevant document chunks from the FAISS vector store."""
    docs = vector_store.similarity_search(query, k=top_k)
    return [doc.page_content for doc in docs]


def generate_answer_stream(
    user_query: str,
    vector_store: FAISS,
    chat_history: List[Dict[str, str]],
    api_key: Optional[str] = None,
    top_k: int = 4,
) -> Tuple[Generator[str, None, None], str]:
    """
    Full RAG pipeline:
    1. Reformulate question using conversational history.
    2. Retrieve relevant document chunks.
    3. Stream answers from Gemini using retrieved context + dialogue history.
    
    Returns:
        (token_generator, standalone_query)
    """
    # 1. Resolve follow-up ambiguities
    standalone_query = contextualize_question(user_query, chat_history, api_key=api_key)

    # 2. Retrieve context chunks
    context_chunks = retrieve_relevant_chunks(standalone_query, vector_store, top_k=top_k)
    context_text = "\n\n---\n\n".join(context_chunks)

    # 3. Construct prompt
    system_instruction = (
        "You are a helpful, precise document assistant. "
        "Answer the user's question using the provided document context below as your primary source. "
        "Maintain conversational continuity with the recent chat history. "
        "If the answer cannot be found in the document context, clearly state that the document does not contain that information. "
        "Be concise, clear, and direct."
    )

    messages = [SystemMessage(content=system_instruction)]

    # Append recent chat history (up to last 6 messages)
    for msg in chat_history[-6:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    # Current prompt with context
    user_prompt = f"""Document Context:
{context_text}

User Question: {standalone_query}"""

    messages.append(HumanMessage(content=user_prompt))

    # 4. Stream response
    llm = get_chat_model(api_key=api_key, streaming=True)

    def stream_generator():
        for chunk in llm.stream(messages):
            txt = extract_text(chunk.content)
            if txt:
                yield txt

    return stream_generator(), standalone_query
