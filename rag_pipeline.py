# rag_pipeline.py

import os
from dotenv import load_dotenv
from openai import OpenAI
from retriever import retrieve

load_dotenv()

CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY missing in .env")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

PROMPT_TEMPLATE = """
You are a helpful RAG assistant. Use ONLY the information in the retrieved documents.
If the answer is not found in the documents, say so.

History:
{history}

User Question:
{question}

Retrieved Documents:
{docs}

Provide a concise answer.
"""

def answer_question(question: str, chat_history: list = None, top_k: int = 5):
    """
    Returns a dict: {"answer": str, "token_usage": dict}
    """
    if chat_history is None:
        chat_history = []
        
    # Retrieve relevant docs
    docs = retrieve(question, top_k=top_k)
    docs_text = "\n\n---\n\n".join([d["payload"].get("text", "") for d in docs])

    # Format history for prompt
    # Taking last 10 exchanges to improve context retention
    # This helps maintain conversation continuity
    history_text = ""
    for msg in chat_history[-10:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    # Build prompt
    prompt = PROMPT_TEMPLATE.format(
        history=history_text, 
        question=question, 
        docs=docs_text
    )

    # Call OpenAI SDK (supports new v1.0+)
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a strict RAG assistant. You MUST answer in the same language as the user's question (e.g. if User asks in English, answer in English; if Indonesian, answer in Indonesian)."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=400
    )

    answer = response.choices[0].message.content
    usage = response.usage
    
    # Return structure with detailed usage
    token_usage = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens
    }

    return {
        "answer": answer,
        "token_usage": token_usage,
        "source_docs": docs  # Optional: return sources if needed for UI
    }

