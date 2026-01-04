import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

ROUTER_PROMPT = """
You are a playful and helpful Router Agent.
Your job is to decide whether a user's question is best answered by:
1. "SQL" - Querying a structured database (orders, customers, products, sellers, etc.).
2. "RAG" - Retrieving information from text documents (policies, general info, definitions, comment review, product category).

CRITICAL RULES:
- If the user asks for "graphs", "plots", "charts", or "trends" (e.g. "show me the monthly graph"), and the Chat History contains previous data/SQL results, you MUST route to "SQL".
- If the user asks a follow-up question like "what about 2018?" or "filter by Campinas", route to "SQL".

{chat_history}

User Question:
{question}

Output ONLY "SQL" or "RAG".
"""

def route_query(question: str, chat_history: list = None) -> str:
    """
    Decides whether to use SQL Agent or RAG Agent.
    Returns: 'SQL' or 'RAG'
    """
    if chat_history is None:
        chat_history = []
        
    history_str = ""
    if chat_history:
        history_str = "Chat History:\n"
        recent = chat_history[-5:]
        for msg in recent:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            history_str += f"{role.upper()}: {content}\n"

    router_prompt_filled = ROUTER_PROMPT.format(question=question, chat_history=history_str)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a classifier agent."},
            {"role": "user", "content": router_prompt_filled}
        ],
        temperature=0,
        max_tokens=10
    )
    
    choice = response.choices[0].message.content.strip().upper()
    # Fallback if something weird happens, though 'SQL' or 'RAG' is expected
    if "SQL" in choice:
        return "SQL"
    return "RAG"
