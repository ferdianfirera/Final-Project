import os
import json
import sqlite3
import re
from dotenv import load_dotenv
from openai import OpenAI
from retriever import retrieve

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
SQLITE_FILE = os.getenv("SQLITE_FILE", "olist.db")

client = OpenAI(api_key=OPENAI_API_KEY)

def get_schema_and_samples(conn, max_rows=3):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [r[0] for r in cur.fetchall()]
    schema = {}
    samples = {}
    for t in tables:
        cur.execute(f"PRAGMA table_info({t})")
        cols = [{"cid": r[0], "name": r[1], "type": r[2]} for r in cur.fetchall()]
        schema[t] = cols
        cur.execute(f"SELECT * FROM {t} LIMIT {max_rows}")
        rows = cur.fetchall()
        samples[t] = {"columns": [c['name'] for c in cols], "rows": rows}
    return schema, samples


SQL_PROMPT = '''
You are an assistant that must produce a single SQL query (SQLite dialect) to answer the user's request.
Return only JSON: {{ "query": "SELECT ..." }}

Schema:
{schema}

Sample rows:
{samples}

Retrieved docs (may include clarifying details):
{docs}

{chat_history}

User question:
{question}

If the question cannot be answered by SQL on the provided schema, return a query that returns zero rows, e.g.: SELECT NULL WHERE 0

Rules:
1. **STRING COMPARISONS (CRITICAL for finding data)**:
   - ALWAYS use `UPPER()` on BOTH sides for case-insensitive matching: `UPPER(TRIM(city)) LIKE UPPER('%rio%')`
   - ALWAYS use `TRIM()` to remove leading/trailing whitespace: `TRIM(column_name)`
   - ALWAYS use `%` wildcards for partial matching: `LIKE '%value%'`
   - Example: `WHERE UPPER(TRIM(customer_city)) LIKE UPPER('%campinas%')`
   - NEVER use simple `=` for text unless exact match is required
   - Handle NULL: Use `COALESCE(column, '')` if NULL values might exist

2. Do not hallucinate columns. Use strict table.column names from schema.

3. **RELATIONSHIPS**: `orders` table joins with `customers` on `orders.customer_id = customers.customer_id`. DO NOT use `customer_unique_id` for joining orders.

4. **AGGREGATION**: If the user asks for "monthly", "trends", "graph", or "how many order per...", you MUST use this expression for grouping: `substr(order_purchase_timestamp, instr(order_purchase_timestamp, ' ') - 4, 4) || '-' || printf('%02d', substr(order_purchase_timestamp, 1, instr(order_purchase_timestamp, '/') - 1))`. Use this for both `GROUP BY` and the `SELECT` column label.

5. **CONTEXT INHERITANCE**: Check `Chat History` carefully. If the user asks follow-up questions, INHERIT filters and context from previous questions:
   - If previous question was about "pesanan di Campinas", and user asks "monthly graphs", apply the same city filter
   - If previous question was about specific date range, inherit the same date range
   - If previous question was about specific location city, inherit the same city location
   - If previous question was about specific location state, inherit the same state location

6. **AMBIGUOUS FOLLOW-UP QUESTIONS**: Pay special attention to ambiguous questions like "berapa totalnya", "total berapa", "how much total":
   - FIRST, check the Chat History to understand what the user was asking about
   - If previous question was about COUNT of orders (jumlah pesanan), then "total" means SUM of the COUNT
   - If previous question was about prices/payments, then "total" means SUM of prices
   - If previous question was about monthly breakdown, "total" means aggregate SUM across all months
   - **CRITICAL**: Match the metric type (COUNT vs SUM) from the previous question

7. **LIMIT (IMPORTANT - Avoid missing data)**:
   - For specific searches (WHERE conditions on names, cities, etc): Use `LIMIT 100` to ensure data is found
   - For aggregations (COUNT, SUM, AVG, GROUP BY): DO NOT use LIMIT (return all groups)
   - Only use LIMIT 20 for generic "list all" queries without specific filters
   - If user asks for "total" or "all": DO NOT use LIMIT

8. **VISUALIZATION**: To enable graphs, ensure your query returns categorical columns (labels) and numeric columns (values).

9. **COMPARISON BY LOCATION (CRITICAL for color differentiation)**:
   - When user asks to COMPARE between cities/regions (e.g., "compare Campinas and Rio de Janeiro"), you MUST include the city/state column in SELECT
   - Example: `SELECT customer_city, month, COUNT(*) as total_sales FROM ... GROUP BY customer_city, month`
   - This allows the visualization to show different colors for each city/region
   - Keywords that indicate comparison: "compare", "bandingkan", "versus", "vs", "between", "antara"
   - If comparing multiple cities, ALWAYS add city column to both SELECT and GROUP BY

10. **TIMESTAMPS**: The `order_purchase_timestamp` column is in format "M/D/YYYY HH:MM" (e.g. "10/2/2017 10:56" or "1/5/2017 08:30").
   - For YEAR filtering: Use `order_purchase_timestamp LIKE '%/2017 %'` or `order_purchase_timestamp LIKE '%/2018 %'`
   - For MONTH filtering: Use substring expression from Rule #4
   - NEVER use `YYYY-MM-DD` format comparisons
   - Handle variable month/day lengths (can be 1 or 2 digits)

IMPORTANT FOR FOLLOW-UP QUESTIONS:
- If user asks "berapa totalnya" or "total berapa" after a question about ORDER COUNT, calculate the SUM of order counts
- If user asks "berapa totalnya" after a question about PRICES/PAYMENTS, calculate the SUM of prices
- Always look at the Previous SQL query in Chat History to understand the context
- Maintain consistency with the previous query's metric type (COUNT, SUM, AVG, etc.)
'''


FINAL_PROMPT = '''
You are an assistant that synthesizes a final answer.

User question:
{question}

Retrieved documents:
{docs}

Executed SQL:
{sql}

SQL results (as CSV or table):
{results}

Instructions:
1. Provide a concise, human-readable answer explaining the results.
2. **LANGUAGE CONSISTENCY RULE (CRITICAL)**: Detect the language used in the "User question" (and "transcribed text" if evident). You MUST answer in the EXACT SAME language.
   - If user asks in Indonesian, answer in Indonesian.
   - If user asks in English, answer in English.
   - **NEVER** switch to Portuguese.
   - **DO NOT** output the instructions, just the answer.
'''

VISUALIZATION_PROMPT = '''
You are helpful assistant. Based on the following query and results, determine the best way to visualize the data.
Return valid JSON only.

User Query: {question}
SQL: {sql}
Results (sample): {results_sample}

Schema:
options: "bar", "line", "area", "scatter", "pie", "map", "none"
response format (JSON):
{{
    "type": "bar",
    "x": "column_name_for_x_axis",
    "y": "column_name_for_y_axis",
    "title": "Chart Title",
    "description": "Brief description of what this chart shows"
}}

Rules:
1. "x" and "y" MUST be EXACT column names from the Results (sample). Do not invent names.
2. If the data is just a single number (e.g. COUNT(*)), do NOT visualize. Return "type": "none".
3. If the data has 2+ rows and contains at least one categorical column and one numeric column, PREFER "type": "bar" or "line" (especially for time/dates).
4. If appropriate, generate a title.
'''


def generate_sql(question: str, schema: str, samples: str, docs: str, chat_history: str = "") -> str:
    prompt = SQL_PROMPT.format(schema=schema, samples=samples, docs=docs, question=question, chat_history=chat_history)

    # NEW OpenAI API (v1)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0,
    )

    raw = resp.choices[0].message.content

    # Clean markdown if present
    cleaned = raw.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(cleaned)
        return parsed.get("query")
    except Exception:
        import re
        # Try finding SELECT in the raw or cleaned string
        m = re.search(r"SELECT[\s\S]*?;", raw, re.I)
        if m:
            return m.group(0).strip()
        # Fallback: try cleaned string without JSON structure
        return cleaned


def execute_sql(conn, sql):
    cur = conn.cursor()
    try:
        cur.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall() # Fetchall to get all data for chart
        return {"columns": cols, "rows": rows, "error": None}
    except Exception as e:
        return {"columns": [], "rows": [], "error": str(e)}

def get_query_plan(conn, sql):
    cur = conn.cursor()
    try:
        cur.execute(f"EXPLAIN QUERY PLAN {sql}")
        rows = cur.fetchall()
        # Format typical output: selectid, order, from, detail
        plan_text = "\n".join([f"{r[0]}|{r[1]}|{r[2]}|{r[3]}" for r in rows])
        return plan_text
    except Exception as e:
        return f"Could not get query plan: {e}"

def decide_visualization(question, sql, result_dict):
    if result_dict.get("error") or not result_dict.get("rows"):
        return None
    
    # Take a sample of results to avoid overflowing context
    sample_rows = result_dict["rows"][:5]
    sample_res = {"columns": result_dict["columns"], "rows": sample_rows}
    
    prompt = VISUALIZATION_PROMPT.format(
        question=question,
        sql=sql,
        results_sample=str(sample_res)
    )
    
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0,
        )
        raw = resp.choices[0].message.content
        # Clean potential markdown
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        if data.get("type") == "none":
            return None
        return data
    except Exception as e:
        print(f"Error deciding visualization: {e}")
        return None


def sql_agent_query(question: str, chat_history: list = None, top_k: int = 3):
    if chat_history is None:
        chat_history = []
    
    # Format chat history for prompt
    # Build active context string to force-feed into prompt
    active_filters = []
    
    # Debugging: Print to console to verify context extraction
    print(f"\n[DEBUG] Scanning history for context (Total messages: {len(chat_history)})...")

    # Format chat history for prompt
    history_str = ""
    if chat_history:
        history_str = "Chat History (for context on follow-up questions):\n"
        history_str += "=" * 60 + "\n"
        recent = chat_history[-10:]
        
        # Scan history to find the LATEST active filters
        # Scan in reverse order (newest first) to find the most recent location context
        latest_city = None
        latest_state = None
        
        for msg in reversed(recent):
            if msg.get("sql"):
                sql = msg.get("sql").upper()
                if "WHERE" in sql:
                    where_part = sql.split("WHERE")[1]
                    
                    # Capture City (Robust against UPPER/LOWER/Whitespaces)
                    # Matches: customer_city ... like '...' OR customer_city ... = '...'
                    # Also handles UPPER(col) LIKE UPPER('val')
                    if not latest_city:
                         city_m = re.search(r"(?:customer_city|geolocation_city).*?(=|LIKE).*?['\"]%?([^'\"]+?)%?['\"]", where_part, re.IGNORECASE)
                         if city_m:
                             latest_city = city_m.group(2)
                             active_filters.append(f"CITY: {latest_city}")
                             print(f"[DEBUG] Found City Context: {latest_city}")

                    # Capture State (Only if City is NOT present - Hierarchy Rule)
                    # If we have a specific City, we don't need to enforce State explicitly
                    if not latest_state and not latest_city:
                         state_m = re.search(r"(?:customer_state|geolocation_state).*?(=|LIKE).*?['\"]%?(\w{2})%?['\"]", where_part, re.IGNORECASE)
                         if state_m:
                             latest_state = state_m.group(2)
                             active_filters.append(f"STATE: {latest_state}")
                             print(f"[DEBUG] Found State Context: {latest_state}")
                
                # Stop if we found location context, we don't want old stale ones
                if latest_city or latest_state:
                    break
        
        # Construct the history string for the LLM
        for i, msg in enumerate(recent, 1):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            history_str += f"\n[Message {i}] {role.upper()}: {content}\n"
            if msg.get("sql"):
                history_str += f"  → SQL Query: {msg.get('sql')}\n"
        
        history_str += "=" * 60 + "\n"

    # FORCE INJECT CONTEXT
    # Wrap the user question to make it impossible to ignore
    if active_filters:
        context_block = "\n".join([f"- {f}" for f in active_filters])
        
        new_prompt = f"""
!!! ACTIVE CONTEXT/HISTORY (Apply UNLESS User changes it) !!!
{context_block}
--------------------------------------------------
USER QUESTION:
{question}
--------------------------------------------------
INSTRUCTIONS for AI:
1. Check if the User Question explicitly changes a filter (e.g. asks for a different City/State).
2. IF YES (New Location) -> IGNORE the Active Context for that field and prioritize the User Question.
3. IF NO (Refinement like 'year 2017') -> YOU MUST COMBINE the Active Context with the User Question.

Example 1 (Refinement):
Context: [City: Campinas]
User: "Total in 2017"
Result: ... WHERE customer_city='Campinas' AND year='2017' ...

Example 2 (Overwrite):
Context: [City: Campinas]
User: "Total orders in Curitiba"
Result: ... WHERE customer_city='Curitiba' ... (Campinas is DROPPED)
"""
        question = new_prompt
        print(f"[DEBUG] Final Enforced Question:\n{question}")

    docs = retrieve(question, top_k=top_k)
    docs_text = "\n\n---\n\n".join([d["payload"].get("text", "") for d in docs])

    conn = sqlite3.connect(SQLITE_FILE, check_same_thread=False)
    
    # 1. Get Schema
    schema, samples = get_schema_and_samples(conn)
    schema_str = json.dumps(schema, default=str, indent=2)
    samples_str = json.dumps(samples, default=str, indent=2)

    # 2. Generate SQL
    sql = generate_sql(question, schema_str, samples_str, docs_text, history_str)
    
    # 3. Execute SQL
    result = execute_sql(conn, sql)
    
    # 4. Get Query Plan (only if valid SQL execution)
    plan = None
    if not result.get("error"):
        plan = get_query_plan(conn, sql)

    # 5. Decide Visualization (only if data exists)
    viz = None
    if not result.get("error") and result.get("rows"):
        viz = decide_visualization(question, sql, result)

    # 6. Generate Final Answer
    # Truncate results for LLM context if too large
    res_str = str(result)
    if len(res_str) > 6000:
        res_str = res_str[:6000] + "... (truncated)"

    final_prompt = FINAL_PROMPT.format(
        question=question,
        docs=docs_text[:4000],
        sql=sql,
        results=res_str
    )

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": final_prompt}],
        max_tokens=500,
        temperature=0.2,
    )

    final_answer = resp.choices[0].message.content
    
    # Usage extraction
    usage = resp.usage
    token_usage = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens
    }

    return {
        "sql": sql, 
        "result": result, 
        "plan": plan,
        "visualization": viz,
        "answer": final_answer,
        "token_usage": token_usage
    }
