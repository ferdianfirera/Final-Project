import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import uuid
from datetime import datetime
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go

# Import local modules
from rag_pipeline import answer_question
from sql_agent import sql_agent_query
from router import route_query
from chat_memory import ChatMemory
from recommendation_engine import QueryRecommender

# Load environment variables
load_dotenv()

# Initialize OpenAI client for Whisper transcription
openai_client = OpenAI()

# ----------------------------
# Streamlit UI Configuration
# ----------------------------
st.set_page_config(
    page_title="Olist AI Assistant",
    layout="wide",
    page_icon="images/olistPage.png"
)

col1, col2 = st.columns([1, 10], vertical_alignment="center")
with col1:
    st.image("images/olistSidebar.png", width=100)
with col2:
    st.title("AI Assistant")

# Custom CSS for recommendation bubbles
st.markdown("""
<style>
/* Recommendation bubble styling */
div[data-testid="column"] > div > div > button[kind="secondary"] {
    border-radius: 20px !important;
    background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%) !important;
    color: #1565C0 !important;
    font-weight: 500 !important;
    border: 1px solid #90CAF9 !important;
    padding: 10px 16px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 44px !important;
}

div[data-testid="column"] > div > div > button[kind="secondary"]:hover {
    background: linear-gradient(135deg, #BBDEFB 0%, #90CAF9 100%) !important;
    transform: scale(1.02) !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
}

div[data-testid="column"] > div > div > button[kind="secondary"]:active {
    transform: scale(0.98) !important;
}

/* Recommendation section spacing */
.recommendation-section {
    margin-bottom: 1rem;
    padding: 1rem;
    background-color: #f8f9fa;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Session State Initialization
# ----------------------------
# Initialize chat memory
chat_memory = ChatMemory()

# Initialize or load session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.session_created = datetime.now()

if "messages" not in st.session_state:
    # Try to load from database
    loaded_messages = chat_memory.load_session(st.session_state.session_id)
    st.session_state.messages = loaded_messages if loaded_messages else []

if "token_usage" not in st.session_state:
    st.session_state.token_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }

if "last_agent" not in st.session_state:
    st.session_state.last_agent = "None"

if "recommender" not in st.session_state:
    st.session_state.recommender = QueryRecommender()

if "selected_recommendation" not in st.session_state:
    st.session_state.selected_recommendation = None

# ----------------------------
# Sidebar Controls
# ----------------------------
with st.sidebar:
    col_sb1, col_sb2 = st.columns([1, 4], vertical_alignment="center")
    with col_sb1:
        st.image("images/olistSidebar.png", width='stretch')
    with col_sb2:
        st.header("Dashboard")
    
    # Session Info
    st.subheader("💾 Session Info")
    st.caption(f"Session ID: {st.session_state.session_id[:8]}...")
    st.caption(f"Messages: {len(st.session_state.messages)}")
    
    st.metric(label="Last Used Agent", value=st.session_state.last_agent)
    
    st.subheader("📊 Total Token Usage")
    col1, col2, col3 = st.columns(3)
    col1.metric("Prompt", st.session_state.token_usage["prompt_tokens"])
    col2.metric("Completion", st.session_state.token_usage["completion_tokens"])
    col3.metric("Total", st.session_state.token_usage["total_tokens"])
    
    # Show recent sessions
    st.markdown("---")
    st.subheader("📜 Recent Sessions")
    recent_sessions = chat_memory.get_recent_sessions(limit=5)
    
    if recent_sessions:
        for session in recent_sessions:
            is_current = session["session_id"] == st.session_state.session_id
            prefix = "▶️ " if is_current else "📝 "
            
            with st.expander(f"{prefix}{session['preview'] if session['preview'] else 'Empty session'}", expanded=False):
                st.caption(f"Messages: {session['message_count']}")
                st.caption(f"Last updated: {session['last_updated']}")
                
                if not is_current:
                    if st.button(f"Load", key=f"load_{session['session_id']}"):
                        st.session_state.session_id = session["session_id"]
                        st.session_state.messages = chat_memory.load_session(session["session_id"])
                        st.rerun()
    else:
        st.caption("No previous sessions")

    # Session Management Buttons
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🆕 New Chat", width='stretch'):
            # Create new session
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.last_agent = "None"
            st.session_state.token_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
            st.rerun()
    
    with col_b:
        if st.button("🗑️ Clear", width='stretch'):
            # Clear current session from memory
            chat_memory.delete_session(st.session_state.session_id)
            st.session_state.messages = []
            st.session_state.last_agent = "None"
            st.rerun()


# ----------------------------
# Helper Functions
# ----------------------------
def transcribe_audio(audio_input):
    """Transcribe audio using OpenAI transcribe API"""
    try:
        # Handle UploadedFile object from Streamlit
        if hasattr(audio_input, 'read'):
            # It's a file-like object (UploadedFile)
            audio_file = audio_input
        else:
            # It's raw bytes
            import io
            audio_file = io.BytesIO(audio_input)
            audio_file.name = "audio.wav"  # Whisper API needs a filename
        
        # Transcribe using Whisper
        transcript = openai_client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file,
            language=None,
        )
        
        return transcript.text
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return None

# ----------------------------
# Chat Display
# ----------------------------
def detect_comparison_column(dataframe, x_col):
    """Deteksi kolom yang mengandung informasi untuk pembandingan (kota, daerah, tahun, kategori)"""
    if not isinstance(dataframe, pd.DataFrame):
        return None
    
    # PRIORITAS 1: Cek kolom city/kota/daerah terlebih dahulu (paling penting)
    for col in dataframe.columns:
        col_lower = col.lower()
        # Cek apakah kolom ini berisi informasi lokasi/kota
        if any(keyword in col_lower for keyword in ['city', 'kota', 'cidade', 'customer_city', 'geolocation_city']):
            # Validasi bahwa kolom ini memiliki lebih dari 1 nilai unik (berarti ada perbandingan)
            if dataframe[col].nunique() > 1:
                return col
        # Cek kolom state/negara bagian
        if any(keyword in col_lower for keyword in ['state', 'estado', 'negara', 'customer_state', 'geolocation_state']):
            if dataframe[col].nunique() > 1:
                return col
    
    # PRIORITAS 2: Cek apakah x_col mengandung format tahun-bulan (2017-01, 2018-01, dll)
    # Ini untuk kasus time-series comparison
    if x_col in dataframe.columns:
        sample_val = str(dataframe[x_col].iloc[0]) if not dataframe.empty else ""
        if "-" in sample_val and len(sample_val.split("-")[0]) == 4:
            # Extract tahun dari format YYYY-MM untuk membedakan warna per tahun
            dataframe['_comparison_group'] = dataframe[x_col].astype(str).str[:4]
            return '_comparison_group'
    
    # PRIORITAS 3: Cek kolom kategori lainnya
    for col in dataframe.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['year', 'tahun', 'ano', 'category', 'kategori', 'categoria']):
            if dataframe[col].nunique() > 1:
                return col
    
    return None

def get_color_palette(n_colors):
    """Generate palet warna yang berbeda untuk setiap kategori"""
    colors = [
        '#3498db',  # Blue
        '#2ecc71',  # Green
        '#e74c3c',  # Red
        '#f39c12',  # Orange
        '#9b59b6',  # Purple
        '#1abc9c',  # Turquoise
        '#34495e',  # Dark Gray
        '#e67e22',  # Carrot
        '#16a085',  # Green Sea
        '#c0392b',  # Pomegranate
    ]
    return colors[:n_colors] if n_colors <= len(colors) else colors * (n_colors // len(colors) + 1)

def display_chart(dataframe, viz_config):
    if not viz_config or not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        return
    
    chart_type = viz_config.get("type")
    x_col = viz_config.get("x")
    y_col = viz_config.get("y")
    title = viz_config.get("title", "")
    
    # Validasi kolom
    if x_col not in dataframe.columns or y_col not in dataframe.columns:
        st.warning(f"Chart data error: Columns {x_col} or {y_col} not found in data.")
        return
    
    # Deteksi kolom perbandingan
    comparison_col = detect_comparison_column(dataframe, x_col)
    
    # Buat chart dengan Plotly untuk kontrol warna yang lebih baik
    try:
        if chart_type == "bar":
            if comparison_col and comparison_col in dataframe.columns:
                # Ada perbandingan - gunakan warna berbeda
                unique_groups = dataframe[comparison_col].unique()
                colors = get_color_palette(len(unique_groups))
                color_map = {group: colors[i] for i, group in enumerate(unique_groups)}
                
                fig = px.bar(
                    dataframe, 
                    x=x_col, 
                    y=y_col,
                    color=comparison_col,
                    title=title,
                    color_discrete_map=color_map,
                    labels={x_col: x_col, y_col: y_col, comparison_col: comparison_col}
                )
            else:
                # Tidak ada perbandingan - warna tunggal
                fig = px.bar(
                    dataframe, 
                    x=x_col, 
                    y=y_col,
                    title=title,
                    color_discrete_sequence=['#3498db']
                )
            
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title=y_col,
                height=500,
                showlegend=True if comparison_col else False
            )
            st.plotly_chart(fig, width='stretch')
            
        elif chart_type == "line":
            if comparison_col and comparison_col in dataframe.columns:
                unique_groups = dataframe[comparison_col].unique()
                colors = get_color_palette(len(unique_groups))
                color_map = {group: colors[i] for i, group in enumerate(unique_groups)}
                
                fig = px.line(
                    dataframe, 
                    x=x_col, 
                    y=y_col,
                    color=comparison_col,
                    title=title,
                    color_discrete_map=color_map,
                    markers=True
                )
            else:
                fig = px.line(
                    dataframe, 
                    x=x_col, 
                    y=y_col,
                    title=title,
                    color_discrete_sequence=['#3498db'],
                    markers=True
                )
            
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title=y_col,
                height=500,
                showlegend=True if comparison_col else False
            )
            st.plotly_chart(fig, width='stretch')
            
        elif chart_type == "area":
            if comparison_col and comparison_col in dataframe.columns:
                unique_groups = dataframe[comparison_col].unique()
                colors = get_color_palette(len(unique_groups))
                color_map = {group: colors[i] for i, group in enumerate(unique_groups)}
                
                fig = px.area(
                    dataframe, 
                    x=x_col, 
                    y=y_col,
                    color=comparison_col,
                    title=title,
                    color_discrete_map=color_map
                )
            else:
                fig = px.area(
                    dataframe, 
                    x=x_col, 
                    y=y_col,
                    title=title,
                    color_discrete_sequence=['#3498db']
                )
            
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title=y_col,
                height=500,
                showlegend=True if comparison_col else False
            )
            st.plotly_chart(fig, width='stretch')
            
        elif chart_type == "scatter":
            if comparison_col and comparison_col in dataframe.columns:
                unique_groups = dataframe[comparison_col].unique()
                colors = get_color_palette(len(unique_groups))
                color_map = {group: colors[i] for i, group in enumerate(unique_groups)}
                
                fig = px.scatter(
                    dataframe, 
                    x=x_col, 
                    y=y_col,
                    color=comparison_col,
                    title=title,
                    color_discrete_map=color_map
                )
            else:
                fig = px.scatter(
                    dataframe, 
                    x=x_col, 
                    y=y_col,
                    title=title,
                    color_discrete_sequence=['#3498db']
                )
            
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title=y_col,
                height=500,
                showlegend=True if comparison_col else False
            )
            st.plotly_chart(fig, width='stretch')
    
    except Exception as e:
        st.error(f"Error displaying chart: {str(e)}")
        # Fallback ke Streamlit chart bawaan
        if chart_type == "bar":
            st.bar_chart(dataframe, x=x_col, y=y_col)
        elif chart_type == "line":
            st.line_chart(dataframe, x=x_col, y=y_col)
    
    # Cleanup temporary column
    if '_comparison_group' in dataframe.columns:
        dataframe.drop('_comparison_group', axis=1, inplace=True)
    
    if viz_config.get("description"):
        st.caption(viz_config.get("description"))


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # SQL Agent Extra Details
        if message.get("agent") == "SQL":
            if "sql" in message:
                 with st.expander("Show SQL Query"):
                     st.code(message["sql"], language="sql")
                        
            if "dataframe" in message:
                with st.expander("Show Data"):
                    st.dataframe(message["dataframe"])
            
            if "viz_config" in message and "dataframe" in message:
                display_chart(message["dataframe"], message["viz_config"])


# ----------------------------
# Query Recommendations
# ----------------------------
# Display recommendations if there's chat history
if st.session_state.messages and len(st.session_state.messages) > 0:
    try:
        # Get the last user message for context
        last_user_msg = None
        for msg in reversed(st.session_state.messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break
        
        if last_user_msg:
            recommendations = st.session_state.recommender.get_recommendations(
                current_query=last_user_msg,
                chat_history=st.session_state.messages,
                num_recommendations=6
            )
            
            if recommendations:
                st.markdown("")
                st.markdown("💡 **Pertanyaan yang mungkin Anda butuhkan:**")
                
                # Display in 3 columns for desktop, responsive on mobile
                cols = st.columns(3)
                for idx, rec in enumerate(recommendations):
                    col_idx = idx % 3
                    with cols[col_idx]:
                        # Create unique key for each button
                        button_key = f"rec_{rec['id']}_{idx}_{len(st.session_state.messages)}"
                        if st.button(
                            rec["text"],
                            key=button_key,
                            width='stretch',
                            type="secondary"
                        ):
                            # Store selected recommendation
                            st.session_state.selected_recommendation = rec["text"]
                            st.rerun()
                
                st.markdown("")  # Add spacing
    except Exception as e:
        # Silently fail if recommendation engine has issues
        print(f"Recommendation error: {e}")
        pass

# ----------------------------
# Chat Logic
# ----------------------------
# Chat input with voice support
user_input = st.chat_input(
    "Ask a question about Olist data (SQL) or general info (RAG)...",
    accept_audio=True
)

# Check if user clicked a recommendation
prompt = None

if st.session_state.selected_recommendation:
    prompt = st.session_state.selected_recommendation
    st.session_state.selected_recommendation = None  # Clear it
elif user_input:
    # Extract text from ChatInputValue or plain string
    # Process input - handle both text and voice
    if isinstance(user_input, str):
        prompt = user_input
    elif user_input:
        # Check for audio input first (from voice)
        if hasattr(user_input, 'audio') and user_input.audio:
            with st.spinner("Transcribing voice input..."):
                transcribed_text = transcribe_audio(user_input.audio)
                if transcribed_text:
                    prompt = transcribed_text
        # Check for text input (rich object)
        elif hasattr(user_input, 'text'):
            prompt = user_input.text
        # Fallback
        else:
             prompt = str(user_input)

if prompt:
        
        # 1. Add user message to history
        user_msg = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_msg)
        
        # Save user message to database
        chat_memory.save_message(st.session_state.session_id, user_msg)
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Route the query
        with st.spinner("Routing query..."):
            # Exclude the current user message from history
            agent_type = route_query(prompt, st.session_state.messages[:-1])
            st.session_state.last_agent = agent_type

        # 3. Process with selected agent
        response_content = ""
        sql_query = None
        df_result = None
        viz_config = None
        
        with st.spinner(f"Processing with {agent_type} Agent..."):
            try:
                if agent_type == "SQL":
                    # Exclude the current user message from history
                    result = sql_agent_query(prompt, st.session_state.messages[:-1])
                    response_content = result.get("answer", "No answer generated.")
                    sql_query = result.get("sql")
                    viz_config = result.get("visualization")
                    
                    # Check for data rows to display
                    if result.get("result") and not result["result"].get("error"):
                        rows = result["result"].get("rows", [])
                        cols = result["result"].get("columns", [])
                        if rows:
                            # Deduplicate column names if necessary
                            if len(cols) != len(set(cols)):
                                seen = {}
                                new_cols = []
                                for c in cols:
                                    if c in seen:
                                        seen[c] += 1
                                        new_cols.append(f"{c}_{seen[c]}")
                                    else:
                                        seen[c] = 0
                                        new_cols.append(c)
                                cols = new_cols
                            
                            df_result = pd.DataFrame(rows, columns=cols)

                    # Update usage
                    usage = result.get("token_usage", {})
                    st.session_state.token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    st.session_state.token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                    st.session_state.token_usage["total_tokens"] += usage.get("total_tokens", 0)

                else: # RAG
                    history_for_rag = st.session_state.messages[:-1] 
                    result = answer_question(prompt, chat_history=history_for_rag)
                    response_content = result.get("answer", "No answer found.")
                    
                    # Update usage
                    usage = result.get("token_usage", {})
                    st.session_state.token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    st.session_state.token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                    st.session_state.token_usage["total_tokens"] += usage.get("total_tokens", 0)

            except Exception as e:
                response_content = f"⚠️ An error occurred: {str(e)}"
                import traceback
                traceback.print_exc()

        # 4. Display Assistant Response
        with st.chat_message("assistant"):
            st.markdown(response_content)
            
            if agent_type == "SQL":
                if sql_query:
                    with st.expander("Show SQL Query"):
                        st.code(sql_query, language="sql")

                if df_result is not None:
                    with st.expander("Show Data"):
                        st.dataframe(df_result)
                
                if viz_config and df_result is not None:
                    display_chart(df_result, viz_config)

        # 5. Append assistant message to history
        assistant_msg = {
            "role": "assistant", 
            "content": response_content,
            "agent": agent_type
        }
        if sql_query:
            assistant_msg["sql"] = sql_query
        if df_result is not None:
            assistant_msg["dataframe"] = df_result
        if viz_config:
            assistant_msg["viz_config"] = viz_config
        
        st.session_state.messages.append(assistant_msg)
        
        # Save assistant message to database
        chat_memory.save_message(st.session_state.session_id, assistant_msg)
        
        # Rerun to update recommendations immediately
        st.rerun()
