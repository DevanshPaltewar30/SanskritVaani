"""
JnanaSetu – Sanskrit Knowledge Assistant
Interactive Streamlit Application
"""

import os
import sys
import time
import psutil
import streamlit as st

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config import (
    CORPUS_DIR,
    UPLOADS_DIR,
    DEFAULT_TOP_K,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_CPU_THREADS
)
from src.rag.engine import JnanaSetuEngine
from src.nlp.sanskrit_processor import SanskritProcessor


# Streamlit Page Config
st.set_page_config(
    page_title="JnanaSetu – Sanskrit Knowledge Assistant",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Devanagari typography, Sanskrit aesthetic, modern chat UI)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Yatra+One&family=Noto+Sans+Devanagari:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', 'Noto Sans Devanagari', sans-serif;
    }

    .main-title {
        font-family: 'Yatra One', 'Noto Sans Devanagari', serif;
        font-size: 2.3rem;
        font-weight: 700;
        color: #b45309;
        margin-bottom: 0.1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .tagline {
        font-size: 1.05rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }

    .telemetry-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #fef3c7;
        color: #92400e;
        margin-right: 0.4rem;
        border: 1px solid #fde68a;
    }

    .source-box {
        background-color: #f8fafc;
        border-left: 4px solid #d97706;
        border-radius: 0.375rem;
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }

    .source-header {
        font-weight: 600;
        color: #b45309;
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.3rem;
    }

    .sanskrit-text {
        font-family: 'Noto Sans Devanagari', sans-serif;
        color: #1e293b;
        line-height: 1.6;
    }

    .stButton>button {
        border-radius: 0.5rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing JnanaSetu Engine on CPU...")
def load_rag_engine():
    """Initializes and caches the JnanaSetu RAG Engine."""
    return JnanaSetuEngine()


def get_system_metrics():
    """Reads live CPU and RAM resource metrics."""
    process = psutil.Process()
    cpu_pct = psutil.cpu_percent(interval=None)
    ram_mb = process.memory_info().rss / (1024 * 1024)
    return cpu_pct, ram_mb


# Initialize Engine
try:
    engine = load_rag_engine()
except Exception as e:
    st.error(f"Failed to initialize JnanaSetu Engine: {e}")
    st.stop()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_history" not in st.session_state:
    st.session_state.uploaded_files_history = []


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("### 🕉️ ज्ञानसेतुः (JnanaSetu)")
    st.caption("**Bridging Ancient Sanskrit Knowledge with Modern AI**")
    st.divider()

    # System Status & CPU Monitor
    st.markdown("#### ⚡ Local CPU Hardware Monitor")
    cpu_pct, ram_mb = get_system_metrics()
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.metric("CPU Load", f"{cpu_pct:.1f}%")
    with col_c2:
        st.metric("RAM Used", f"{ram_mb:.0f} MB")
    
    st.caption(f"⚙️ Execution Mode: **Strictly CPU Only** | Threads: **{DEFAULT_CPU_THREADS}**")
    st.divider()

    # RAG Hyperparameters
    st.markdown("#### ⚙️ Retrieval Parameters")
    top_k = st.slider("Top Passages (K)", min_value=1, max_value=6, value=DEFAULT_TOP_K)
    temperature = st.slider("Temperature", min_value=0.0, max_value=0.8, value=DEFAULT_TEMPERATURE, step=0.05)
    st.divider()

    # Document Upload Section
    st.markdown("#### 📄 Upload Sanskrit Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload Sanskrit literature in Devanagari script (PDF/TXT)."
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.uploaded_files_history:
                dest_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
                with open(dest_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                with st.spinner(f"Indexing {uploaded_file.name}..."):
                    chunks_added = engine.add_document(dest_path)
                st.session_state.uploaded_files_history.append(uploaded_file.name)
                st.success(f"Added **{uploaded_file.name}** ({chunks_added} chunks)")

    st.divider()

    # Corpus Overview
    st.markdown("#### 📚 Active Sanskrit Corpus")
    all_corpus_files = []
    if os.path.exists(CORPUS_DIR):
        all_corpus_files.extend(os.listdir(CORPUS_DIR))
    if os.path.exists(UPLOADS_DIR):
        all_corpus_files.extend(os.listdir(UPLOADS_DIR))
    
    unique_files = sorted(list(set(all_corpus_files)))
    for f in unique_files:
        icon = "📕" if f.endswith(".pdf") else "📄"
        st.caption(f"{icon} {f}")

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        engine.clear_history()
        st.rerun()


# ==========================================
# MAIN INTERFACE
# ==========================================

st.markdown('<div class="main-title">ज्ञानसेतुः – JnanaSetu</div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">Bridging Ancient Sanskrit Knowledge with Modern AI • Local CPU RAG System</div>', unsafe_allow_html=True)

# Sample Query Quick Buttons
st.markdown("##### 💡 Sample Queries (Select to Test):")
sample_queries = [
    ("मूर्खभृत्यस्य कथा", "मूर्खभृत्येन खड्गेन किं कृतम्?"),
    ("Clever Kalidasa", "How did Kalidasa resolve the grammatical dilemma of 'शीतं बहु बाधति'?"),
    ("Old Woman's Wit", "What clever trick did the old woman play on the thieves?"),
    ("Effort vs Fate", "Explain the moral of the story about personal effort and divine assistance (दैवं च पुरुषार्थश्च)."),
    ("Transliterated Query", "chaturasya kalidasasya katha varnayatu")
]

cols = st.columns(len(sample_queries))
selected_sample = None
for i, (label, q_text) in enumerate(sample_queries):
    with cols[i]:
        if st.button(label, key=f"sample_{i}", use_container_width=True):
            selected_sample = q_text

# Render Previous Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧘‍♂️" if msg["role"] == "user" else "🕉️"):
        st.markdown(msg["content"])
        
        # Display Sources and Telemetry if available
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander(f"📜 View Retrieved Sanskrit Passages ({len(msg['sources'])} sources)"):
                for idx, src in enumerate(msg["sources"], start=1):
                    doc_name = src["metadata"].get("source", "Unknown")
                    page_num = src["metadata"].get("page", 1)
                    score = src.get("score", 0.0)
                    chunk_text = src.get("chunk", "")
                    
                    st.markdown(f"""
                    <div class="source-box">
                        <div class="source-header">
                            <span>📖 Passage {idx}: <b>{doc_name}</b> (Page {page_num})</span>
                            <span>Relevance: <b>{score:.3f}</b></span>
                        </div>
                        <div class="sanskrit-text">{chunk_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            if "telemetry" in msg:
                t = msg["telemetry"]
                st.markdown(f"""
                <div style="margin-top: 0.4rem;">
                    <span class="telemetry-badge">⏱️ Retrieval: {t.get('retrieval_time', 0):.3f}s</span>
                    <span class="telemetry-badge">⚡ Generation: {t.get('generation_time', 0):.2f}s ({t.get('tokens_per_second', 0):.1f} t/s)</span>
                    <span class="telemetry-badge">🖥️ CPU: {t.get('cpu', 0)}%</span>
                    <span class="telemetry-badge">💾 RAM: {t.get('ram', 0)} MB</span>
                </div>
                """, unsafe_allow_html=True)

# Chat Input Handler
user_input = st.chat_input("Ask a question in Sanskrit (Devanagari), English, or Roman transliteration...")
query_to_run = selected_sample or user_input

if query_to_run:
    # Append User Message
    st.session_state.messages.append({"role": "user", "content": query_to_run})
    with st.chat_message("user", avatar="🧘‍♂️"):
        st.markdown(query_to_run)

    # Generate Assistant Response
    with st.chat_message("assistant", avatar="🕉️"):
        with st.spinner("विचारयति (Searching Sanskrit texts and generating response)..."):
            response_data = engine.ask(
                question=query_to_run,
                top_k=top_k,
                temperature=temperature
            )

        answer_text = response_data["answer"]
        st.markdown(answer_text)

        # Show Sources
        sources = response_data.get("sources", [])
        if sources:
            with st.expander(f"📜 View Retrieved Sanskrit Passages ({len(sources)} sources)"):
                for idx, src in enumerate(sources, start=1):
                    doc_name = src["metadata"].get("source", "Unknown")
                    page_num = src["metadata"].get("page", 1)
                    score = src.get("score", 0.0)
                    chunk_text = src.get("chunk", "")
                    
                    st.markdown(f"""
                    <div class="source-box">
                        <div class="source-header">
                            <span>📖 Passage {idx}: <b>{doc_name}</b> (Page {page_num})</span>
                            <span>Relevance: <b>{score:.3f}</b></span>
                        </div>
                        <div class="sanskrit-text">{chunk_text}</div>
                    </div>
                    """, unsafe_allow_html=True)

        telemetry = {
            "retrieval_time": response_data.get("retrieval_time", 0),
            "generation_time": response_data.get("generation_time", 0),
            "tokens_per_second": response_data.get("tokens_per_second", 0),
            "cpu": response_data.get("cpu_utilization", 0),
            "ram": response_data.get("ram_usage_mb", 0)
        }
        
        st.markdown(f"""
        <div style="margin-top: 0.4rem;">
            <span class="telemetry-badge">⏱️ Retrieval: {telemetry['retrieval_time']:.3f}s</span>
            <span class="telemetry-badge">⚡ Generation: {telemetry['generation_time']:.2f}s ({telemetry['tokens_per_second']:.1f} t/s)</span>
            <span class="telemetry-badge">🖥️ CPU: {telemetry['cpu']}%</span>
            <span class="telemetry-badge">💾 RAM: {telemetry['ram']} MB</span>
        </div>
        """, unsafe_allow_html=True)

        # Save Assistant Message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "sources": sources,
            "telemetry": telemetry
        })

