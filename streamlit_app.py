"""
STREAMLIT_APP.PY - Giao diện Streamlit cho TOEIC Chatbot
Chạy lệnh: streamlit run streamlit_app.py
"""

# ==========================================
# PYTORCH FIX
# ==========================================
import torch
try:
    torch.classes.__path__ = []
except Exception:
    pass

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Accessing.*__path__.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*alias will be removed.*")

# ==========================================
# IMPORTS
# ==========================================
import sys
import logging
import streamlit as st
from src.ask import stream_bot_response
from src.rag import get_vector_db, _get_bm25_retriever
from src.config import HYBRID_SEARCH
from src.memory import ConversationMemory

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="AI TOEIC Chatbot",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ==========================================
# INJECT CSS CUSTOM
# ==========================================
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root & body ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── Background ── */
.stApp {
    background:
        linear-gradient(135deg, rgba(108,92,231,0.55) 0%, rgba(0,206,201,0.30) 100%),
        url('https://images.unsplash.com/photo-1497366216548-37526070297c?w=1600&q=80')
        center/cover no-repeat fixed;
}

/* ── Main block backdrop ── */
.block-container {
    background: rgba(15, 12, 41, 0.35) !important;
    backdrop-filter: blur(2px);
    border-radius: 0 !important;
    padding-top: 0 !important;
    max-width: 860px !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ══════════════════════════════════════
   CUSTOM HEADER
══════════════════════════════════════ */
.custom-header {
    position: sticky;
    top: 0;
    z-index: 999;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 24px;
    background: rgba(15, 12, 41, 0.65);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(255,255,255,0.15);
    margin: -1rem -1rem 1.5rem -1rem;
}

.header-left-text {
    font-size: 14px;
    font-weight: 600;
    color: rgba(255,255,255,0.80);
    display: flex;
    align-items: center;
    gap: 7px;
}

.status-dot {
    width: 8px; height: 8px;
    background: #00b894;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 6px #00b894;
    animation: pulse 2s ease infinite;
}

@keyframes pulse {
    0%,100% { opacity:1; } 50% { opacity:0.45; }
}

.header-center-logo {
    display: flex;
    align-items: center;
    gap: 10px;
}

.logo-circle {
    width: 38px; height: 38px;
    background: linear-gradient(135deg,#6C5CE7,#a78bfa);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}

.logo-title {
    font-size: 20px;
    font-weight: 700;
    background: linear-gradient(135deg,#fff 30%,#a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ══════════════════════════════════════
   CHAT MESSAGES
══════════════════════════════════════ */
/* Bot message bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: rgba(255,255,255,0.93) !important;
    border-radius: 0 18px 18px 18px !important;
    padding: 14px 18px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.12) !important;
    margin-bottom: 6px !important;
    border: none !important;
    max-width: 82% !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) * {
    color: #1e1b4b !important;
}

/* User message bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(135deg,#6C5CE7 0%,#a78bfa 100%) !important;
    border-radius: 18px 0 18px 18px !important;
    padding: 14px 18px !important;
    box-shadow: 0 4px 20px rgba(108,92,231,0.40) !important;
    margin-left: auto !important;
    margin-bottom: 6px !important;
    border: none !important;
    max-width: 72% !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) * {
    color: #ffffff !important;
}

/* Avatar styles */
[data-testid="stChatMessageAvatarAssistant"] {
    background: linear-gradient(135deg,#6C5CE7,#a78bfa) !important;
    border: 2px solid rgba(255,255,255,0.3) !important;
    border-radius: 50% !important;
}

[data-testid="stChatMessageAvatarUser"] {
    background: linear-gradient(135deg,#00CEC9,#6C5CE7) !important;
    border: 2px solid rgba(255,255,255,0.3) !important;
    border-radius: 50% !important;
    color: #fff !important;
}

/* Message content text */
[data-testid="stChatMessage"] p {
    line-height: 1.7 !important;
    font-size: 14.5px !important;
}

/* ══════════════════════════════════════
   SUGGESTION CHIPS
══════════════════════════════════════ */
.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: flex-end;
    margin: 12px 0 16px;
}

/* Override Streamlit button for chips */
div[data-testid="stHorizontalBlock"] .stButton > button {
    padding: 7px 15px !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255,255,255,0.32) !important;
    background: rgba(255,255,255,0.14) !important;
    backdrop-filter: blur(10px) !important;
    color: rgba(255,255,255,0.92) !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
    white-space: nowrap !important;
    width: auto !important;
    min-width: 0 !important;
}

div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    background: rgba(255,255,255,0.28) !important;
    border-color: rgba(255,255,255,0.55) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(108,92,231,0.3) !important;
    color: #fff !important;
}

/* ══════════════════════════════════════
   CLEAR BUTTON (sidebar)
══════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.75) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255,255,255,0.12) !important;
}

section[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.10) !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    color: rgba(255,255,255,0.85) !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    width: 100% !important;
    transition: all 0.2s !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.20) !important;
    color: #fff !important;
}

/* ══════════════════════════════════════
   BOTTOM CONTAINER — xóa khung đen
══════════════════════════════════════ */
[data-testid="stBottomBlockContainer"] {
    background: transparent !important;
    backdrop-filter: none !important;
    padding: 6px 0 10px !important;
    box-shadow: none !important;
    border-top: none !important;
}

/* ══════════════════════════════════════
   FORM INPUT BAR
══════════════════════════════════════ */
/* Wrapper form */
[data-testid="stForm"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* Text input field */
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.95) !important;
    color: #1e1b4b !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14.5px !important;
    border: 1.5px solid rgba(108,92,231,0.25) !important;
    border-radius: 28px !important;
    padding: 12px 20px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.18) !important;
    outline: none !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

[data-testid="stTextInput"] input:focus {
    border-color: #6C5CE7 !important;
    box-shadow: 0 8px 32px rgba(108,92,231,0.28) !important;
}

[data-testid="stTextInput"] input::placeholder {
    color: #9ca3af !important;
}

/* Remove default label & wrapper padding */
[data-testid="stTextInput"] label { display: none !important; }
[data-testid="stTextInput"] > div { padding: 0 !important; }

/* Send button */
button[kind="primaryFormSubmit"],
button[data-testid="baseButton-primaryFormSubmit"] {
    background: linear-gradient(135deg,#6C5CE7,#a78bfa) !important;
    border: none !important;
    border-radius: 50% !important;
    width: 46px !important;
    height: 46px !important;
    padding: 0 !important;
    color: white !important;
    font-size: 18px !important;
    box-shadow: 0 4px 14px rgba(108,92,231,0.45) !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin-top: 0 !important;
}

button[kind="primaryFormSubmit"]:hover,
button[data-testid="baseButton-primaryFormSubmit"]:hover {
    transform: scale(1.10) !important;
    box-shadow: 0 6px 20px rgba(108,92,231,0.60) !important;
}

/* Clear button (secondary) */
button[kind="secondaryFormSubmit"],
button[data-testid="baseButton-secondaryFormSubmit"] {
    background: rgba(255,255,255,0.15) !important;
    border: 1.5px solid rgba(255,255,255,0.35) !important;
    border-radius: 50% !important;
    width: 46px !important;
    height: 46px !important;
    padding: 0 !important;
    color: rgba(255,255,255,0.90) !important;
    font-size: 18px !important;
    backdrop-filter: blur(10px) !important;
    transition: all 0.2s !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin-top: 0 !important;
}

button[kind="secondaryFormSubmit"]:hover,
button[data-testid="baseButton-secondaryFormSubmit"]:hover {
    background: rgba(255,255,255,0.28) !important;
    border-color: rgba(255,255,255,0.60) !important;
    transform: scale(1.08) !important;
}

/* ══════════════════════════════════════
   DIVIDER & MISC
══════════════════════════════════════ */
hr {
    border-color: rgba(255,255,255,0.12) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.25);
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# CUSTOM HEADER HTML
# ==========================================
st.markdown("""
<div class="custom-header">
    <div class="header-left-text">
        <span class="status-dot"></span>
        AI TOEIC Trợ Lý
    </div>
    <div class="header-center-logo">
        <div class="logo-circle">🎓</div>
        <span class="logo-title">AI TOEIC</span>
    </div>
    <div style="width:120px"></div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# SESSION STATE INIT
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()

if "system_ready" not in st.session_state:
    st.session_state.system_ready = False

if "show_chips" not in st.session_state:
    st.session_state.show_chips = True

if "chip_query" not in st.session_state:
    st.session_state.chip_query = None


# ==========================================
# LOGGING
# ==========================================
for lib in ("transformers", "httpx", "httpcore", "urllib3", "sentence_transformers"):
    logging.getLogger(lib).setLevel(logging.ERROR)


# ==========================================
# WARM-UP MODELS
# ==========================================
@st.cache_resource(show_spinner=False)
def load_system():
    db = get_vector_db()
    retriever = _get_bm25_retriever() if HYBRID_SEARCH else None
    return db, retriever


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Cài đặt")
    st.markdown("---")

    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory = ConversationMemory()
        st.session_state.show_chips = True
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="color:rgba(255,255,255,0.6); font-size:12px; line-height:1.6;">
    <b style="color:rgba(255,255,255,0.85)">AI TOEIC Chatbot</b><br>
    Gia sư TOEIC AI thông minh<br><br>
    🔍 Hybrid Search (BM25 + Chroma)<br>
    ⚡ Streaming Response<br>
    🧠 Conversation Memory
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# LOAD SYSTEM
# ==========================================
with st.spinner("⏳ Đang khởi động hệ thống..."):
    try:
        load_system()
        st.session_state.system_ready = True
    except Exception as e:
        st.error(f"❌ Khởi động thất bại: {e}\n\nHãy đảm bảo **Ollama đang chạy**: `ollama serve`")
        st.stop()


# ==========================================
# WELCOME MESSAGE
# ==========================================
if not st.session_state.messages:
    welcome = (
        "Xin chào! 👋 Tôi là **Gia sư TOEIC AI** của bạn.\n\n"
        "Tôi có thể giúp bạn:\n"
        "- Luyện tập **Part 5 Ngữ pháp** và giải thích chi tiết\n"
        "- Cung cấp **từ vựng TOEIC** phổ biến kèm ví dụ\n"
        "- Hướng dẫn **chiến thuật làm bài** hiệu quả\n"
        "- Trả lời mọi thắc mắc về **ngữ pháp tiếng Anh**\n\n"
        "Bạn muốn bắt đầu từ đâu? 😊"
    )
    st.session_state.messages.append({"role": "assistant", "content": welcome})


# ==========================================
# RENDER CHAT HISTORY
# ==========================================
for msg in st.session_state.messages:
    avatar = "🎓" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


# ==========================================
# SUGGESTION CHIPS
# ==========================================
CHIPS = [
    ("📖 Mệnh đề quan hệ",       "Mệnh đề quan hệ là gì?"),
    ("❓ Cấu trúc ĐK loại 3",   "Cấu trúc câu điều kiện loại 3 là gì?"),
    ("📝 'Look forward to'",    "'Look forward to' có nghĩa là gì?"),
    ("⏳ Thì Quá khứ đơn",      "Công thức THÌ QUÁ KHỨ ĐƠN là gì?"),
]

if st.session_state.show_chips and len(st.session_state.messages) <= 1:
    cols = st.columns(len(CHIPS))
    for col, (label, query) in zip(cols, CHIPS):
        with col:
            if st.button(label, key=f"chip_{label}"):
                st.session_state.chip_query = query
                st.session_state.show_chips = False
                st.rerun()


# ==========================================
# PROCESS CHIP QUERY
# ==========================================
pending_query = None
if st.session_state.chip_query:
    pending_query = st.session_state.chip_query
    st.session_state.chip_query = None


# ==========================================
# INPUT FORM
# ==========================================
with st.container():
    with st.form(key="chat_form", clear_on_submit=True, border=False):
        col_input, col_send, col_clear = st.columns([10, 1, 1])

        with col_input:
            typed = st.text_input(
                label="msg",
                placeholder="Nhập tin nhắn của bạn...",
                key="text_input",
                label_visibility="collapsed",
            )

        with col_send:
            send_clicked = st.form_submit_button("➤", type="primary")

        with col_clear:
            clear_clicked = st.form_submit_button("🗑", type="secondary")

user_input = None
if send_clicked and typed:
    user_input = typed.strip()
elif pending_query:
    user_input = pending_query

if clear_clicked:
    st.session_state.messages = []
    st.session_state.memory = ConversationMemory()
    st.session_state.show_chips = True
    st.rerun()


# ==========================================
# HANDLE USER MESSAGE + STREAM RESPONSE
# ==========================================
if user_input:
    st.session_state.show_chips = False

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    history_str = st.session_state.memory.format_for_prompt()

    with st.chat_message("assistant", avatar="🎓"):
        try:
            response = st.write_stream(
                stream_bot_response(user_input, history_str)
            )
        except ConnectionError:
            response = "⚠️ **Không kết nối được Ollama.**\n\nHãy chạy: `ollama serve` trong cửa sổ khác."
            st.error(response)
        except Exception as e:
            response = f"⚠️ **Lỗi:** {e}"
            st.error(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.memory.add_turn(user_input, response)
    st.rerun()
