"""
APP.PY - Giao diện Web UI bằng Streamlit cho TOEIC Master Chatbot
Chạy lệnh: streamlit run app.py
"""

# Khắc phục lỗi xung đột giữa Streamlit File Watcher và PyTorch (torch.classes)
try:
    import torch
    torch.classes.__path__ = []
except Exception:
    pass

# Tắt các cảnh báo (UserWarning) lặp đi lặp lại từ thư viện transformers khi Streamlit quét file
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Accessing.*__path__.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*alias will be removed.*")

import sys
import streamlit as st
import logging

# Thiết lập cấu hình trang Streamlit centered để tối ưu giao diện giống Gemini
st.set_page_config(
    page_title="TOEIC Master — Gia sư TOEIC AI",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

from src.ask import get_bot_response, stream_bot_response
from src.rag import get_vector_db, _get_bm25_retriever
from src.config import (
    OLLAMA_MODEL_NAME,
    HYBRID_SEARCH,
    SCORE_THRESHOLD,
    TEMPERATURE,
    STREAM_OUTPUT
)
from src.memory import ConversationMemory

# Cấu hình logging
logger = logging.getLogger("streamlit_app")


# ==========================================
# CUSTOM CSS FOR GEMINI-STYLE & NARROW DESIGN (No avatars, glassmorphism, right-aligned user)
# ==========================================
custom_css = """
<style>
    /* Import font Outfit */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    /* Main App Reset */
    .stApp {
        background: transparent !important;
        font-family: 'Outfit', sans-serif !important;
    }

    /* Video Background */
    #bg-video {
        position: fixed;
        right: 0;
        bottom: 0;
        min-width: 100%;
        min-height: 100%;
        width: auto;
        height: auto;
        z-index: -100;
        object-fit: cover;
        pointer-events: none;
    }

    #bg-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle at center, rgba(15, 23, 42, 0.45) 0%, rgba(8, 10, 20, 0.85) 100%);
        backdrop-filter: blur(8px);
        z-index: -99;
    }

    /* Hide Sidebar & Default Controls */
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }

    header, [data-testid="stHeader"] {
        background: transparent !important;
    }

    footer {
        visibility: hidden !important;
    }

    /* Container Padding & Narrow Max-Width (Tăng kích thước to hơn tí lên 760px) */
    .block-container, [data-testid="stAppViewBlockContainer"] {
        max-width: 760px !important;
        padding-top: 3rem !important;
        padding-bottom: 7rem !important;
        margin: 0 auto !important;
    }

    /* Badge Styling */
    .badge-container {
        display: flex;
        justify-content: center;
        margin-bottom: 1.2rem;
    }
    .hero-badge {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.15) 100%);
        border: 1px solid rgba(168, 85, 247, 0.3);
        border-radius: 50px;
        padding: 6px 18px;
        color: #c084fc;
        font-size: 0.85rem;
        font-weight: 500;
        letter-spacing: 0.05em;
        box-shadow: 0 4px 15px rgba(168, 85, 247, 0.15);
    }

    /* Hero Section Text */
    .hero-container {
        text-align: center;
        margin-top: 4vh;
        margin-bottom: 2rem;
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #ffffff 40%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.8rem;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.15rem;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* Style the Streamlit Form to look like the central card */
    div[data-testid="stForm"] {
        background: rgba(15, 23, 42, 0.35) !important;
        backdrop-filter: blur(25px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 24px !important;
        padding: 24px !important;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3) !important;
        margin-bottom: 1.5rem !important;
        width: 100% !important;
    }

    .input-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.8rem;
        color: #64748b;
        margin-bottom: 15px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding-bottom: 12px;
    }
    .header-left {
        color: #cbd5e1;
        font-weight: 500;
    }
    .header-right {
        font-weight: 500;
        color: #818cf8;
    }

    /* Style text input inside form */
    div[data-testid="stForm"] div[data-testid="stTextInput"] input {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
        font-size: 1rem !important;
        padding: 12px 16px !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stForm"] div[data-testid="stTextInput"] input:focus {
        border-color: rgba(129, 140, 248, 0.4) !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
        box-shadow: 0 0 15px rgba(129, 140, 248, 0.1) !important;
    }

    /* Form Buttons styling */
    .clear-btn-wrapper button {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #94a3b8 !important;
        border-radius: 12px !important;
        padding: 8px 16px !important;
        font-size: 0.85rem !important;
        transition: all 0.2s ease !important;
    }
    .clear-btn-wrapper button:hover {
        background: rgba(239, 68, 68, 0.12) !important;
        border-color: rgba(239, 68, 68, 0.3) !important;
        color: #f87171 !important;
    }

    .send-btn-wrapper button {
        background: #ffffff !important;
        border: none !important;
        color: #0f172a !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 15px rgba(255, 255, 255, 0.15) !important;
        width: 100% !important;
        height: auto !important;
    }
    .send-btn-wrapper button:hover {
        background: #cbd5e1 !important;
        transform: scale(1.02) !important;
    }

    /* Suggestion Title & Chips Group */
    .suggestion-group-title {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        text-align: center;
    }
    .chip-group div.stButton > button {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 16px !important;
        color: #cbd5e1 !important;
        font-size: 0.85rem !important;
        padding: 8px 16px !important;
        transition: all 0.2s ease !important;
        white-space: normal !important;
        height: auto !important;
        min-height: 40px !important;
    }
    .chip-group div.stButton > button:hover {
        background: rgba(129, 140, 248, 0.15) !important;
        border-color: rgba(129, 140, 248, 0.4) !important;
        color: #a5b4fc !important;
        transform: translateY(-2px) !important;
    }

    /* Custom Chat Bubble Container */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0px !important;
        margin-bottom: 1.8rem !important;
    }

    /* Gỡ bỏ hoàn toàn mọi loại Avatar trong khung chat để sạch sẽ */
    [data-testid="stChatMessageAvatar"],
    [data-testid="chatAvatar"],
    div[class*="ChatMessageAvatar"],
    div[class*="chatAvatar"],
    img[class*="Avatar"] {
        display: none !important;
    }

    [data-testid="stChatMessageContent"] {
        padding: 0px !important;
        margin-left: 0px !important;
    }

    /* User Bubble (Căn phải, dạng box kính mờ gọn gàng) */
    [data-testid="stChatMessageUser"] {
        display: flex !important;
        justify-content: flex-end !important;
        flex-direction: row-reverse !important;
    }
    [data-testid="stChatMessageUser"] [data-testid="stChatMessageContent"] {
        background-color: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 20px 20px 4px 20px !important;
        color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(255, 255, 255, 0.02) !important;
        padding: 12px 20px !important;
        max-width: 85% !important;
    }

    /* Assistant Bubble (Phong cách Gemini: Trơn, không viền, không nền, chữ chạy trực tiếp trên nền video) */
    [data-testid="stChatMessageAssistant"] {
        display: flex !important;
        justify-content: flex-start !important;
    }
    [data-testid="stChatMessageAssistant"] [data-testid="stChatMessageContent"] {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0px !important;
        color: #f1f5f9 !important;
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
        width: 100% !important;
        max-width: 100% !important;
    }

    /* Định dạng độ rộng và căn giữa thanh Chat Input phía dưới (Active State) */
    div[data-testid="stChatInputContainer"] {
        position: fixed !important;
        bottom: 30px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        width: 760px !important;
        max-width: 90% !important;
        background: rgba(15, 23, 42, 0.45) !important;
        backdrop-filter: blur(25px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 20px !important;
        box-shadow: 0 -10px 40px rgba(0, 0, 0, 0.3) !important;
        padding: 10px 15px !important;
        z-index: 999 !important;
    }
    div[data-testid="stChatInputContainer"] textarea {
        background-color: transparent !important;
        color: #f8fafc !important;
        font-size: 1rem !important;
    }

    /* Float clear button styling (Active State) */
    .clear-btn-container {
        position: fixed;
        bottom: 95px;
        left: 50%;
        transform: translateX(-50%);
        width: 760px;
        max-width: 90%;
        z-index: 999;
        display: flex;
        justify-content: flex-start;
        pointer-events: none;
    }
    .clear-btn-container button {
        pointer-events: auto;
        background: rgba(239, 68, 68, 0.15) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        color: #f87171 !important;
        border-radius: 12px !important;
        padding: 6px 14px !important;
        font-size: 0.8rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 10px rgba(239, 68, 68, 0.1) !important;
    }
    .clear-btn-container button:hover {
        background: rgba(239, 68, 68, 0.25) !important;
        border-color: rgba(239, 68, 68, 0.5) !important;
        transform: translateY(-1px) !important;
    }
</style>
"""


# ==========================================
# LAZY SYSTEM WARM-UP (Streamlit cache)
# ==========================================
@st.cache_resource(show_spinner=False)
def initialize_engine():
    """Tải FAISS Database và khởi tạo BM25 index."""
    try:
        db = get_vector_db()
        doc_count = len(db.docstore._dict)
        _get_bm25_retriever()
        return True, doc_count, None
    except FileNotFoundError as e:
        return False, 0, f"Không tìm thấy thư mục dữ liệu RAG: {e}"
    except Exception as e:
        return False, 0, f"Lỗi khởi động hệ thống: {e}"


def safe_rerun():
    """Rerun Streamlit tương thích với nhiều phiên bản."""
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


# ==========================================
# MAIN APP FLOW
# ==========================================
def main():
    # 1. Khởi chạy background video & custom CSS
    st.markdown("""
    <video autoplay loop muted playsinline id="bg-video">
        <source src="https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260329_050842_be71947f-f16e-4a14-810c-06e83d23ddb5.mp4" type="video/mp4">
    </video>
    <div id="bg-overlay"></div>
    """, unsafe_allow_html=True)
    
    st.markdown(custom_css, unsafe_allow_html=True)

    # 2. Khởi tạo Engine RAG
    with st.spinner("🚀 Đang khởi chạy hệ thống (tải mô hình ngôn ngữ và vector index)..."):
        success, doc_count, err_msg = initialize_engine()

    if not success:
        st.error(err_msg)
        st.info("💡 Hãy đảm bảo bạn đã đặt thư mục `data/` chứa file `index.faiss` ở thư mục gốc.")
        st.stop()

    # 3. Quản lý Session State (Bộ nhớ cuộc trò chuyện & lịch sử chat UI)
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []  # Lưu lịch sử hiển thị trên UI

    if "bot_memory" not in st.session_state:
        st.session_state.bot_memory = ConversationMemory()  # Bộ nhớ context cho LLM

    # 4. Giao diện trống (Hero Section & Cụm Ô Nhập Liệu Trung Tâm)
    if len(st.session_state.chat_messages) == 0:
        # Badge & Hero Text
        st.markdown("""
        <div class="hero-container">
            <div class="badge-container">
                <span class="hero-badge">✦ Tự học TOEIC hiệu quả</span>
            </div>
            <div class="hero-title">TOEIC Master</div>
            <div class="hero-subtitle">Luyện thi cá nhân hóa hoàn toàn offline. Đặt câu hỏi để nhận ngay lời giải thích chi tiết.</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Central Input Box (Được bọc hoàn toàn bởi st.form để nhận background glassmorphism)
        with st.form("hero_form", clear_on_submit=True):
            st.markdown("""
            <div class="input-card-header">
                <span class="header-left">🎓 Gia sư TOEIC AI</span>
                <span class="header-right">⚡ Powered by Local LLM</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Khung nhập liệu nằm ngang
            col_input, col_send = st.columns([8.5, 1.5])
            with col_input:
                hero_query = st.text_input(
                    "Query Input",
                    placeholder="Hỏi tôi về ngữ pháp, từ vựng hoặc đề thi TOEIC Part 5...",
                    label_visibility="collapsed",
                    key="hero_query_input"
                )
            with col_send:
                st.markdown('<div class="send-btn-wrapper">', unsafe_allow_html=True)
                submit_clicked = st.form_submit_button("Send ↗")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Hàng dưới chứa nút Clear
            col_clear, col_info = st.columns([2.5, 9.5])
            with col_clear:
                st.markdown('<div class="clear-btn-wrapper">', unsafe_allow_html=True)
                clear_clicked = st.form_submit_button("🗑️ Clear")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Xử lý Clear History từ Hero Box
        if clear_clicked:
            st.session_state.chat_messages = []
            st.session_state.bot_memory.clear()
            st.toast("Đã xóa sạch lịch sử trò chuyện!", icon="🗑️")
            safe_rerun()
            
        # Xử lý Submit từ Hero Box
        if submit_clicked and hero_query:
            st.session_state.pending_query = hero_query
            safe_rerun()

        # Cụm Prompt Gợi ý (Chips)
        st.markdown('<div class="suggestion-group-title">💡 Gợi ý chủ đề câu hỏi:</div>', unsafe_allow_html=True)
        st.markdown('<div class="chip-group">', unsafe_allow_html=True)
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            if st.button("📝 Phân biệt Since & For trong HTHT", key="sug_since_for", use_container_width=True):
                st.session_state.pending_query = "Phân biệt cách dùng Since và For trong thì hiện tại hoàn thành, cho ví dụ minh họa."
                safe_rerun()
        with col_s2:
            if st.button("✏️ Giải thích cấu trúc Despite/Although", key="sug_despite_although", use_container_width=True):
                st.session_state.pending_query = "Giải thích cấu trúc ngữ pháp và cách dùng của Despite, In spite of, Although, Even though."
                safe_rerun()
        with col_s3:
            if st.button("📖 Đề thi Part 5: Từ loại (Gerund)", key="sug_gerund", use_container_width=True):
                st.session_state.pending_query = "Giải thích quy tắc khi nào dùng Danh động từ (Gerund) sau giới từ trong bài thi TOEIC Part 5."
                safe_rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # 5. Giao diện Chat đã có tin nhắn
    else:
        # Hiển thị lịch sử chat
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        # Nút clear nổi ở góc dưới khi chat đang hoạt động
        st.markdown('<div class="clear-btn-container">', unsafe_allow_html=True)
        if st.button("🗑️ Clear", key="clear_chat_active"):
            st.session_state.chat_messages = []
            st.session_state.bot_memory.clear()
            st.toast("Đã xóa sạch lịch sử trò chuyện!", icon="🗑️")
            safe_rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # 6. Xử lý nhận câu hỏi mới (chỉ hiển thị chat_input khi đã có tin nhắn để tránh bị trùng lặp)
    user_query = None
    if len(st.session_state.chat_messages) > 0:
        user_query = st.chat_input("Hỏi tôi về ngữ pháp, từ vựng hoặc đề thi TOEIC Part 5...")
    
    if "pending_query" in st.session_state and st.session_state.pending_query:
        user_query = st.session_state.pending_query
        del st.session_state.pending_query  # Xóa câu hỏi chờ

    if user_query:
        # Thêm câu hỏi của user vào danh sách hiển thị
        st.session_state.chat_messages.append({"role": "user", "content": user_query})
        st.session_state.processing_query = user_query
        safe_rerun()

    # Xử lý sinh phản hồi từ LLM
    if "processing_query" in st.session_state and st.session_state.processing_query:
        query_to_process = st.session_state.processing_query
        del st.session_state.processing_query

        # Hiển thị toàn bộ lịch sử và tin nhắn mới trước khi stream
        for msg in st.session_state.chat_messages[:-1]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Hiển thị tin nhắn user cuối cùng
        with st.chat_message("user"):
            st.markdown(query_to_process)

        # Kiểm tra lệnh thoát
        if query_to_process.lower() in ('exit', 'quit', 'thoat', 'thoát'):
            goodbye_msg = "👋 Cảm ơn bạn đã học cùng TOEIC Master! Bạn có thể đóng tab trình duyệt này để kết thúc phiên học tập. Hẹn gặp lại bạn và chúc bạn thi tốt! 🎯"
            st.session_state.chat_messages.append({"role": "assistant", "content": goodbye_msg})
            with st.chat_message("assistant"):
                st.markdown(goodbye_msg)
            st.stop()

        # Lấy lịch sử dạng chuỗi đưa vào context prompt
        history_str = st.session_state.bot_memory.format_for_prompt()

        # Gọi LLM sinh phản hồi
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            try:
                if STREAM_OUTPUT:
                    token_generator = stream_bot_response(query_to_process, history_str)
                    for token in token_generator:
                        full_response += token
                        response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)
                else:
                    with st.spinner("Đang suy luận..."):
                        full_response = get_bot_response(query_to_process, history_str)
                        response_placeholder.markdown(full_response)

                # Lưu vào bộ nhớ hiển thị UI
                st.session_state.chat_messages.append({"role": "assistant", "content": full_response})
                # Lưu vào bộ nhớ context của AI
                st.session_state.bot_memory.add_turn(query_to_process, full_response)

            except Exception as e:
                err_text = f"❌ **Đã xảy ra lỗi:** {e}\n\n*Vui lòng kiểm tra lại kết nối Ollama bằng lệnh `ollama serve`.*"
                st.error(err_text)
                logger.error("Lỗi sinh phản hồi: %s", e, exc_info=True)
            
            safe_rerun()


if __name__ == "__main__":
    main()
