"""
APP.PY - Giao diện Web UI bằng Streamlit cho TOEIC Master Chatbot
Chạy lệnh: streamlit run app.py
"""

import sys
import streamlit as st
import logging

# Thiết lập cấu hình trang Streamlit đầu tiên (Bắt buộc phải gọi trước các phần khác)
st.set_page_config(
    page_title="TOEIC Master — Gia sư TOEIC AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
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
# CUSTOM CSS FOR PREMIUM DESIGN (Glassmorphism & SLEEK DARK ACCENTS)
# ==========================================
st.markdown("""
<style>
    /* Tổng thể & Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Tùy chỉnh Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Thiết kế Glassmorphic card cho tiêu đề */
    .header-container {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 25px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    .header-title {
        background: linear-gradient(to right, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 5px;
        letter-spacing: -0.025em;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 400;
    }
    
    /* Customize Chat Input */
    [data-testid="stChatInput"] {
        background-color: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
    }
    
    /* Custom status & info box */
    .info-card {
        background: rgba(129, 140, 248, 0.08);
        border-left: 4px solid #818cf8;
        padding: 12px 16px;
        border-radius: 0 12px 12px 0;
        margin-bottom: 15px;
    }
    
    /* Căn chỉnh lại padding của chat bubbles */
    .stChatMessage {
        border-radius: 16px !important;
        padding: 15px !important;
        margin-bottom: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.02) !important;
    }
    
    /* User chat bubble */
    [data-testid="stChatMessageUser"] {
        background-color: rgba(56, 189, 248, 0.08) !important;
        border-left: 4px solid #38bdf8 !important;
    }
    
    /* Bot chat bubble */
    [data-testid="stChatMessageAssistant"] {
        background-color: rgba(129, 140, 248, 0.05) !important;
        border-left: 4px solid #818cf8 !important;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# LAZY SYSTEM WARM-UP (Streamlit cache)
# ==========================================
@st.cache_resource(show_spinner=False)
def initialize_engine():
    """Tải FAISS Database và khởi tạo BM25 index."""
    try:
        db = get_vector_db()
        doc_count = len(db.docstore._dict)
        # Khởi động sẵn BM25 nếu cấu hình hybrid được bật
        bm25_retriever = _get_bm25_retriever()
        return True, doc_count, None
    except FileNotFoundError as e:
        return False, 0, f"Không tìm thấy thư mục dữ liệu RAG: {e}"
    except Exception as e:
        return False, 0, f"Lỗi khởi động hệ thống: {e}"


# ==========================================
# MAIN APP FLOW
# ==========================================
def main():
    # 1. Hiển thị Header
    st.markdown("""
    <div class="header-container">
        <div class="header-title">🎓 TOEIC MASTER</div>
        <div class="header-subtitle">Gia sư TOEIC AI thông minh — Luyện thi cá nhân hóa hoàn toàn offline</div>
    </div>
    """, unsafe_allow_html=True)

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

    # 4. SIDEBAR CONFIGURATION
    st.sidebar.title("⚙️ Cấu hình hệ thống")
    
    st.sidebar.markdown(f"""
    <div class="info-card">
        <strong>📚 Kho tri thức RAG:</strong><br>
        Đang chứa <b>{doc_count}</b> chunks tài liệu TOEIC/Ngữ pháp.
    </div>
    """, unsafe_allow_html=True)

    # Các tham số tinh chỉnh LLM & RAG
    st.sidebar.subheader("🔍 Thiết lập Tìm kiếm & LLM")
    
    # Model name
    model_name = st.sidebar.text_input(
        "Tên mô hình Ollama", 
        value=OLLAMA_MODEL_NAME, 
        help="Đảm bảo Ollama đang chạy mô hình này (ví dụ: gemma2, qwen2.5:7b)"
    )
    
    # Toggle Hybrid search
    is_hybrid = st.sidebar.toggle(
        "Bật Hybrid Search", 
        value=HYBRID_SEARCH,
        help="Kết hợp FAISS (ngữ nghĩa) và BM25 (từ khóa chính xác) để nâng cao chất lượng tìm kiếm."
    )
    
    # Score Threshold slider
    th_score = st.sidebar.slider(
        "Ngưỡng Lọc (Score Threshold)", 
        min_value=0.5, 
        max_value=2.0, 
        value=SCORE_THRESHOLD, 
        step=0.1,
        help="Chỉ áp dụng khi tắt Hybrid Search. Giá trị càng nhỏ lọc càng khắt khe."
    )
    
    # Temperature slider
    temp = st.sidebar.slider(
        "Độ sáng tạo (Temperature)", 
        min_value=0.0, 
        max_value=1.0, 
        value=TEMPERATURE, 
        step=0.05,
        help="Thấp hơn = câu trả lời tập trung và logic; Cao hơn = sáng tạo và đa dạng."
    )

    # Dynamic cập nhật cấu hình vào config hệ thống thông qua override
    import src.config as cfg
    cfg.OLLAMA_MODEL_NAME = model_name
    cfg.HYBRID_SEARCH = is_hybrid
    cfg.SCORE_THRESHOLD = th_score
    cfg.TEMPERATURE = temp

    # Nút dọn dẹp lịch sử
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.chat_messages = []
        st.session_state.bot_memory.clear()
        st.toast("Đã xóa sạch lịch sử trò chuyện!", icon="🗑️")
        st.rerun()

    # 5. HIỂN THỊ LỊCH SỬ CHAT TRÊN GIAO DIỆN
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 6. NHẬN CÂU HỎI MỚI TỪ HỌC VIÊN
    if user_query := st.chat_input("Hỏi tôi về ngữ pháp, từ vựng hoặc đề thi TOEIC Part 5..."):
        
        # Hiển thị câu hỏi của user
        st.session_state.chat_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Lấy lịch sử dạng chuỗi đưa vào context prompt
        history_str = st.session_state.bot_memory.format_for_prompt()

        # Gọi LLM sinh phản hồi
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            try:
                if STREAM_OUTPUT:
                    # Stream phản hồi thời gian thực
                    token_generator = stream_bot_response(user_query, history_str)
                    for token in token_generator:
                        full_response += token
                        response_placeholder.markdown(full_response + "▌")
                    # Hiển thị text sạch sau khi stream hoàn tất
                    response_placeholder.markdown(full_response)
                else:
                    # Xử lý blocking
                    with st.spinner("Đang suy luận..."):
                        full_response = get_bot_response(user_query, history_str)
                        response_placeholder.markdown(full_response)

                # Lưu vào bộ nhớ hiển thị UI
                st.session_state.chat_messages.append({"role": "assistant", "content": full_response})
                # Lưu vào bộ nhớ context của AI
                st.session_state.bot_memory.add_turn(user_query, full_response)

            except Exception as e:
                err_text = f"❌ **Đã xảy ra lỗi:** {e}\n\n*Vui lòng kiểm tra lại kết nối Ollama bằng lệnh `ollama serve`.*"
                st.error(err_text)
                logger.error("Lỗi sinh phản hồi: %s", e, exc_info=True)


if __name__ == "__main__":
    main()
