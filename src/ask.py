"""
ASK.PY - Điều phối toàn bộ pipeline: RAG → Prompt → LLM → Output

Cải tiến:
- Chain Singleton: build chain 1 lần duy nhất, tái sử dụng cho mọi câu hỏi.
- Streaming: stream_bot_response() yield từng token thay vì chờ toàn bộ.
"""

import re
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.model import get_llm
from src.rag import retrieve_context
from src.config import SYSTEM_PROMPT_TOEIC, PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

# ==========================================
# QUERY PREPROCESSOR (tối ưu retrieval cho MCQ)
# ==========================================
def _build_retrieval_query(question: str) -> str:
    """
    Xây dựng query tối ưu cho retriever:
    - Nếu là MCQ (có A), B), C), D)): xóa các dòng đáp án và dấu ______
      để BM25 khớp được từ khóa ngữ pháp thay vì các lựa chọn A/B/C/D.
    - Câu lý thuyết thường: dùng nguyên xi.
    """
    # Phát hiện MCQ bằng dấu hiệu "A)" hoặc "A." ở đầu dòng
    is_mcq = bool(re.search(r'(?m)^\s*[A-D][).]', question))

    if is_mcq:
        lines = question.strip().split('\n')
        # Chỉ giữ lại dòng câu chính, bỏ dòng đáp án
        main_lines = [
            l for l in lines
            if not re.match(r'^\s*[A-D][).\s]', l)
        ]
        query = ' '.join(main_lines).strip()
        # Xóa dấu ______ (nhiễu cho BM25, không mang thông tin ngữ pháp)
        query = re.sub(r'_{2,}', '', query).strip()
        logger.debug("[ASK] MCQ query được xử lý: %s", query[:80])
        return query

    return question


# ==========================================
# CONTEXTUAL QUERY REFORMULATION
# ==========================================
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_template(
    "Dựa vào lịch sử trò chuyện và câu hỏi mới của người dùng dưới đây, "
    "hãy viết lại câu hỏi mới thành một câu hỏi độc lập (standalone query) chứa đầy đủ ngữ cảnh từ lịch sử. "
    "Nếu câu hỏi mới đang ám chỉ điều gì đó trong lịch sử (như 'cho tôi bài tập', 'giải thích lại phần đó'), hãy thêm chủ đề đó vào. "
    "Tuyệt đối KHÔNG trả lời câu hỏi, CHỈ viết lại câu hỏi thành 1 câu duy nhất.\n\n"
    "Lịch sử trò chuyện:\n{history}\n\n"
    "Câu hỏi mới: {question}\n\n"
    "Câu hỏi độc lập:"
)

def _get_standalone_question(inputs: dict) -> str:
    """Nếu có lịch sử, dùng LLM để dịch lại câu hỏi cho đầy đủ nghĩa trước khi đem đi search."""
    question = inputs["question"]
    history = inputs.get("history", "")
    if not history.strip():
        return question  # Không có lịch sử thì giữ nguyên
    
    llm = get_llm()
    chain = CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    standalone = chain.invoke({"history": history, "question": question}).strip()
    
    logger.info("[ASK] Cau hoi goc: %s", question)
    logger.info("[ASK] Cau hoi doc lap (Standalone): %s", standalone)
    return standalone

# ==========================================
# SINGLETON CHAIN (build 1 lần duy nhất)
# ==========================================
_chain = None


def _get_chain():
    """
    Khởi tạo và cache LangChain LCEL pipeline.
    Singleton: tránh rebuild PromptTemplate + bind LLM mỗi lần hỏi.
    """
    global _chain
    if _chain is None:
        logger.info("[ASK] Khoi tao chain lan dau...")
        llm = get_llm()
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        _chain = (
            RunnablePassthrough.assign(
                standalone_question=_get_standalone_question
            )
            | {
                "system_prompt": lambda x: SYSTEM_PROMPT_TOEIC,
                # Dùng query đã viết lại để tìm kiếm tài liệu chính xác hơn
                "context": lambda x: retrieve_context(
                    _build_retrieval_query(x["standalone_question"])
                ),
                "history": lambda x: x["history"],
                "question": lambda x: x["standalone_question"], # Dùng câu hỏi đã được làm rõ ngữ cảnh để LLM không bị lạc đề
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.info("[ASK] Chain san sang.")
    return _chain


def get_bot_response(user_query: str, history: str = "") -> str:
    """
    Nhận câu hỏi, trả về toàn bộ câu trả lời dưới dạng chuỗi (blocking).
    Dùng khi STREAM_OUTPUT = False.
    """
    logger.info("[ASK] Nhan cau hoi (invoke mode): %s", user_query[:80])
    return _get_chain().invoke({"question": user_query, "history": history})


def stream_bot_response(user_query: str, history: str = ""):
    """
    Generator: yield từng token khi LLM sinh ra.
    Dùng khi STREAM_OUTPUT = True — người dùng thấy text xuất hiện dần.
    """
    logger.info("[ASK] Nhan cau hoi (stream mode): %s", user_query[:80])
    yield from _get_chain().stream({"question": user_query, "history": history})


# Chạy thử trực tiếp trên Terminal
if __name__ == "__main__":
    from src.config import STREAM_OUTPUT
    while True:
        cau_hoi = input("\n🧑 Bạn: ")
        if cau_hoi.lower() in ('quit', 'exit', 'thoat'):
            print("Tạm biệt! Chúc bạn học tốt 👋")
            break
        print("\n🤖 Bot: ", end="", flush=True)
        if STREAM_OUTPUT:
            for token in stream_bot_response(cau_hoi):
                print(token, end="", flush=True)
            print()
        else:
            print(get_bot_response(cau_hoi))