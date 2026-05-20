"""
ASK.PY - Điều phối toàn bộ pipeline: RAG → Prompt → LLM → Output

Cải tiến:
- Chain Singleton: build chain 1 lần duy nhất, tái sử dụng cho mọi câu hỏi.
- Streaming: stream_bot_response() yield từng token thay vì chờ toàn bộ.
"""

import logging
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.model import get_llm
from src.rag import retrieve_context
from src.config import SYSTEM_PROMPT_TOEIC, PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

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
        prompt = PromptTemplate(
            input_variables=["system_prompt", "context", "question"],
            template=PROMPT_TEMPLATE,
        )
        _chain = (
            {
                "system_prompt": lambda x: SYSTEM_PROMPT_TOEIC,
                "context": lambda x: retrieve_context(x),  # Truy xuất FAISS
                "question": RunnablePassthrough(),           # Câu hỏi gốc
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.info("[ASK] Chain san sang.")
    return _chain


def get_bot_response(user_query: str) -> str:
    """
    Nhận câu hỏi, trả về toàn bộ câu trả lời dưới dạng chuỗi (blocking).
    Dùng khi STREAM_OUTPUT = False.
    """
    logger.info("[ASK] Nhan cau hoi (invoke mode): %s", user_query[:80])
    return _get_chain().invoke(user_query)


def stream_bot_response(user_query: str):
    """
    Generator: yield từng token khi LLM sinh ra.
    Dùng khi STREAM_OUTPUT = True — người dùng thấy text xuất hiện dần.
    """
    logger.info("[ASK] Nhan cau hoi (stream mode): %s", user_query[:80])
    yield from _get_chain().stream(user_query)


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