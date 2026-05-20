"""
ASK.PY - Điều phối toàn bộ pipeline: RAG → Prompt → LLM → Output
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.model import get_llm
from src.rag import retrieve_context
from src.config import SYSTEM_PROMPT_TOEIC, PROMPT_TEMPLATE


def get_bot_response(user_query: str) -> str:
    """
    Nhận câu hỏi của học viên, truy xuất tài liệu từ FAISS,
    ghép prompt và gọi LLM để trả về câu trả lời có cấu trúc.
    """
    # 1. Khởi tạo LLM và Prompt Template
    llm = get_llm()
    prompt = PromptTemplate(
        input_variables=["system_prompt", "context", "question"],
        template=PROMPT_TEMPLATE,
    )

    # 2. Định nghĩa luồng LCEL: Retriever → Prompt → LLM → Output Parser
    chain = (
        {
            "system_prompt": lambda x: SYSTEM_PROMPT_TOEIC,
            "context": lambda x: retrieve_context(x),  # Truy xuất tài liệu FAISS
            "question": RunnablePassthrough(),           # Câu hỏi gốc
        }
        | prompt
        | llm
        | StrOutputParser()  # Bóc tách text từ response AI
    )

    # 3. Kích hoạt toàn bộ chuỗi
    print("Đang suy luận...\n")
    return chain.invoke(user_query)


# Chạy thử trực tiếp trên Terminal
if __name__ == "__main__":
    while True:
        cau_hoi = input("\n🧑 Bạn: ")
        if cau_hoi.lower() in ('quit', 'exit', 'thoat'):
            print("Tạm biệt! Chúc bạn học tốt 👋")
            break
        tra_loi = get_bot_response(cau_hoi)
        print(f"\n🤖 Bot: {tra_loi}")