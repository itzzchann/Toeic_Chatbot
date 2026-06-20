"""
MEMORY.PY - Quản lý lịch sử hội thoại (Conversational Memory)

Lưu trữ tối đa MAX_HISTORY_TURNS lượt hội thoại gần nhất.
Khi đầy, tự động xóa lượt cũ nhất (sliding window).
"""

import logging
from collections import deque
import threading
from src.config import MAX_HISTORY_TURNS, MEMORY_ENABLED
from src.model import get_small_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    "Bạn là một trợ lý trí nhớ. Hãy cập nhật bản tóm tắt lịch sử trò chuyện bằng cách gộp thông tin từ lượt chat mới dưới đây vào. "
    "Yêu cầu:\n"
    "- Giữ cho tóm tắt thật ngắn gọn, dưới 3-4 câu.\n"
    "- KHÔNG giải thích dài dòng, CHỈ ghi nhớ các mốc thông tin quan trọng mang tính cá nhân (ví dụ: mục tiêu điểm số của học viên, tên học viên, điểm yếu, các chủ đề cũ đã học).\n"
    "- Tuyệt đối viết bằng Tiếng Việt.\n\n"
    "Tóm tắt cũ:\n{summary}\n\n"
    "Lượt chat mới:\nHọc viên: {question}\nGia sư: {answer}\n\n"
    "Tóm tắt mới:"
)


class ConversationMemory:
    """
    Summary-Buffer conversational memory (Trí nhớ Đệm - Tóm tắt).
    
    Lưu giữ tối đa `max_turns` lượt chat gần nhất để AI bắt ngữ cảnh chính xác.
    Các lượt chat cũ hơn sẽ được LLM tự động tóm tắt lại thành một đoạn văn ngắn
    và lưu vĩnh viễn, tránh làm tràn bộ nhớ mà vẫn không bị mất thông tin cũ.
    """

    def __init__(self, max_turns: int = 3): # Giữ 3 lượt gần nhất, còn lại tóm tắt
        self.max_turns = max_turns
        self._history: deque[tuple[str, str]] = deque()
        self.summary = ""
        logger.info("[Memory] Khoi tao ConversationMemory (max_turns=%d, Summary Enabled).", max_turns)

    def add_turn(self, question: str, answer: str) -> None:
        """Lưu 1 lượt hội thoại (câu hỏi + câu trả lời)."""
        if not MEMORY_ENABLED:
            return
            
        self._history.append((question.strip(), answer.strip()))
        
        # Nếu vượt quá số lượt raw buffer, pop lượt cũ nhất và tóm tắt nó
        if len(self._history) > self.max_turns:
            oldest_q, oldest_a = self._history.popleft()
            # Chạy việc tóm tắt ở background thread để không làm chậm luồng chat chính
            thread = threading.Thread(target=self._update_summary, args=(oldest_q, oldest_a))
            thread.start()
            
        logger.info("[Memory] Da luu luot. Buffer: %d/%d. Summary len: %d", len(self._history), self.max_turns, len(self.summary))

    def _update_summary(self, question: str, answer: str):
        """Dùng LLM để gộp lượt chat bị đẩy ra khỏi buffer vào đoạn tóm tắt."""
        try:
            llm = get_small_llm()
            logger.info("[Memory] Đang dùng mô hình phụ (%s) để tóm tắt trí nhớ ngầm...", getattr(llm, 'model', 'unknown'))
            chain = SUMMARY_PROMPT | llm | StrOutputParser()
            new_summary = chain.invoke({
                "summary": self.summary if self.summary else "Chưa có thông tin gì.",
                "question": question,
                # Rút gọn câu trả lời của gia sư để tóm tắt nhanh hơn, chỉ cần ý chính
                "answer": answer[:250] + "..." if len(answer) > 250 else answer
            }).strip()
            self.summary = new_summary
            logger.info("[Memory] Da cap nhat Summary thanh cong.")
        except Exception as e:
            logger.error("[Memory] Loi khi update summary: %s", e)

    def format_for_prompt(self) -> str:
        """
        Định dạng lịch sử thành text đưa vào prompt.
        Bao gồm: [Tóm tắt lịch sử cũ] + [Các lượt chat gần nhất]
        """
        if not MEMORY_ENABLED:
            return ""
            
        lines = []
        if self.summary:
            lines.append(f"[TÓM TẮT THÔNG TIN CŨ TỪ HỌC VIÊN]\n{self.summary}\n")
            
        for i, (q, a) in enumerate(self._history, start=1):
            lines.append(f"[Lượt gần đây {i}]")
            lines.append(f"Học viên: {q}")
            # Giới hạn độ dài câu trả lời trong history để tránh prompt quá dài
            short_answer = a if len(a) <= 500 else a[:500] + "...(rút gọn)"
            lines.append(f"Gia sư: {short_answer}")
            lines.append("")   # Dòng trống giữa các lượt

        return "\n".join(lines).strip()

    def clear(self) -> None:
        """Xóa toàn bộ lịch sử hội thoại và tóm tắt."""
        count = len(self._history)
        self._history.clear()
        self.summary = ""
        logger.info("[Memory] Da xoa toan bo %d luot lich su va summary.", count)

    def __len__(self) -> int:
        return len(self._history) + (1 if self.summary else 0)

    def __bool__(self) -> bool:
        return len(self._history) > 0 or bool(self.summary)
