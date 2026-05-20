"""
MEMORY.PY - Quản lý lịch sử hội thoại (Conversational Memory)

Lưu trữ tối đa MAX_HISTORY_TURNS lượt hội thoại gần nhất.
Khi đầy, tự động xóa lượt cũ nhất (sliding window).
"""

import logging
from collections import deque
from src.config import MAX_HISTORY_TURNS, MEMORY_ENABLED

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Sliding-window conversational memory.

    Lưu tối đa `max_turns` cặp (câu hỏi, câu trả lời).
    Khi số lượt vượt quá max_turns, lượt cũ nhất tự động bị xóa.
    """

    def __init__(self, max_turns: int = MAX_HISTORY_TURNS):
        self.max_turns = max_turns
        # deque với maxlen tự động drop phần tử cũ khi đầy
        self._history: deque[tuple[str, str]] = deque(maxlen=max_turns)
        logger.info("[Memory] Khoi tao ConversationMemory (max_turns=%d).", max_turns)

    def add_turn(self, question: str, answer: str) -> None:
        """Lưu 1 lượt hội thoại (câu hỏi + câu trả lời)."""
        if not MEMORY_ENABLED:
            return
        self._history.append((question.strip(), answer.strip()))
        logger.info("[Memory] Da luu luot %d/%d.", len(self._history), self.max_turns)

    def format_for_prompt(self) -> str:
        """
        Định dạng lịch sử thành text đưa vào prompt.
        Trả về chuỗi rỗng nếu chưa có lịch sử.
        """
        if not self._history or not MEMORY_ENABLED:
            return ""

        lines = []
        for i, (q, a) in enumerate(self._history, start=1):
            lines.append(f"[Lượt {i}]")
            lines.append(f"Học viên: {q}")
            # Giới hạn độ dài câu trả lời trong history để tránh prompt quá dài
            short_answer = a if len(a) <= 500 else a[:500] + "...(rút gọn)"
            lines.append(f"Gia sư: {short_answer}")
            lines.append("")   # Dòng trống giữa các lượt

        return "\n".join(lines).strip()

    def clear(self) -> None:
        """Xóa toàn bộ lịch sử hội thoại."""
        count = len(self._history)
        self._history.clear()
        logger.info("[Memory] Da xoa %d luot lich su.", count)

    def __len__(self) -> int:
        return len(self._history)

    def __bool__(self) -> bool:
        return len(self._history) > 0
