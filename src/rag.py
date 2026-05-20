"""
RAG.PY - Load FAISS vector database và tìm kiếm tài liệu liên quan (Singleton Pattern)
Cache FAISS DB vào bộ nhớ - chỉ load từ disk 1 lần duy nhất.
"""

import os
import logging
from langchain_community.vectorstores import FAISS
from src.model import get_embedding_model
from src.config import FAISS_DB_PATH, TOP_K_RETRIEVE, SCORE_THRESHOLD, DEBUG_MODE

# ==========================================
# LOGGER SETUP
# ==========================================
logger = logging.getLogger(__name__)

# ==========================================
# SINGLETON INSTANCE (module-level cache)
# ==========================================
_vector_db = None


def get_vector_db() -> FAISS:
    """
    Load kho dữ liệu FAISS từ local disk.
    Singleton: chỉ load từ disk 1 lần, cache vào RAM cho các lần sau.
    Raises: FileNotFoundError nếu thư mục data/ không tồn tại.
    """
    global _vector_db
    if _vector_db is None:
        if not os.path.exists(FAISS_DB_PATH):
            raise FileNotFoundError(
                f"[RAG] Khong tim thay thu muc FAISS: '{FAISS_DB_PATH}'\n"
                f"Hay dam bao thu muc 'data/' ton tai va co files index.faiss + index.pkl"
            )
        print("[RAG] Dang tai FAISS database lan dau...")
        _vector_db = FAISS.load_local(
            folder_path=FAISS_DB_PATH,
            embeddings=get_embedding_model(),
            allow_dangerous_deserialization=True  # Bat buoc de load FAISS local
        )
        print("[RAG] FAISS database san sang!")
    return _vector_db


def retrieve_context(query: str) -> str:
    """
    Tìm kiếm tài liệu liên quan, áp dụng score threshold để lọc chunk kém,
    rồi ghép thành 1 đoạn văn bản đưa vào prompt.
    """
    try:
        db = get_vector_db()

        docs_and_scores = db.similarity_search_with_score(query, k=TOP_K_RETRIEVE)

        if not docs_and_scores:
            logger.warning("[RAG] Khong co ket qua tra ve tu FAISS.")
            return "Không tìm thấy tài liệu liên quan."

        context_parts = []
        rejected = 0

        for i, (doc, score) in enumerate(docs_and_scores):
            # Log debug: chỉ hiện khi DEBUG_MODE=True, không in ra cho người dùng
            logger.debug("[RAG] Chunk %d | Score: %.4f | Noi dung: %s...", i + 1, score, doc.page_content[:60])

            if score > SCORE_THRESHOLD:
                # Chunk quá xa về mặt ngữ nghĩa → loại bỏ để tránh nhiễu prompt
                rejected += 1
                logger.debug("[RAG] Chunk %d bi loai (score %.4f > nguong %.4f)", i + 1, score, SCORE_THRESHOLD)
                continue

            context_parts.append(doc.page_content)

        if rejected > 0:
            logger.info("[RAG] Da loai %d/%d chunk do score vuot nguong %.2f.", rejected, len(docs_and_scores), SCORE_THRESHOLD)

        if not context_parts:
            logger.info("[RAG] Tat ca chunk bi loai do score qua thap lien quan.")
            return "Không tìm thấy tài liệu đủ liên quan trong cơ sở dữ liệu. Tôi sẽ trả lời dựa trên kiến thức của mình."

        logger.info("[RAG] Su dung %d/%d chunk.", len(context_parts), len(docs_and_scores))
        return "\n\n".join(context_parts)

    except FileNotFoundError as e:
        logger.error(str(e))
        return "Không tìm thấy tài liệu."
    except Exception as e:
        logger.error("[RAG] Loi khong xac dinh: %s", e)
        return "Không tìm thấy tài liệu."