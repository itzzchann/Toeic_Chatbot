"""
RAG.PY - Load FAISS vector database và tìm kiếm tài liệu liên quan (Singleton Pattern)
Cache FAISS DB vào bộ nhớ - chỉ load từ disk 1 lần duy nhất.
"""

import os
from langchain_community.vectorstores import FAISS
from src.model import get_embedding_model
from src.config import FAISS_DB_PATH, TOP_K_RETRIEVE

# ==========================================
# SINGLETON INSTANCE (module-level cache)
# ==========================================
_vector_db = None


def get_vector_db() -> FAISS:
    """
    Load kho dữ liệu FAISS từ local disk.
    Singleton: chỉ load từ disk 1 lần, cache vào RAM cho các lần sau.
    Raises: FileNotFoundError nếu thư mục Toeic_db không tồn tại.
    """
    global _vector_db
    if _vector_db is None:
        if not os.path.exists(FAISS_DB_PATH):
            raise FileNotFoundError(
                f"[RAG] Khong tim thay thu muc FAISS: '{FAISS_DB_PATH}'\n"
                f"Hay dam bao thu muc 'Toeic_db/' ton tai va co files index.faiss + index.pkl"
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
    """Tìm kiếm tài liệu liên quan và ghép thành 1 đoạn văn bản."""
    try:
        db = get_vector_db()
        
        # Dùng similarity_search_with_score thay vì similarity_search
        docs_and_scores = db.similarity_search_with_score(query, k=TOP_K_RETRIEVE)
        
        if not docs_and_scores:
            return "Khong tim thay tai lieu lien quan."
            
        context_parts = []
        for i, (doc, score) in enumerate(docs_and_scores):
            # In ra terminal để dev theo dõi (càng gần 0 hoặc càng cao tuỳ model, thường L2 distance thì càng thấp càng tốt)
            print(f"[DEBUG RAG] Chunk {i+1} | Score: {score:.4f}") 
            context_parts.append(doc.page_content)
            
        return "\n\n".join(context_parts)
        
    except FileNotFoundError as e:
        print(e)
        return "Khong tim thay tai lieu."
    except Exception as e:
        print(f"[RAG] Loi khong xac dinh: {e}")
        return "Khong tim thay tai lieu."