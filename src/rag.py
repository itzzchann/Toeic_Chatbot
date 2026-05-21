"""
RAG.PY - Load FAISS vector database và tìm kiếm tài liệu liên quan

"""

import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from src.model import get_embedding_model
from src.config import (
    FAISS_DB_PATH,
    TOP_K_RETRIEVE, SCORE_THRESHOLD,
    HYBRID_SEARCH,
    BM25_WEIGHT, FAISS_WEIGHT,
    TOP_K_BM25, TOP_K_FAISS,
)

logger = logging.getLogger(__name__)

# ==========================================
# SINGLETON INSTANCES
# ==========================================
_vector_db: FAISS | None = None
_bm25_retriever: BM25Retriever | None = None


# ==========================================
# FAISS SINGLETON
# ==========================================
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
            allow_dangerous_deserialization=True
        )
        print(f"[RAG] FAISS database san sang! ({len(_vector_db.docstore._dict)} documents)")
    return _vector_db


# ==========================================
# BM25 SINGLETON
# ==========================================
def _get_bm25_retriever() -> BM25Retriever:
    """
    Khởi tạo BM25Retriever từ toàn bộ documents trong FAISS docstore.
    Singleton: tokenize 2000+ docs mất ~2-5s, chỉ làm 1 lần.
    """
    global _bm25_retriever
    if _bm25_retriever is None:
        db = get_vector_db()
        all_docs: list[Document] = list(db.docstore._dict.values())
        logger.info("[RAG] Khoi tao BM25 voi %d documents...", len(all_docs))
        print(f"[RAG] Dang xay BM25 index ({len(all_docs)} docs)...")
        _bm25_retriever = BM25Retriever.from_documents(all_docs, k=TOP_K_BM25)
        print("[RAG] BM25 index san sang!")
    return _bm25_retriever


# ==========================================
# RECIPROCAL RANK FUSION (RRF)
# ==========================================
def _reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    weights: list[float],
    k: int = 60,
) -> list[Document]:
    """
    Merge nhiều ranked list bằng Reciprocal Rank Fusion (RRF).

    Công thức: score(doc) = Σ weight_i / (k + rank_i)
    - k=60: hằng số chuẩn của RRF, giảm ảnh hưởng của rank quá cao.
    - weight_i: trọng số của từng retriever (BM25_WEIGHT, FAISS_WEIGHT).
    - Deduplicate bằng page_content hash.

    Trả về: list[Document] sắp xếp theo RRF score giảm dần.
    """
    scores: dict[str, float] = {}       # content_hash → RRF score
    doc_map: dict[str, Document] = {}   # content_hash → Document

    for docs, weight in zip(ranked_lists, weights):
        for rank, doc in enumerate(docs, start=1):
            # Dùng content làm key dedup (tránh trùng lặp giữa 2 retriever)
            key = hash(doc.page_content)
            doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)

    # Sắp xếp giảm dần theo RRF score
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)

    for i, key in enumerate(sorted_keys):
        logger.debug(
            "[RAG][RRF] Rank %d | RRF=%.4f | preview: %s...",
            i + 1, scores[key], doc_map[key].page_content[:60]
        )

    return [doc_map[key] for key in sorted_keys]


# ==========================================
# RETRIEVE CONTEXT
# ==========================================
def retrieve_context(query: str) -> str:
    """
    Tìm kiếm tài liệu liên quan và ghép thành 1 đoạn văn bản đưa vào prompt.

    - HYBRID_SEARCH=True  : BM25 + FAISS, merge bằng RRF.
    - HYBRID_SEARCH=False : FAISS-only với score threshold lọc chunk kém.
    """
    try:
        if HYBRID_SEARCH:
            return _retrieve_hybrid(query)
        else:
            return _retrieve_faiss_only(query)

    except FileNotFoundError as e:
        logger.error(str(e))
        return "Không tìm thấy tài liệu."
    except Exception as e:
        logger.error("[RAG] Loi khong xac dinh: %s", e, exc_info=True)
        return "Không tìm thấy tài liệu."


def _retrieve_hybrid(query: str) -> str:
    """
    Hybrid Search: BM25 (keyword) + FAISS (semantic), merge bằng RRF.

    Pipeline:
      1. BM25 retriever → top-K_BM25 docs (theo TF-IDF/BM25 score)
      2. FAISS retriever → top-K_FAISS docs (theo L2 distance embedding)
      3. RRF merge + deduplicate → final ranking
      4. Lấy TOP_K_RETRIEVE docs đầu tiên đưa vào prompt
    """
    # 1. BM25 search
    bm25 = _get_bm25_retriever()
    bm25_docs = bm25.invoke(query)
    logger.info("[RAG][Hybrid] BM25 tra ve %d docs.", len(bm25_docs))

    # 2. FAISS search
    db = get_vector_db()
    faiss_docs = db.similarity_search(query, k=TOP_K_FAISS)
    logger.info("[RAG][Hybrid] FAISS tra ve %d docs.", len(faiss_docs))

    # 3. RRF merge
    merged = _reciprocal_rank_fusion(
        ranked_lists=[bm25_docs, faiss_docs],
        weights=[BM25_WEIGHT, FAISS_WEIGHT],
    )
    logger.info("[RAG][Hybrid] Sau RRF: %d docs unique.", len(merged))

    # 4. Lấy TOP_K_RETRIEVE doc tốt nhất
    final_docs = merged[:TOP_K_RETRIEVE]

    if not final_docs:
        return "Không tìm thấy tài liệu liên quan."

    return "\n\n".join(doc.page_content for doc in final_docs)


def _retrieve_faiss_only(query: str) -> str:
    """
    FAISS-only Search với score threshold để lọc chunk kém chất lượng.
    Dùng khi HYBRID_SEARCH=False.
    """
    db = get_vector_db()
    docs_and_scores = db.similarity_search_with_score(query, k=TOP_K_RETRIEVE)

    if not docs_and_scores:
        logger.warning("[RAG][FAISS] Khong co ket qua tra ve.")
        return "Không tìm thấy tài liệu liên quan."

    context_parts = []
    rejected = 0

    for i, (doc, score) in enumerate(docs_and_scores):
        logger.debug("[RAG][FAISS] Chunk %d | Score: %.4f | %s...", i + 1, score, doc.page_content[:60])

        if score > SCORE_THRESHOLD:
            rejected += 1
            logger.debug("[RAG][FAISS] Chunk %d bi loai (%.4f > %.4f)", i + 1, score, SCORE_THRESHOLD)
            continue

        context_parts.append(doc.page_content)

    if rejected > 0:
        logger.info("[RAG][FAISS] Da loai %d/%d chunk (score > %.2f).", rejected, len(docs_and_scores), SCORE_THRESHOLD)

    if not context_parts:
        return "Không tìm thấy tài liệu đủ liên quan. Tôi sẽ trả lời dựa trên kiến thức của mình."

    logger.info("[RAG][FAISS] Su dung %d/%d chunk.", len(context_parts), len(docs_and_scores))
    return "\n\n".join(context_parts)