"""
RAG.PY - Load Chroma vector database và tìm kiếm tài liệu liên quan

"""

import os
import re
import logging
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from src.model import get_embedding_model, get_llm, get_small_llm
from src.config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    TOP_K_RETRIEVE, SCORE_THRESHOLD,
    HYBRID_SEARCH,
    BM25_WEIGHT, CHROMA_WEIGHT,
    TOP_K_BM25, TOP_K_CHROMA,
)

logger = logging.getLogger(__name__)


# ==========================================
# QUERY CLASSIFIER — METADATA FILTER
# ==========================================
def _classify_query(query: str) -> dict | None:
    """
    Phân loại câu hỏi dựa trên hình thái ngôn ngữ → trả về Chroma metadata filter.

    Phân bổ thực tế trong DB (1194 chunks):
      grammar      : 787 chunks — ngữ pháp lý thuyết, collocation, phrasal verbs
      practice_test: 363 chunks — đề ETS, bài tập Part 5 có đáp án
      tips         :  44 chunks — mẹo làm bài Part 5

    Returns:
        dict  : Chroma where-filter, ví dụ {"category": "grammar"}
        None  : không filter, tìm toàn bộ database (fallback)
    """
    # ── NHÓM 1: Câu Part 5 trắc nghiệm có (A)(B)(C)(D)
    # Dấu hiệu: bắt đầu "Giải thích" VÀ có "(A)" trong câu hỏi
    if re.search(r'Gi\u1ea3i th\u00edch', query) and re.search(r'\(A\)', query):
        return {"category": {"$in": ["grammar", "practice_test"]}}

    # ── NHÓM 2: Collocation / cụm từ cố định
    # Dấu hiệu: câu bắt đầu "Cụm từ '...'"
    # Collocation nằm trong grammar (file grammar-200_Collocation...)
    # Vẫn include tips phòng trường hợp mẹo liên quan
    if re.match(r"^C\u1ee5m t\u1eeb\s+'", query):
        return {"category": {"$in": ["grammar", "tips"]}}

    # ── NHÓM 3: Lý thuyết ngữ pháp (công thức, cách dùng, từ nhận biết)
    # Dấu hiệu: các từ khoá chỉ câu hỏi lý thuyết
    if re.search(
        r'(C\u00f4ng th\u1ee9c|C\u00e1ch d\u00f9ng|t\u1eeb nh\u1eadn bi\u1ebft'
        r'|Nh\u1eefng t\u1eeb n\u00e0o|C\u00e1c tr\u1ea1ng t\u1eeb'
        r'|C\u00e1c t\u1eeb n\u00e0o|ngh\u0129a l\u00e0 g\u00ec|Kh\u00e1i ni\u1ec7m|l\u00e0 g\u00ec)',
        query, re.IGNORECASE
    ):
        return {"category": "grammar"}

    # ── FALLBACK: không nhận dạng được → tìm toàn bộ database
    logger.debug("[RAG][Classify] Khong nhan dang duoc query, tim toan bo DB.")
    return None

# ==========================================
# SINGLETON INSTANCES
# ==========================================
_vector_db: Chroma | None = None
_bm25_retriever: BM25Retriever | None = None


# ==========================================
# CHROMA SINGLETON
# ==========================================
def get_vector_db() -> Chroma:
    """
    Load kho dữ liệu Chroma từ local disk.
    Singleton: chỉ load từ disk 1 lần, cache vào RAM cho các lần sau.
    Raises: FileNotFoundError nếu thư mục data/ không tồn tại hoặc không có chroma.sqlite3.
    """
    global _vector_db
    if _vector_db is None:
        chroma_sqlite = os.path.join(CHROMA_DB_PATH, "chroma.sqlite3")
        if not os.path.exists(CHROMA_DB_PATH) or not os.path.exists(chroma_sqlite):
            raise FileNotFoundError(
                f"[RAG] Khong tim thay Chroma DB tai: '{CHROMA_DB_PATH}'\n"
                f"Hay dam bao thu muc 'data/' ton tai va co file 'chroma.sqlite3'"
            )
        print("[RAG] Dang tai Chroma database lan dau...")
        _vector_db = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=get_embedding_model(),
            collection_name=CHROMA_COLLECTION_NAME,
        )
        count = _vector_db._collection.count()
        print(f"[RAG] Chroma database san sang! ({count} documents)")
    return _vector_db


# ==========================================
# BM25 SINGLETON
# ==========================================
def _get_bm25_retriever() -> BM25Retriever:
    """
    Khởi tạo BM25Retriever từ toàn bộ documents trong Chroma.
    Singleton: tokenize nhiều docs mất thời gian, chỉ làm 1 lần.
    """
    global _bm25_retriever
    if _bm25_retriever is None:
        db = get_vector_db()
        # Lấy toàn bộ documents từ Chroma collection
        result = db._collection.get(include=["documents", "metadatas"])
        all_docs: list[Document] = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(result["documents"], result["metadatas"])
        ]
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
    - weight_i: trọng số của từng retriever (BM25_WEIGHT, CHROMA_WEIGHT).
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
def preprocess_query_for_retrieval(query: str) -> str:
    """
    Tiền xử lý câu hỏi để tối ưu hóa tìm kiếm bằng LLM (Keyword Extraction).
    """
    try:
        llm = get_small_llm()
        logger.info("[RAG] Đang dùng mô hình phụ (%s) để trích xuất từ khóa...", getattr(llm, 'model', 'unknown'))
        prompt = f"""Bạn là một hệ thống trích xuất từ khóa tìm kiếm. 
Nhiệm vụ: Trích xuất GIỮ NGUYÊN VĂN các danh từ, động từ quan trọng nhất và TOÀN BỘ từ tiếng Anh từ câu hỏi.
TUYỆT ĐỐI KHÔNG DỊCH SANG TIẾNG ANH (Ví dụ: không dịch "đảo ngữ" thành "inversion"). Giữ nguyên ngôn ngữ gốc.
Bỏ qua các từ vô nghĩa như: 'là gì', 'có nghĩa là', 'như thế nào', 'cấu trúc', 'công thức', 'phân biệt', 'khác nhau'.
CHỈ IN RA CÁC TỪ KHÓA, CÁCH NHAU BỞI KHOẢNG TRẮNG. TUYỆT ĐỐI KHÔNG GIẢI THÍCH HAY VIẾT THÊM BẤT CỨ GÌ KHÁC.
Câu hỏi: "{query}"
Từ khóa:"""
        
        # Invoke LLM
        response = llm.invoke(prompt)
        clean_query = response.content if hasattr(response, 'content') else str(response)
        
        # Clean up formatting (remove newlines, extra spaces, quotes)
        clean_query = clean_query.replace('"', '').replace("'", '').replace('\n', ' ')
        clean_query = re.sub(r"\s+", " ", clean_query).strip()
        
        if clean_query:
            return clean_query
            
    except Exception as e:
        logger.error("[RAG][Preprocess] LLM Extraction loi: %s. Fallback ve regex.", e)
        
    # Fallback if LLM fails
    clean_query = query.lower()
    vietnamese_stopwords = [
        r"\bcụm từ\b", r"\bnghĩa là gì\b", r"\bví dụ\b", r"\bnhư thế nào\b",
        r"\blà gì\b", r"\bđối với\b", r"\bcủa\b", r"\bnó\b", r"\bra sao\b",
        r"\bnhững từ nào\b", r"\bgiải thích câu hỏi\b", r"\bgiải thích\b", r"\bcâu hỏi\b",
        r"\bcó nghĩa là gì và ví dụ đi kèm ra sao\b",
        r"\bcó nghĩa là gì và ví dụ đi kèm như thế nào\b",
        r"\bcó ý nghĩa gì và ví dụ đi kèm là gì\b",
        r"\bcó nghĩa là gì\b", r"\bđi kèm ra sao\b", r"\bđi kèm như thế nào\b"
    ]
    for word in vietnamese_stopwords:
        clean_query = re.sub(word, "", clean_query)
    clean_query = re.sub(r"[?.,!]", "", clean_query)
    clean_query = re.sub(r"\s+", " ", clean_query).strip()
    return clean_query if clean_query else query


# ==========================================
# RETRIEVE CONTEXT
# ==========================================
def retrieve_context(raw_query: str, bm25_optimized_query: str = None) -> str:
    """
    Tìm kiếm tài liệu liên quan và ghép thành 1 đoạn văn bản đưa vào prompt.

    - raw_query: Dùng cho Chroma (giữ nguyên cấu trúc ____ và A/B/C/D)
    - bm25_optimized_query: Dùng cho BM25 (đã loại bỏ rác gây nhiễu)
    """
    if bm25_optimized_query is None:
        bm25_optimized_query = raw_query
        
    try:
        if HYBRID_SEARCH:
            return _retrieve_hybrid(raw_query, bm25_optimized_query)
        else:
            return _retrieve_chroma_only(raw_query, bm25_optimized_query)

    except FileNotFoundError as e:
        logger.error(str(e))
        return "Không tìm thấy tài liệu."
    except Exception as e:
        logger.error("[RAG] Loi khong xac dinh: %s", e, exc_info=True)
        return "Không tìm thấy tài liệu."


def _retrieve_hybrid(raw_query: str, bm25_optimized_query: str) -> str:
    """
    Hybrid Search: BM25 (keyword) + Chroma (semantic), merge bằng RRF.
    """
    # Bước 1: BM25 dùng câu hỏi đã tối ưu (đã xoá A/B/C/D) để nhặt từ khoá
    bm25_query = preprocess_query_for_retrieval(bm25_optimized_query)
    logger.info("[RAG][Hybrid] Query Semantic: '%s' | Query BM25: '%s' | BM25 Keywords: '%s'", raw_query, bm25_optimized_query, bm25_query)

    # 1. BM25 search
    bm25 = _get_bm25_retriever()
    bm25_docs = bm25.invoke(bm25_query)
    logger.info("[RAG][Hybrid] BM25 tra ve %d docs.", len(bm25_docs))

    # 2. Chroma semantic search (có metadata filter theo loại câu hỏi)
    db = get_vector_db()
    meta_filter = _classify_query(raw_query) # Cần dùng raw_query để nhận diện (A)
    try:
        if meta_filter:
            logger.info("[RAG][Hybrid] Ap dung metadata filter: %s", meta_filter)
        # Sử dụng nguyên gốc query cho Chroma để không làm mất ngữ nghĩa (đặc biệt là dấu ____)
        docs_and_scores = db.similarity_search_with_score(raw_query, k=TOP_K_CHROMA, filter=meta_filter)
    except Exception as filter_err:
        # ChromaDB 1.x bug: metadata filter crash khi DB tạo bằng version cũ
        # Fallback về tìm không filter để không bị block
        logger.warning(
            "[RAG][Hybrid] Metadata filter loi (%s) - fallback khong filter.",
            filter_err
        )
        docs_and_scores = db.similarity_search_with_score(raw_query, k=TOP_K_CHROMA)
        
    # Lọc chunk Chroma bằng SCORE_THRESHOLD
    chroma_docs = []
    rejected = 0
    for doc, score in docs_and_scores:
        if score <= SCORE_THRESHOLD:
            chroma_docs.append(doc)
        else:
            rejected += 1
            logger.debug("[RAG][Hybrid] Chunk Chroma bi loai vi score %.4f > %.4f", score, SCORE_THRESHOLD)
            
    if rejected > 0:
        logger.info("[RAG][Hybrid] Da loai %d/%d chunk Chroma kem chat luong.", rejected, len(docs_and_scores))

    logger.info("[RAG][Hybrid] Chroma tra ve %d docs.", len(chroma_docs))

    # 3. RRF merge
    merged = _reciprocal_rank_fusion(
        ranked_lists=[bm25_docs, chroma_docs],
        weights=[BM25_WEIGHT, CHROMA_WEIGHT],
    )
    logger.info("[RAG][Hybrid] Sau RRF: %d docs unique.", len(merged))

    # 4. Lấy TOP_K_RETRIEVE doc tốt nhất + lọc chunk rác quá ngắn
    candidates = merged[:TOP_K_RETRIEVE * 2]   # lấy dư rồi lọc
    final_docs = [
        d for d in candidates
        if len(d.page_content.strip()) >= 100  # Hạ từ 150 → 100: tránh lọc mất collocation ngắn
    ][:TOP_K_RETRIEVE]

    # Nếu lọc quá nhiều, fallback về merged gốc
    if not final_docs:
        final_docs = merged[:TOP_K_RETRIEVE]

    logger.info("[RAG][Hybrid] Final: %d docs sau khi loc.", len(final_docs))

    if not final_docs:
        return "Không tìm thấy tài liệu liên quan."

    return "\n\n".join(doc.page_content for doc in final_docs)


def _retrieve_chroma_only(raw_query: str, bm25_optimized_query: str) -> str:
    """
    Chroma-only Search với score threshold để lọc chunk kém chất lượng.
    Dùng khi HYBRID_SEARCH=False.
    Lưu ý: Chroma trả về (doc, distance) - distance càng thấp càng tốt (cosine distance).
    """
    db = get_vector_db()
    meta_filter = _classify_query(raw_query)
    
    try:
        if meta_filter:
            logger.info("[RAG][Chroma] Ap dung metadata filter: %s", meta_filter)
        # Sửa lỗi logic cũ: Chroma phải dùng raw_query chứa đầy đủ ngữ nghĩa, KHÔNG dùng clean_query (chỉ toàn từ khoá rời rạc)
        docs_and_scores = db.similarity_search_with_score(raw_query, k=TOP_K_RETRIEVE, filter=meta_filter)
    except Exception as filter_err:
        logger.warning(
            "[RAG][Chroma] Metadata filter loi (%s) - fallback khong filter.",
            filter_err
        )
        docs_and_scores = db.similarity_search_with_score(raw_query, k=TOP_K_RETRIEVE)

    if not docs_and_scores:
        logger.warning("[RAG][Chroma] Khong co ket qua tra ve.")
        return "Không tìm thấy tài liệu liên quan."

    context_parts = []
    rejected = 0

    for i, (doc, score) in enumerate(docs_and_scores):
        logger.debug("[RAG][Chroma] Chunk %d | Distance: %.4f | %s...", i + 1, score, doc.page_content[:60])

        if score > SCORE_THRESHOLD:
            rejected += 1
            logger.debug("[RAG][Chroma] Chunk %d bi loai (%.4f > %.4f)", i + 1, score, SCORE_THRESHOLD)
            continue

        context_parts.append(doc.page_content)

    if rejected > 0:
        logger.info("[RAG][Chroma] Da loai %d/%d chunk (distance > %.2f).", rejected, len(docs_and_scores), SCORE_THRESHOLD)

    if not context_parts:
        return "Không tìm thấy tài liệu đủ liên quan. Tôi sẽ trả lời dựa trên kiến thức của mình."

    logger.info("[RAG][Chroma] Su dung %d/%d chunk.", len(context_parts), len(docs_and_scores))
    return "\n\n".join(context_parts)