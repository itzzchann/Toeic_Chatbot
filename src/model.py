"""
MODEL.PY - Khởi tạo Embedding Model và LLM (Singleton Pattern)
Chỉ load một lần duy nhất, tái sử dụng cho mọi câu hỏi tiếp theo.
"""

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL_NAME, OLLAMA_MODEL_NAME, TEMPERATURE, TOP_P

# ==========================================
# SINGLETON INSTANCES (module-level cache)
# ==========================================
_embedding_model = None
_llm = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Trả về embedding model đa ngôn ngữ.
    Singleton: chỉ khởi tạo 1 lần, tái dùng cho mọi lần gọi sau.
    """
    global _embedding_model
    if _embedding_model is None:
        print("[Model] Dang tai embedding model lan dau...")
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True}
        )
        print("[Model] Embedding model san sang!")
    return _embedding_model


def get_llm() -> ChatOllama:
    """
    Trả về LLM chạy offline qua Ollama.
    Singleton: chỉ khởi tạo 1 lần, tái dùng cho mọi lần gọi sau.
    Raises: ConnectionError nếu Ollama chưa được khởi động.
    """
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=OLLAMA_MODEL_NAME,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            num_ctx=8192,
            num_predict=2048
        )
    return _llm