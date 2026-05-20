"""
MODEL.PY - Khởi tạo Embedding Model và LLM (Singleton Pattern)
Chỉ load một lần duy nhất, tái sử dụng cho mọi câu hỏi tiếp theo.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
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


def get_llm() -> OllamaLLM:
    """
    Trả về LLM (qwen2.5:7b) chạy offline qua Ollama.
    Singleton: chỉ khởi tạo 1 lần, tái dùng cho mọi lần gọi sau.
    Raises: ConnectionError nếu Ollama chưa được khởi động.
    """
    global _llm
    if _llm is None:
        _llm = OllamaLLM(
            model=OLLAMA_MODEL_NAME,
            temperature=TEMPERATURE,
            top_p=TOP_P
        )
    return _llm