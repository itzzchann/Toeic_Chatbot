"""
EVALUATE.PY - Script chính chạy RAGAS evaluation cho TOEIC Chatbot
========================================================================
Cách chạy (từ thư mục gốc TTNT/):
    python -m tests.ragas_eval.evaluate

Metrics dùng:
    - faithfulness      : Câu trả lời có bịa thêm thông tin ngoài context không?
    - answer_relevancy  : Câu trả lời có đúng trọng tâm câu hỏi không?
    - context_recall    : Context trích xuất có đầy đủ để trả lời không?
    - context_precision : Context trích xuất có bị dư thừa nhiễu không?
========================================================================
"""

import sys
import time
import logging
import os
from datetime import datetime
from pathlib import Path

# Thiết lập OPENAI_API_KEY ảo để tránh RAGAS yêu cầu API Key
os.environ["OPENAI_API_KEY"] = "sk-dummy"

# ── Thêm project root vào sys.path để import src/ ──
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Tắt log ồn ào ──
for lib in ("transformers", "httpx", "httpcore", "urllib3",
            "sentence_transformers", "faiss", "ragas"):
    logging.getLogger(lib).setLevel(logging.ERROR)

# ══════════════════════════════════════════════════════════════
# IMPORT RAGAS & LANGCHAIN WRAPPERS
# ══════════════════════════════════════════════════════════════
try:
    from ragas import evaluate, EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_ollama import OllamaLLM
except ImportError as e:
    print(f"\n❌ Thiếu thư viện hoặc phiên bản không khớp: {e}")
    print("   Hãy chạy: pip install ragas datasets langchain-ollama sacrebleu rouge-score\n")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# IMPORT HỆ THỐNG HIỆN TẠI
# ══════════════════════════════════════════════════════════════
try:
    import torch
    try:
        torch.classes.__path__ = []
    except Exception:
        pass

    from src.rag import retrieve_context, get_vector_db, _get_bm25_retriever
    from src.ask import get_bot_response
    from src.config import HYBRID_SEARCH
    from src.model import get_embedding_model
except ImportError as e:
    print(f"\n❌ Không import được src/: {e}")
    print("   Hãy chạy từ thư mục gốc: python -m tests.ragas_eval.evaluate\n")
    sys.exit(1)

from tests.ragas_eval.dataset import TOEIC_GOLDEN_DATASET

OUTPUT_DIR = PROJECT_ROOT / "tests" / "ragas_eval" / "reports"
EVAL_LLM_MODEL = "llama3" # Model nhẹ làm judge (Có thể đổi thành qwen2.5:3b tùy ý)

# ══════════════════════════════════════════════════════════════
# WARM-UP HỆ THỐNG RAG
# ══════════════════════════════════════════════════════════════
def warmup_system():
    print("\n⏳ Đang khởi động hệ thống RAG...")
    try:
        get_vector_db()
        if HYBRID_SEARCH:
            _get_bm25_retriever()
        print("✅ Hệ thống RAG sẵn sàng!\n")
    except FileNotFoundError as e:
        print(f"❌ {e}\n")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════
# CHẠY PIPELINE VỚI TỪNG CÂU HỎI
# ══════════════════════════════════════════════════════════════
def run_pipeline_for_sample(question: str) -> dict:
    """Gọi hệ thống thật: lấy context + answer cho 1 câu hỏi."""
    raw_context = retrieve_context(question)
    contexts    = [c.strip() for c in raw_context.split("\n\n") if c.strip()]
    answer      = get_bot_response(question)
    return {
        "user_input":         question,
        "response":           answer,
        "retrieved_contexts": contexts,
    }


# ══════════════════════════════════════════════════════════════
# BUILD EVALUATION DATASET
# ══════════════════════════════════════════════════════════════
def build_ragas_dataset(golden_data: list[dict]) -> EvaluationDataset:
    samples = []
    total   = len(golden_data)

    for i, item in enumerate(golden_data, start=1):
        question  = item["user_input"]
        reference = item["reference"]
        print(f"  [{i:2d}/{total}] ⚙️  {question[:70]}...")
        try:
            result = run_pipeline_for_sample(question)
            sample = SingleTurnSample(
                user_input         = result["user_input"],
                response           = result["response"],
                retrieved_contexts = result["retrieved_contexts"],
                reference          = reference,
            )
            samples.append(sample)
            print(f"         ✅ {len(result['retrieved_contexts'])} chunks | "
                  f"{len(result['response'])} chars")
        except Exception as e:
            print(f"         ⚠️  Bỏ qua: {e}")

    return EvaluationDataset(samples=samples)


# ══════════════════════════════════════════════════════════════
# IN KẾT QUẢ
# ══════════════════════════════════════════════════════════════
def print_results(result, elapsed: float, csv_path):
    WIDTH = 62
    print("\n" + "=" * WIDTH)
    print("📊  RAGAS EVALUATION RESULTS — TOEIC CHATBOT")
    print("=" * WIDTH)

    try:
        df = result.to_pandas()
    except Exception as e:
        print(f"  ❌ Không đọc được kết quả: {e}")
        return

    METRIC_LABELS = {
        "faithfulness":      "Faithfulness      (Generator)",
        "answer_relevancy":  "Answer Relevancy  (Generator)",
        "context_recall":    "Context Recall    (Retriever)",
        "context_precision": "Context Precision (Retriever)",
    }

    scores = {}
    for col, label in METRIC_LABELS.items():
        if col in df.columns:
            scores[label] = float(df[col].mean())
        else:
            scores[label] = None

    for label, score in scores.items():
        if score is not None:
            bar   = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            grade = "🟢" if score >= 0.75 else ("🟡" if score >= 0.5 else "🔴")
            print(f"  {grade} {label}: {score:.4f}  [{bar}]")
        else:
            print(f"  ⚪ {label}: N/A")

    print("-" * WIDTH)
    print(f"  ⏱️  Thời gian chạy : {elapsed:.1f}s")
    print(f"  📝 Số câu đã test : {len(df)}")
    if csv_path:
        print(f"  💾 Report đã lưu : {csv_path}")
    print("=" * WIDTH)


# ══════════════════════════════════════════════════════════════
# XUẤT CSV
# ══════════════════════════════════════════════════════════════
def save_report(result, dataset: EvaluationDataset):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = OUTPUT_DIR / f"ragas_report_{timestamp}.csv"
    try:
        df = result.to_pandas()
        if "user_input" not in df.columns:
            df.insert(0, "user_input", [s.user_input for s in dataset.samples][:len(df)])
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return csv_path
    except Exception as e:
        print(f"  ⚠️  Không lưu được CSV: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("\n" + "╔" + "═" * 60 + "╗")
    print(f"║  🎓  RAGAS EVALUATION — AI TOEIC CHATBOT" + " " * 18 + "║")
    print(f"║  Judge LLM: Ollama {EVAL_LLM_MODEL:<35} ║")
    print("╚" + "═" * 60 + "╝")

    # Bước 1: Warm-up RAG
    warmup_system()

    # Bước 2: Build dataset (gọi pipeline thật)
    print(f"📋 Build dataset ({len(TOEIC_GOLDEN_DATASET)} câu hỏi)...")
    t0           = time.time()
    eval_dataset = build_ragas_dataset(TOEIC_GOLDEN_DATASET)
    print(f"\n✅ Dataset sẵn sàng: {len(eval_dataset.samples)} mẫu ({time.time()-t0:.1f}s)\n")

    if not eval_dataset.samples:
        print("❌ Không có mẫu nào. Kiểm tra pipeline.")
        sys.exit(1)

    # Bước 3: Khởi tạo Ragas Judge & Embeddings Wrapper
    print(f"⚙️  Khởi tạo Ragas Judge LLM ({EVAL_LLM_MODEL}) & Embeddings...")
    try:
        # Khởi tạo model làm judge
        judge_llm = LangchainLLMWrapper(OllamaLLM(model=EVAL_LLM_MODEL, temperature=0.0))
        
        # Bọc embedding model hiện tại của hệ thống (multilingual-e5)
        evaluator_embeddings = LangchainEmbeddingsWrapper(get_embedding_model())
    except Exception as e:
        print(f"\n❌ Lỗi khởi tạo Ollama: {e}")
        print("   Hãy chắc chắn rằng Ollama đang chạy (`ollama serve`)!")
        sys.exit(1)

    # Bước 4: RAGAS evaluate
    print("🚀 Đang chạy RAGAS evaluation...")
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

    eval_start = time.time()
    try:
        result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=evaluator_embeddings
        )
    except Exception as e:
        print(f"\n❌ RAGAS lỗi: {e}")
        print("   Gợi ý: Hãy kiểm tra xem bạn đã chạy lệnh: pip install sacrebleu rouge-score")
        print(f"          và đã tải model `{EVAL_LLM_MODEL}` bằng cách chạy: ollama pull {EVAL_LLM_MODEL}")
        sys.exit(1)

    elapsed = time.time() - eval_start

    # Bước 5: In + lưu
    csv_path = save_report(result, eval_dataset)
    print_results(result, elapsed, csv_path)


if __name__ == "__main__":
    main()
