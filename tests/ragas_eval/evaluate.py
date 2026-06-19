"""
EVALUATE.PY - Script chính chạy RAGAS evaluation cho TOEIC Chatbot
========================================================================
Cách chạy (từ thư mục gốc TTNT/):
    python -m tests.ragas_eval.evaluate            # Tự động dùng cache nếu có
    python -m tests.ragas_eval.evaluate --rebuild  # Bắt buộc build lại từ đầu

Cache:
    Dataset (câu hỏi + câu trả lời của chatbot + context) được lưu vào:
        tests/ragas_eval/dataset_cache.json
    Lần đầu chạy: build dataset (gọi chatbot 119 lần) → lưu cache → eval
    Lần sau     : load cache (2 giây) → eval ngay (bỏ qua bước gọi chatbot)

Judge LLM: Local Ollama (qwen2.5) — chạy offline hoàn toàn miễn phí

Metrics dùng:
    - faithfulness      : Câu trả lời có bịa thêm thông tin ngoài context không?
    - answer_relevancy  : Câu trả lời có đúng trọng tâm câu hỏi không?
    - context_recall    : Context trích xuất có đầy đủ để trả lời không?
    - context_precision : Context trích xuất có bị dư thừa nhiễu không?
========================================================================"""

import sys
import time
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import logging
import os
import argparse
from datetime import datetime
from pathlib import Path

# RAGAS vẫn cần biến này để không bị lỗi import — không dùng OpenAI thật
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
    from ragas.llms import llm_factory
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from openai import OpenAI as OpenAIClient
except ImportError as e:
    print(f"\n❌ Thiếu thư viện hoặc phiên bản không khớp: {e}")
    print("   Hãy chạy: pip install ragas datasets openai sacrebleu rouge-score\n")
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

OUTPUT_DIR   = PROJECT_ROOT / "tests" / "ragas_eval" / "reports"
CACHE_FILE   = PROJECT_ROOT / "tests" / "ragas_eval" / "dataset_cache.json"
# Dùng qwen2.5 chạy local trên Ollama làm Judge LLM
EVAL_LLM_MODEL = "qwen2.5:7b"

# Fix BadRequestError: RAGAS answer_relevancy mặc định gửi n=3 → Một số model local không hỗ trợ.
# Đặt strictness=1 để chỉ gửi n=1 mỗi request.
answer_relevancy.strictness = 1


# ══════════════════════════════════════════════════════════════
# CACHE: LƯU / TẢI DATASET
# ══════════════════════════════════════════════════════════════
def save_cache(samples_raw: list[dict]) -> None:
    """Lưu danh sách dict (user_input, response, retrieved_contexts, reference) ra JSON."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(samples_raw, f, ensure_ascii=False, indent=2)
    print(f"  💾 Cache đã lưu → {CACHE_FILE}")


def load_cache() -> list[dict] | None:
    """
    Tải cache từ JSON nếu tồn tại.
    Trả về list[dict] hoặc None nếu chưa có cache.
    """
    if not CACHE_FILE.exists():
        return None
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  📂 Đã tìm thấy cache ({len(data)} mẫu) → {CACHE_FILE}")
        return data
    except Exception as e:
        print(f"  ⚠️  Cache bị lỗi, sẽ build lại: {e}")
        return None


def raw_to_dataset(samples_raw: list[dict]) -> EvaluationDataset:
    """Chuyển list[dict] từ cache thành EvaluationDataset của RAGAS."""
    samples = [
        SingleTurnSample(
            user_input         = item["user_input"],
            response           = item["response"],
            retrieved_contexts = item["retrieved_contexts"],
            reference          = item["reference"],
        )
        for item in samples_raw
    ]
    return EvaluationDataset(samples=samples)


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
# BUILD DATASET (gọi chatbot thật, tốn thời gian)
# ══════════════════════════════════════════════════════════════
def build_and_cache_dataset(golden_data: list[dict]) -> EvaluationDataset:
    """
    Gọi chatbot cho toàn bộ golden_data, lưu kết quả vào cache JSON,
    sau đó trả về EvaluationDataset.
    """
    samples_raw = []
    total = len(golden_data)

    for i, item in enumerate(golden_data, start=1):
        question  = item["user_input"]
        reference = item.get("reference") or item.get("references")
        print(f"  [{i:2d}/{total}] ⚙️  {question[:70]}...")
        try:
            result = run_pipeline_for_sample(question)
            samples_raw.append({
                "user_input":         result["user_input"],
                "response":           result["response"],
                "retrieved_contexts": result["retrieved_contexts"],
                "reference":          reference,
            })
            print(f"         ✅ {len(result['retrieved_contexts'])} chunks | "
                  f"{len(result['response'])} chars")
        except Exception as e:
            print(f"         ⚠️  Bỏ qua: {e}")

    # Lưu cache ngay sau khi build xong
    save_cache(samples_raw)
    return raw_to_dataset(samples_raw)


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
    global CACHE_FILE
    
    # ── Parse tham số dòng lệnh ──
    parser = argparse.ArgumentParser(description="RAGAS Evaluation cho TOEIC Chatbot")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Bỏ qua cache, build lại dataset từ đầu (gọi chatbot thật)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Đường dẫn đến file dataset JSON thay vì dùng dataset mặc định"
    )
    args = parser.parse_args()

    print("\n" + "╔" + "═" * 60 + "╗")
    print(f"║  🎓  RAGAS EVALUATION — AI TOEIC CHATBOT" + " " * 18 + "║")
    print(f"║  Judge LLM: Ollama {EVAL_LLM_MODEL:<35} ║")
    print("╚" + "═" * 60 + "╝")

    # Xác định dataset cần chạy
    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"\n❌ Lỗi: Không tìm thấy file dataset {args.dataset}")
            sys.exit(1)
        with open(dataset_path, "r", encoding="utf-8") as f:
            golden_data = json.load(f)
        # Sửa tên cache để không đè lên cache mặc định
        CACHE_FILE = CACHE_FILE.parent / f"dataset_cache_{dataset_path.stem}.json"
        print(f"\n📂 Dùng dataset tùy chỉnh: {args.dataset} ({len(golden_data)} mẫu)")
    else:
        from tests.ragas_eval.dataset import TOEIC_GOLDEN_DATASET
        golden_data = TOEIC_GOLDEN_DATASET

    # ── Bước 1: Load cache hoặc Build dataset ──
    cached = None if args.rebuild else load_cache()

    if cached is not None:
        # ✅ Có cache → bỏ qua bước gọi chatbot
        print(f"\n⚡ Dùng cache có sẵn — bỏ qua bước gọi chatbot.")
        print(f"   (Dùng --rebuild để build lại từ đầu)\n")
        eval_dataset = raw_to_dataset(cached)
    else:
        # 🔨 Chưa có cache hoặc --rebuild → build từ đầu
        if args.rebuild:
            print("\n🔨 --rebuild: Xóa cache cũ, build lại toàn bộ dataset...")
        else:
            print(f"\n📋 Chưa có cache. Build dataset ({len(golden_data)} câu hỏi)...")

        # Cần Ollama + RAG để build
        warmup_system()

        t0 = time.time()
        eval_dataset = build_and_cache_dataset(golden_data)
        print(f"\n✅ Dataset sẵn sàng: {len(eval_dataset.samples)} mẫu ({time.time()-t0:.1f}s)\n")

    if not eval_dataset.samples:
        print("❌ Không có mẫu nào. Kiểm tra pipeline hoặc xóa cache bằng --rebuild.")
        sys.exit(1)

    # ── Bước 2: Khởi tạo Ragas Judge & Embeddings Wrapper ──
    print(f"⚙️  Khởi tạo Ragas Judge LLM ({EVAL_LLM_MODEL}) & Embeddings...")
    try:
        # Dùng OpenAI-compatible client trỏ vào Ollama local endpoint
        ollama_client = OpenAIClient(
            api_key="ollama",
            base_url="http://localhost:11434/v1",
        )
        judge_llm = llm_factory(EVAL_LLM_MODEL, client=ollama_client)

        # Embedding: vẫn dùng multilingual-e5 local (chạy offline, không tốn quota)
        evaluator_embeddings = LangchainEmbeddingsWrapper(get_embedding_model())
    except Exception as e:
        print(f"\n❌ Lỗi khởi tạo Ollama: {e}")
        print("   Hãy kiểm tra Ollama service có đang chạy tại http://localhost:11434 không.")
        sys.exit(1)

    # ── Bước 3: RAGAS evaluate ──
    print("🚀 Đang chạy RAGAS evaluation...")
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

    eval_start = time.time()
    try:
        result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=evaluator_embeddings,
            batch_size=2,   # Giảm xuống 2 để tránh burst rate limit trên free tier
        )
    except Exception as e:
        print(f"\n❌ RAGAS lỗi: {e}")
        print("   Gợi ý: Kiểm tra kết nối với Ollama hoặc các thông số cấu hình:")
        print("           pip install ragas datasets openai sacrebleu rouge-score")
        sys.exit(1)

    elapsed = time.time() - eval_start

    # ── Bước 4: In + lưu ──
    csv_path = save_report(result, eval_dataset)
    print_results(result, elapsed, csv_path)


if __name__ == "__main__":
    main()
