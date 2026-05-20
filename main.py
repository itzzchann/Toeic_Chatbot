"""
MAIN.PY - Diem vao chinh de chay chatbot TOEIC
Chay lenh: python main.py

Cải tiến:
- Logging system: log ra file toeic_chatbot.log, không làm bẩn console người dùng.
- Streaming output: text hiện dần từng token khi STREAM_OUTPUT=True.
"""

import sys
import logging
from src.ask import get_bot_response, stream_bot_response
from src.rag import get_vector_db
from src.config import DEBUG_MODE, STREAM_OUTPUT


# ==========================================
# LOGGING SETUP (gọi 1 lần khi khởi động)
# ==========================================
def setup_logging():
    """
    Cấu hình logging toàn hệ thống.
    - DEBUG_MODE=True  → ghi DEBUG log ra file + WARNING ra console
    - DEBUG_MODE=False → chỉ ghi WARNING+ ra file, console sạch
    """
    log_level = logging.DEBUG if DEBUG_MODE else logging.WARNING

    # Handler ghi vào file (luôn bật)
    file_handler = logging.FileHandler("toeic_chatbot.log", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)

    # Tắt log ồn ào từ các thư viện bên thứ 3 (transformers, httpx, faiss...)
    for noisy_lib in ("transformers", "httpx", "httpcore", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy_lib).setLevel(logging.ERROR)


def startup_check():
    """Kiem tra FAISS va Ollama san sang truoc khi bat dau chat."""
    print("Dang khoi dong he thong...")
    try:
        get_vector_db()
        print("He thong san sang!\n")
        return True
    except FileNotFoundError as e:
        print(f"\n[LOI] {e}")
        print("Vui long kiem tra thu muc data/ va thu lai.")
        return False
    except Exception as e:
        print(f"\n[LOI] Khoi dong that bai: {e}")
        print("Hay dam bao Ollama dang chay: ollama serve")
        return False


def main():
    setup_logging()

    print("=" * 50)
    print("   TOEIC MASTER - Gia su TOEIC AI   ")
    print("=" * 50)
    print("Go 'exit' hoac 'thoat' de thoat chuong trinh.")
    print()

    if not startup_check():
        sys.exit(1)

    while True:
        try:
            user_query = input("Ban muon hoi gi ve TOEIC: ").strip()

            if not user_query:
                continue

            if user_query.lower() in ('exit', 'quit', 'thoat'):
                print("\nTam biet! Chuc ban hoc tot va dat diem TOEIC cao!")
                break

            print("\n" + "=" * 50)
            print("TOEIC MASTER TRA LOI:")
            print("=" * 50)

            if STREAM_OUTPUT:
                # Streaming: in từng token ngay khi LLM sinh ra
                for token in stream_bot_response(user_query):
                    print(token, end="", flush=True)
                print()  # Xuống dòng sau khi stream xong
            else:
                # Blocking: chờ toàn bộ rồi in một lần
                answer = get_bot_response(user_query)
                print(answer)

            print("=" * 50 + "\n")

        except ConnectionError:
            print("\n[LOI] Khong ket noi duoc Ollama.")
            print("Hay chay: ollama serve   (trong cua so khac)\n")
        except KeyboardInterrupt:
            print("\n\nTam biet! Chuc ban hoc tot!")
            break
        except Exception as e:
            print(f"\n[LOI] Co loi xay ra: {e}")
            print("Vui long thu lai.\n")


if __name__ == "__main__":
    main()
