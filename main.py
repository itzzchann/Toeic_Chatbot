"""
MAIN.PY - Diem vao chinh de chay chatbot TOEIC
Chay lenh: python main.py
"""

import sys
from src.ask import get_bot_response
from src.rag import get_vector_db


def startup_check():
    """Kiem tra FAISS va Ollama san sang truoc khi bat dau chat."""
    print("Dang khoi dong he thong...")
    try:
        # Warm-up: load FAISS + embedding model ngay tu dau (Singleton)
        get_vector_db()
        print("He thong san sang!\n")
        return True
    except FileNotFoundError as e:
        print(f"\n[LOI] {e}")
        print("Vui long kiem tra thu muc Toeic_db/ va thu lai.")
        return False
    except Exception as e:
        print(f"\n[LOI] Khoi dong that bai: {e}")
        print("Hay dam bao Ollama dang chay: ollama serve")
        return False


def main():
    print("=" * 50)
    print("   TOEIC MASTER - Gia su TOEIC AI   ")
    print("=" * 50)
    print("Go 'exit' hoac 'thoat' de thoat chuong trinh.")
    print()

    # Kiem tra he thong truoc khi bat dau
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

            answer = get_bot_response(user_query)

            print("\n" + "=" * 50)
            print("TOEIC MASTER TRA LOI:")
            print("=" * 50)
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
