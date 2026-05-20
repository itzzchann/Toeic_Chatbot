# 🎓 TOEIC Master — Gia sư TOEIC AI

> Chatbot hỏi-đáp TOEIC thông minh, chạy **hoàn toàn offline** trên máy cá nhân, sử dụng kỹ thuật **RAG (Retrieval-Augmented Generation)** kết hợp **FAISS** và **Ollama**.

---

## ✨ Tính năng

- 🔍 **RAG**: Truy xuất tài liệu TOEIC từ kho vector FAISS cục bộ để trả lời chính xác
- 🤖 **LLM offline**: Chạy model `gemma2` qua Ollama, không cần internet hay API key
- 📚 **Phạm vi chuyên biệt**: Chỉ giải đáp ngữ pháp, từ vựng và bài thi TOEIC
- 🛡️ **Chống prompt injection**: Có cơ chế bảo vệ system prompt
- ♻️ **Singleton Pattern**: Load model & FAISS 1 lần duy nhất, tái sử dụng cho toàn phiên

---

## 🏗️ Cấu trúc dự án

```
TTNT/
├── main.py               # Điểm vào chính — chạy chatbot CLI
├── requirements.txt      # Danh sách thư viện Python
├── src/
│   ├── config.py         # Cấu hình model, đường dẫn, system prompt
│   ├── model.py          # Khởi tạo Embedding model & LLM (Singleton)
│   ├── rag.py            # Load FAISS DB, tìm kiếm tài liệu liên quan
│   └── ask.py            # Pipeline RAG → Prompt → LLM → Output
└── data/
    ├── index.faiss       # FAISS vector index
    └── index.pkl         # Metadata của các chunk tài liệu
```

---

## ⚙️ Yêu cầu hệ thống

| Thành phần | Phiên bản khuyến nghị |
|---|---|
| Python | 3.10+ |
| Ollama | Mới nhất |
| RAM | Tối thiểu 8GB (khuyến nghị 16GB) |
| GPU | Không bắt buộc (CPU vẫn chạy được) |

---

## 🚀 Hướng dẫn cài đặt & chạy

### Bước 1 — Cài Ollama và pull model

Tải Ollama tại: https://ollama.com/download

```bash
# Pull model gemma2 về máy (~5GB)
ollama pull gemma2

# Khởi động Ollama server (để cửa sổ này mở)
ollama serve
```

### Bước 2 — Cài thư viện Python

```bash
# Tạo môi trường ảo (khuyến nghị)
python -m venv venv

# Kích hoạt môi trường ảo
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Cài thư viện
pip install -r requirements.txt
```

### Bước 3 — Chạy chatbot

```bash
python main.py
```

---

## 💬 Ví dụ sử dụng

```
==================================================
   TOEIC MASTER - Gia su TOEIC AI
==================================================
Go 'exit' hoac 'thoat' de thoat chuong trinh.

Dang khoi dong he thong...
He thong san sang!

Ban muon hoi gi ve TOEIC: Despite ------- in the project, the team managed to meet the deadline.
A) participate  B) participating  C) participated  D) participation

==================================================
TOEIC MASTER TRA LOI:
==================================================
**Đáp án đúng:** B) participating

**Dịch nghĩa:** Mặc dù tham gia vào dự án, nhóm vẫn kịp deadline.

**Giải thích chi tiết:** Sau "Despite" (giới từ), động từ phải ở dạng V-ing...
==================================================
```

---

## 🗺️ Roadmap (dự kiến)

- [ ] Giao diện web (Streamlit / Gradio)
- [ ] Hỗ trợ upload file PDF cá nhân
- [ ] Thêm dữ liệu TOEIC Part 6, 7
- [ ] Tích hợp lịch sử chat

---

## ⚠️ Lưu ý

- Phải **khởi động Ollama** (`ollama serve`) trước khi chạy `main.py`
- Thư mục `data/` phải tồn tại với 2 file `index.faiss` và `index.pkl`
- Dự án đang trong giai đoạn **phát triển**, chưa hoàn chỉnh

---

## 📄 License

MIT License — Xem file [LICENSE](LICENSE) để biết thêm chi tiết.
