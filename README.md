# 🎓 TOEIC Master — Gia sư TOEIC AI

> Chatbot hỏi-đáp TOEIC thông minh, chạy **hoàn toàn offline** trên máy cá nhân, sử dụng kỹ thuật **RAG (Retrieval-Augmented Generation)** kết hợp **Hybrid Search (FAISS + BM25)** và **Ollama**.

---

## ✨ Tính năng

- 🔍 **Hybrid Search**: Kết hợp BM25 (keyword) + FAISS (semantic) với Reciprocal Rank Fusion
- 🤖 **LLM offline**: Chạy model `gemma2` qua Ollama, không cần internet hay API key
- 🖥️ **Web UI**: Giao diện Streamlit hiện đại, hỗ trợ streaming response
- 🧠 **Conversational Memory**: Ghi nhớ lịch sử hội thoại trong phiên
- 📚 **Phạm vi chuyên biệt**: Chỉ giải đáp ngữ pháp, từ vựng và bài thi TOEIC
- 🛡️ **Chống prompt injection**: Có cơ chế bảo vệ system prompt
- ♻️ **Singleton + Cache**: Load model & FAISS 1 lần duy nhất, tái sử dụng cho toàn phiên

---

## 🏗️ Cấu trúc dự án

```
TTNT/
├── streamlit_app.py      # Điểm vào chính — Web UI Streamlit
├── requirements.txt      # Danh sách thư viện Python
├── src/
│   ├── __init__.py
│   ├── config.py         # Cấu hình model, đường dẫn, system prompt
│   ├── model.py          # Khởi tạo Embedding model & LLM (Singleton)
│   ├── rag.py            # Hybrid Search: FAISS + BM25 + RRF merge
│   ├── ask.py            # Pipeline RAG → Prompt → LLM → Stream output
│   └── memory.py         # Quản lý lịch sử hội thoại
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

### Bước 3 — Chạy Web UI

```bash
streamlit run streamlit_app.py
```

Trình duyệt sẽ tự mở tại `http://localhost:8501`

> ⏳ **Lần đầu chạy** sẽ mất 10–40s để load Embedding model và FAISS index vào RAM. Các lần sau sẽ nhanh hơn nhờ `@st.cache_resource`.

---

## 💬 Ví dụ sử dụng

```
Bạn: Despite ------- in the project, the team managed to meet the deadline.
     A) participate  B) participating  C) participated  D) participation

Bot: **Đáp án đúng:** B) participating

     **Dịch nghĩa:** Mặc dù tham gia vào dự án, nhóm vẫn kịp deadline.

     **Giải thích chi tiết:** Sau "Despite" (giới từ), động từ phải ở dạng V-ing...
```

---

## 🗺️ Roadmap (dự kiến)

- [x] Giao diện web (Streamlit)
- [x] Hybrid Search (BM25 + FAISS + RRF)
- [x] Tích hợp lịch sử chat (Conversational Memory)
- [x] Streaming response từng token
- [ ] Hỗ trợ upload file PDF cá nhân
- [ ] Thêm dữ liệu TOEIC Part 6, 7

---

## ⚠️ Lưu ý

- Phải **khởi động Ollama** (`ollama serve`) trước khi chạy ứng dụng
- Thư mục `data/` phải tồn tại với 2 file `index.faiss` và `index.pkl`
- Dự án đang trong giai đoạn **phát triển**, chưa hoàn chỉnh

---

## 📄 License

MIT License — Xem file [LICENSE](LICENSE) để biết thêm chi tiết.
