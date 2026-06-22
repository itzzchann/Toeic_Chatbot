# TOEIC Master — Gia sư TOEIC AI

> Chatbot hỏi-đáp TOEIC thông minh, chạy **hoàn toàn offline** trên máy cá nhân, sử dụng kỹ thuật **RAG (Retrieval-Augmented Generation)** kết hợp **Hybrid Search (ChromaDB + BM25)** và **Ollama**.

Hệ thống được triển khai theo hai pipeline:
- Pipeline 1 (Google Colab): thực hiện tiền xử lý dữ liệu, chuyển đổi PDF/DOCX sang Markdown, chunking, embedding và xây dựng Vector Database. Link: https://drive.google.com/drive/u/0/folders/1fqsmrJIrbRBGm8ZoXkPC26LEE75ZlyoA
- Pipeline 2 (Local): sử dụng trực tiếp Vector Database được tạo từ Pipeline 1 để thực hiện truy xuất thông tin và sinh câu trả lời cho chatbot.

Cách triển khai trên xuất phát từ yêu cầu kỹ thuật và quá trình phân công công việc của nhóm, giúp các thành viên có thể phát triển song song, dễ dàng tích hợp kết quả và không cần thực hiện lại bước xây dựng Vector Database trong quá trình phát triển và kiểm thử chatbot.

---

## Tính năng

- **Cải tiến Hybrid Search**: Kết hợp thuật toán tìm kiếm từ khóa BM25 (trọng số 0.7) và tìm kiếm ngữ nghĩa ChromaDB (trọng số 0.3), lấy ra 15 chunk mỗi nhánh và gộp kết quả bằng thuật toán Reciprocal Rank Fusion (RRF).
- **Pipeline xử lý dữ liệu tự động**: Tích hợp luồng chuyển đổi PDF/DOCX sang Markdown. Áp dụng cơ chế lọc nhiễu 2 lớp (thống kê lặp Header/Footer và Regex bắt watermark/quảng cáo), chuẩn hóa Unicode tiếng Việt. Phân mảnh văn bản (Chunking) thông minh kết hợp giữa Semantic và Recursive tùy theo phân loại tài liệu (Ngữ pháp, Đề thi, Mẹo).
- **Tiền xử lý câu hỏi thông minh**: Tách luồng câu hỏi, dùng LLM trích xuất Keyword cho BM25 và giữ nguyên ngữ cảnh cho ChromaDB. Bổ sung bộ lọc siêu dữ liệu (Metadata Filtering) bằng Regex.
- **Lọc kết quả nghiêm ngặt**: Áp dụng ngưỡng Cosine Distance < 0.6 và loại bỏ các chunk rác để đảm bảo chất lượng tài liệu trước khi trả về.
- **LLM offline**: Chạy model `gemma2` qua Ollama, không cần internet hay API key.
- **Web UI**: Giao diện Streamlit hiện đại, hỗ trợ streaming response.
- **Phạm vi chuyên biệt**: Chỉ giải đáp ngữ pháp, từ vựng và bài thi TOEIC. Tích hợp **Conversational Memory** để trò chuyện có ngữ cảnh.
- **Singleton + Cache**: Nạp mô hình ngôn ngữ, ChromaDB và bộ nhúng BM25 1 lần duy nhất, tối ưu tốc độ phản hồi cho toàn phiên.

---

## Cấu trúc dự án

```
TTNT/
├── streamlit_app.py      # Điểm vào chính — Web UI Streamlit
├── requirements.txt      # Danh sách thư viện Python
├── src/
│   ├── __init__.py
│   ├── config.py         # Cấu hình tham số RAG, đường dẫn, trọng số Hybrid Search
│   ├── model.py          # Khởi tạo Embedding model & LLM (Singleton)
│   ├── rag.py            # Hybrid Search (ChromaDB + BM25 + RRF), phân luồng tiền xử lý
│   ├── ask.py            # Pipeline RAG → Prompt → LLM → Stream output
│   └── memory.py         # Quản lý lịch sử hội thoại
├── tests/
│   └── ragas_eval/       # Đánh giá chất lượng RAG bằng Ragas
│       ├── dataset.py    # Xây dựng tập dữ liệu đánh giá
│       └── evaluate.py   # Chạy đánh giá và xuất báo cáo
└── data/
    ├── chroma.sqlite3    # Database vector ChromaDB
    └── (các thư mục ID)  # Dữ liệu index của Chroma
```

---

## Yêu cầu hệ thống

| Thành phần | Phiên bản khuyến nghị |
|---|---|
| Python | 3.10+ |
| Ollama | Mới nhất |
| RAM | Tối thiểu 8GB (khuyến nghị 16GB) |
| GPU | Không bắt buộc (CPU vẫn chạy được) |

---

## Hướng dẫn cài đặt & chạy

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

> **Lần đầu chạy** sẽ mất 10–40s để nạp các mô hình nhúng, đọc cơ sở dữ liệu ChromaDB và tính toán chỉ mục BM25 vào RAM. Các lần sau tốc độ phản hồi sẽ tức thì nhờ tính năng cache của Streamlit (`@st.cache_resource`).

---

## Ví dụ sử dụng

```
Bạn: Despite ------- in the project, the team managed to meet the deadline.
     A) participate  B) participating  C) participated  D) participation

Bot: **Đáp án đúng:** B) participating

     **Dịch nghĩa:** Mặc dù tham gia vào dự án, nhóm vẫn kịp deadline.

     **Giải thích chi tiết:** Sau "Despite" (giới từ), động từ phải ở dạng V-ing...
```

---

## Roadmap (dự kiến)

- [x] Pipeline xử lý dữ liệu tự động (PDF/DOCX → Markdown → Chunking → Vector DB)
- [x] Giao diện web (Streamlit)
- [x] Cải tiến Hybrid Search (BM25 + ChromaDB + RRF)
- [x] Tách luồng tiền xử lý câu hỏi (LLM extraction cho BM25) & Metadata Filtering
- [x] Tích hợp lịch sử chat (Conversational Memory)
- [x] Streaming response từng token
- [x] Đánh giá chất lượng RAG bằng Ragas
- [ ] Hỗ trợ thêm dữ liệu TOEIC Part 6, 7
- [ ] Hỗ trợ upload file PDF/DOCX tài liệu mới vào Vector DB trực tiếp từ giao diện

---

## Lưu ý

- Phải **khởi động Ollama** (`ollama serve`) trước khi chạy ứng dụng
- Thư mục `data/` phải tồn tại và chứa cơ sở dữ liệu `chroma.sqlite3`
- Quá trình chạy lần đầu cần có kết nối mạng nhẹ để thư viện nạp mô hình nhúng `multilingual-e5-base`

---

## License

MIT License — Xem file [LICENSE](LICENSE) để biết thêm chi tiết.
