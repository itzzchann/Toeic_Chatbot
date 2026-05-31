"""
DATASET.PY - Golden Dataset cho RAGAS Evaluation
3 câu đại diện để test nhanh (1 câu mỗi nhóm):
  - 1 câu Part 5 ngữ pháp
  - 1 câu lý thuyết ngữ pháp
  - 1 câu từ vựng TOEIC
"""

TOEIC_GOLDEN_DATASET = [

    # ── NHÓM 1: PART 5 NGỮ PHÁP ──
    {
        "user_input": (
            "Despite ___ extensive experience, the applicant was not offered the position.\n"
            "A) have  B) has  C) having  D) had"
        ),
        "reference": (
            "Đáp án đúng: C) having\n"
            "'Despite' là giới từ (preposition), sau giới từ phải dùng V-ing (gerund). "
            "Do đó 'having' là đúng. A) have và D) had là dạng chia theo ngôi/thì không dùng sau giới từ. "
            "B) has cũng là dạng chia theo ngôi, không hợp lệ."
        ),
    },

    # ── NHÓM 2: LÝ THUYẾT NGỮ PHÁP ──
    {
        "user_input": "Phân biệt cách dùng 'despite' và 'although' trong tiếng Anh?",
        "reference": (
            "'Despite' là giới từ (preposition), theo sau là danh từ hoặc V-ing. "
            "Ví dụ: Despite the rain, we continued.\n"
            "'Although' là liên từ (conjunction), theo sau là mệnh đề đầy đủ (S + V). "
            "Ví dụ: Although it rained, we continued.\n"
            "Không được dùng lẫn lộn: 'Despite that S+V' hay 'Although + N' đều SAI."
        ),
    },

    # ── NHÓM 3: TỪ VỰNG TOEIC ──
    {
        "user_input": "'remuneration' trong TOEIC nghĩa là gì và dùng trong ngữ cảnh nào?",
        "reference": (
            "'Remuneration' (danh từ) nghĩa là thù lao, tiền lương, khoản thưởng cho công việc. "
            "Đây là từ trang trọng trong văn phòng/hợp đồng, thay thế cho 'salary' hay 'payment'.\n"
            "Ví dụ: The remuneration package includes a base salary and annual bonus.\n"
            "Từ liên quan: remunerate (động từ), remunerative (tính từ)."
        ),
    },
]
