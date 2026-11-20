"""[v3.23.753] Phim 大乾悍刀行 (84 tập) — gói kiểm định GLM-OCR (lượt 3.23.751 trên GPU) bất đồng 18/3609 câu; xem ảnh
từng câu: ứng dụng sai 12 (GLM đúng 11), GLM sai 6. Năm cặp chữ nhìn giống nhau tạo cụm không phải từ được sửa ở bảng cụm
đọc nhầm; các cụm SAI chỉ xuất hiện đúng chỗ đọc nhầm trên kết quả 6 phim; "剔" đúng nghĩa (削骨剔髓) không bị đụng.
"""

from __future__ import annotations

import pytest

from subtitles_extractor.application.services.subtitle_pipeline.text_correction import (
    correct_hallucination_typos,
)


@pytest.mark.parametrize(("raw", "fixed"), [
    ("筒直就是卑鄙下流", "简直就是卑鄙下流"),
    ("嘴硬叫器", "嘴硬叫嚣"),
    ("我大乾确实千旱少雨", "我大乾确实干旱少雨"),
    ("一千二净", "一干二净"),
    ("剔高兴地太早", "别高兴地太早"),
    ("剔冲动", "别冲动"),
])
def test_misread_word_fixed(raw: str, fixed: str) -> None:
    assert correct_hallucination_typos(raw) == fixed


@pytest.mark.parametrize("text", ["削骨剔髓", "剔除杂质", "竹筒", "一千两银子", "千里之外", "机器", "简直", "干旱"])
def test_real_words_untouched(text: str) -> None:
    assert correct_hallucination_typos(text) == text
