"""[v3.23.742] "做这等腌攒之事" — chữ trên phim "腌臜" (ảnh, phim nối 86 tập 4996,9 s; cùng lỗi ở phim "[全集]").

Giai đoạn 0 kế hoạch 4.0 (đối chiếu PP-OCR với PaddleOCR-VL đọc lại MỌI câu, lượt 3.23.741 trên GPU): 3634/3663 câu hai
mô hình đồng ý; 29 câu bất đồng xem ảnh → ứng dụng đúng 28, sai đúng câu này. "攒" trông giống "臜"; "腌攒" không phải từ.
Không khai báo "腌" luôn đi với "臜" được (腌制, 腌菜) → bảng cụm đọc nhầm cụ thể.
"""

from __future__ import annotations

import pytest

from subtitles_extractor.application.services.subtitle_pipeline.text_correction import (
    correct_hallucination_typos,
)


@pytest.mark.parametrize(("raw", "fixed"), [
    ("做这等腌攒之事", "做这等腌臜之事"),
    ("你和他的那些腌攒事", "你和他的那些腌臜事"),
])
def test_misread_word_fixed(raw: str, fixed: str) -> None:
    assert correct_hallucination_typos(raw) == fixed


@pytest.mark.parametrize("text", ["腌制的咸菜", "腌菜", "攒钱", "积攒了很久"])
def test_real_words_untouched(text: str) -> None:
    assert correct_hallucination_typos(text) == text
