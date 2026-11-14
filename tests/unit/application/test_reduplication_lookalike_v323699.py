"""[v3.23.699] Từ láy AABB bị OCR đọc nhầm chữ trông giống — "清清自白" → "清清白白" (lượt trích xuất thật GPU, 33/45 khung sai)."""

from __future__ import annotations

import pytest

from subtitles_extractor.application.services.subtitle_pipeline.text_correction import (
    correct_hallucination_typos,
    restore_reduplication,
)


def test_real_misread_is_fixed_through_the_typo_pass() -> None:
    assert correct_hallucination_typos("我跟秦执师兄清清自白的") == "我跟秦执师兄清清白白的"


@pytest.mark.parametrize("text", [
    "我跟秦执师兄清清白白的", "堂堂正正打败叶然", "哈哈哈哈", "你你你你", "等等等等",      # từ láy đúng (có trong đáp án)
    "明明白白", "自自", "清白", "我们家祖祖辈辈", "人人自危",                             # không đủ mẫu / không cặp nhầm
    "AA自白",                                                                       # không phải chữ Hán
])
def test_correct_text_is_untouched(text: str) -> None:
    assert restore_reduplication(text) == text


def test_other_lookalike_pairs() -> None:
    assert restore_reduplication("清清白自") == "清清白白"          # chữ nhầm ở vị trí thứ tư
    assert restore_reduplication("明明曰日") == "明明日日"
    assert restore_reduplication("平平安入") == "平平安入"          # 安≠入 không phải cặp nhầm
    # Cặp mơ hồ / không láy: giữ nguyên (bản đầu bảng hai chiều từng sửa thành "早早己己", "清清自自").
    assert restore_reduplication("早早已己") == "早早已己"
    assert restore_reduplication("千千干万") == "千千干万"
