"""[v3.23.652] Khôi phục chữ rơi không được chèn chữ lạ khi chỉ thiểu số nhỏ khung đọc được (ca thật trên phim 86 tập)."""

from __future__ import annotations

from subtitles_extractor.application.services.subtitle_pipeline.text_correction import apply_yi_suffix_restore


def test_rare_suffix_not_appended() -> None:
    voted = "终于被我逮到了吧"
    candidates = [(voted, 0.99)] * 27 + [(voted + "天", 0.95)] * 4  # 4/31 khung lẫn chữ cảnh sau
    assert apply_yi_suffix_restore(voted, candidates) == voted


def test_frequent_suffix_still_appended() -> None:
    voted = "感觉丹田要撑爆"
    candidates = [(voted, 0.99)] * 16 + [(voted + "了", 0.99)] * 8
    assert apply_yi_suffix_restore(voted, candidates) == voted + "了"


def test_yi_threshold_unchanged() -> None:
    voted = "整整盆灵液"
    candidates = [(voted, 0.99)] * 30 + [("整整一盆灵液", 0.99)]
    assert apply_yi_suffix_restore(voted, candidates) == "整整一盆灵液"
