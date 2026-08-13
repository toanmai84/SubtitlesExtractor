"""[v3.23.194] Test khớp giọng mềm giữa các chế độ + khung hiệu dụng tận dụng gap.

Hai vấn đề từ người dùng (v3 Turbo):
1. Lỗi "Voice 'Ngoc' not found": mỗi chế độ VieNeu có BỘ GIỌNG KHÁC (7 vs 10, tên khác);
   đổi chế độ nhưng UI giữ giọng cũ -> SDK raise phá cả phiên. Fix: ``match_voice_name``
   khớp mềm (không dấu/hoa-thường/tiền tố) + fallback giọng đầu tiên kèm cảnh báo.
2. Giọng kém khi nén tốc độ: nén theo khung gốc lên tới 3x (31 câu >2x nghe gấp/méo).
   Fix: ``effective_available_seconds`` tận dụng GAP tới câu sau (median 1.79s) -> mô
   phỏng trên 498 câu thật: nén mạnh giảm 31 -> 14 (-55%%).
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import (
    effective_available_seconds,
    match_voice_name,
)

_VOICES_V3 = [
    "Trúc Ly", "Phạm Tuyên", "Thái Sơn", "Xuân Vĩnh", "Thanh Bình",
    "Minh Đức", "Ngọc Linh", "Đoan Trang", "Mai Anh", "Thục Đoan",
]


# ── match_voice_name ─────────────────────────────────────────────────────


def test_exact_match() -> None:
    assert match_voice_name("Thanh Bình", _VOICES_V3) == "Thanh Bình"


def test_case_insensitive_match() -> None:
    assert match_voice_name("thanh bình", _VOICES_V3) == "Thanh Bình"


def test_diacritic_insensitive_match() -> None:
    # Ca thực tế từ log lỗi: 'Ngoc' phải khớp 'Ngọc Linh' (tiền tố không dấu).
    assert match_voice_name("Ngoc", _VOICES_V3) == "Ngọc Linh"


def test_folded_full_match() -> None:
    assert match_voice_name("ngoc linh", _VOICES_V3) == "Ngọc Linh"


def test_no_match_returns_none() -> None:
    assert match_voice_name("XYZ", _VOICES_V3) is None


def test_empty_returns_none() -> None:
    assert match_voice_name("", _VOICES_V3) is None


def test_dj_character_folded() -> None:
    # 'đ' -> 'd': 'Doan Trang' khớp 'Đoan Trang'.
    assert match_voice_name("Doan Trang", _VOICES_V3) == "Đoan Trang"


# ── effective_available_seconds ──────────────────────────────────────────


def test_gap_extends_available() -> None:
    # Khung 2s, câu sau bắt đầu ở 3.6 -> gap 1.6 - guard 0.1 = 1.5 -> hiệu dụng 3.5.
    assert effective_available_seconds(0.0, 2.0, 3.6) == 3.5


def test_no_next_event_adds_cap() -> None:
    # Câu cuối: nới bằng trần gap (2.0) -> 2 + 2 = 4.
    assert effective_available_seconds(0.0, 2.0, None) == 4.0


def test_negative_gap_keeps_base() -> None:
    # Câu sau đè lên khung (gap âm) -> giữ nguyên khung gốc.
    assert effective_available_seconds(0.0, 2.0, 1.9) == 2.0


def test_gap_capped_at_max_use() -> None:
    # Gap 10s nhưng trần dùng 2s -> hiệu dụng 2 + 2 = 4.
    assert effective_available_seconds(0.0, 2.0, 12.0) == 4.0


def test_tiny_gap_below_guard() -> None:
    # Gap 0.08 < guard 0.1 -> không dùng được, giữ khung gốc.
    assert effective_available_seconds(0.0, 2.0, 2.08) == 2.0
