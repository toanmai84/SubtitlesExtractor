"""[v3.23.196] Test trần time-stretch ĐÚNG ngữ nghĩa max_speed + trần chất lượng.

Hai lỗi từ người dùng (dữ liệu 55 câu thật):
1. Đặt max 3.0 nhưng có câu đọc 3.90x/3.18x: bug ngữ nghĩa v192 — ``max_speed`` là trần
   tốc độ TỔNG (base x ratio) nhưng bị dùng làm trần riêng của ratio -> 1.3 x 3.0 = 3.9.
   Trần đúng: ``max_speed / base_speed``.
2. Câu nén cao "như không có tiếng": phase-vocoder/WSOLA nén >2x làm tan formant ->
   tiếng gió thều thào. Thêm trần chất lượng ``_QUALITY_STRETCH_CAP = 2.0`` — thà đọc
   RÕ 2x và cắt tối thiểu phần dư còn hơn đọc đủ mà không nghe được.
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import (
    _QUALITY_STRETCH_CAP,
    compute_fit_stretch_ratio,
    stretch_ratio_cap,
)


def test_total_speed_never_exceeds_user_max() -> None:
    # Ca thực tế: base 1.3, max 3.0 -> trần ratio 2.0 -> tổng tối đa 2.6 <= 3.0.
    cap = stretch_ratio_cap(1.3, 3.0)
    assert 1.3 * cap <= 3.0


def test_case_from_user_data_sentence_8() -> None:
    # Câu #8 cần ratio 3.0 (speed cũ 3.90x VƯỢT max) -> giờ chặn 2.0 (speed 2.6x).
    cap = stretch_ratio_cap(1.3, 3.0)
    ratio = compute_fit_stretch_ratio(3.0, 1.0, cap)
    assert ratio == 2.0
    assert 1.3 * ratio == 2.6  # <= 3.0, trong vùng nghe rõ


def test_semantic_cap_tighter_than_quality() -> None:
    # base 2.0, max 3.0 -> trần ngữ nghĩa 1.5 CHẶT hơn trần chất lượng 2.0.
    assert stretch_ratio_cap(2.0, 3.0) == 1.5


def test_quality_cap_applies_when_semantic_loose() -> None:
    # base 1.0, max 10.0 -> ngữ nghĩa 10.0 nhưng chất lượng chặn 2.0.
    assert stretch_ratio_cap(1.0, 10.0) == _QUALITY_STRETCH_CAP


def test_no_compress_when_max_equals_base() -> None:
    # max = base -> không được nén thêm (ratio 1.0).
    assert stretch_ratio_cap(1.3, 1.3) == 1.0


def test_zero_base_falls_back_safely() -> None:
    assert stretch_ratio_cap(0.0, 3.0) == _QUALITY_STRETCH_CAP


def test_cap_never_below_one() -> None:
    # max < base (cấu hình lạ) -> không ép giãn ngược, trả 1.0.
    assert stretch_ratio_cap(2.0, 1.0) == 1.0


def test_custom_quality_cap() -> None:
    assert stretch_ratio_cap(1.0, 5.0, quality_cap=1.8) == 1.8
