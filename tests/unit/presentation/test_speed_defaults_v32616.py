"""[v3.23.216] Test mặc định & chú thích tốc độ nhất quán sau khi base_speed có tác dụng.

Hệ quả của v215 (base_speed NÉN THẬT thay vì chỉ là nhãn):
1. **Mặc định 1.6 trở nên có hại**: mọi câu bị nén thật 1.6x -> mất ~8%% độ sắc phụ âm
   (đo trên giọng VieNeu thật: 1.1x mất 5%%, 1.6x mất 8%%, 2.0x mất 19%%). Chính tooltip
   của ô này đã khuyến nghị "1.0-1.2" -> mặc định 1.6 MÂU THUẪN. Hạ về 1.1.
2. **max_speed > 2.0 là cấu hình MA với VieNeu/Gemini**: trần chất lượng 2.0 chặn cứng
   (nén hậu kỳ hơn 2x làm tan formant). Tooltip nay nói rõ giới hạn theo từng engine.
3. ``stretch_ratio_cap`` / ``compute_fit_stretch_ratio`` không còn trong luồng chạy ->
   đánh dấu deprecated để không dùng nhầm cho code mới.
"""

from __future__ import annotations

import pathlib

_TTS_PAGE = pathlib.Path(
    "src/subtitles_extractor/presentation/pages/tts_page.py"
).read_text(encoding="utf-8")
_VIENEU = pathlib.Path(
    "src/subtitles_extractor/infrastructure/tts/vieneu_tts_adapter.py"
).read_text(encoding="utf-8")
# [v3.23.220] Các hàm thuần về tốc độ/thời lượng đã chuyển sang module dùng chung
# ``timing_math`` (VieNeu re-export lại) — nhãn deprecated nay nằm ở đó.
_TIMING_MATH = pathlib.Path(
    "src/subtitles_extractor/infrastructure/tts/timing_math.py"
).read_text(encoding="utf-8")


def test_default_base_speed_matches_recommendation() -> None:
    # Mặc định phải nằm trong khoảng CHÍNH tooltip khuyến nghị (1.0-1.2).
    assert "self._speed.setValue(1.1)" in _TTS_PAGE
    assert "self._speed.setValue(1.6)" not in _TTS_PAGE  # hết mặc định gấp gáp


def test_reset_defaults_uses_same_value() -> None:
    # "Khôi phục mặc định" và giá trị khởi tạo phải TRÙNG (trước đây cùng 1.6, nay 1.1).
    assert _TTS_PAGE.count("self._speed.setValue(1.1)") >= 2


def test_tooltip_recommendation_still_intact() -> None:
    assert "Khuyến nghị để 1.0–1.2" in _TTS_PAGE  # noqa: RUF001 (khớp tooltip)


def test_max_speed_tooltip_explains_quality_cap() -> None:
    # Người dùng phải BIẾT đặt max > 2.0 vô nghĩa với VieNeu/Gemini.
    assert "VieNeu / Gemini: nén hậu kỳ nên bị chặn ở 2.0×" in _TTS_PAGE  # noqa: RUF001
    assert "Edge TTS: dùng được tới 5.0×" in _TTS_PAGE  # noqa: RUF001


def test_legacy_helpers_marked_deprecated() -> None:
    assert ".. deprecated:: 3.23.215" in _TIMING_MATH
    assert "total_speed_ratio" in _TIMING_MATH  # hàm thay thế tồn tại
    assert "total_speed_ratio" in _VIENEU  # VieNeu vẫn dùng/re-export hàm mới


def test_total_speed_ratio_is_the_one_in_use() -> None:
    # Luồng chạy VieNeu dùng hàm mới, KHÔNG còn gọi hàm cũ.
    process_event = _VIENEU.split("def _process_event")[1]
    assert "total_speed_ratio(" in process_event
    assert "stretch_ratio_cap(" not in process_event
    assert "compute_fit_stretch_ratio(" not in process_event
