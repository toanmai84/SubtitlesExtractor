"""[v3.23.214] Test tuỳ chọn CHỈ-EDGE bị vô hiệu hoá / đánh dấu khi chạy engine khác.

Bug UX (cấu hình ma): quét mã 3 adapter cho thấy **14 trường** TTSRequest chỉ Edge đọc
(timing_strategy, elastic_timing, double_pass, high_quality, comfort_speed_ratio,
min_pause_ratio, max_drift_s, lead_in_s, last_line_max_extend_s, min_stretch_ratio,
max_intra_gap_s, anchor_gap_s, max_segment_s, edge_concurrency) — VieNeu/Gemini BỎ QUA
hoàn toàn. Nhưng UI hiện chúng cho MỌI engine và debug config in ra như đang có hiệu
lực -> người dùng chỉnh vô ích ("Elastic timing: Bật" khi chạy VieNeu) và việc phân
tích kết quả bị lệch hướng.

Fix: khi engine != Edge -> control bị vô hiệu hoá + tooltip "(chỉ áp dụng cho Edge
TTS)"; debug config đánh dấu "(không áp dụng cho engine này)".
"""

from __future__ import annotations

import pathlib

from subtitles_extractor.presentation.pages.tts_page import (
    _EDGE_ONLY_FIELDS,
    _EDGE_ONLY_HINT,
)

_SRC = pathlib.Path(
    "src/subtitles_extractor/presentation/pages/tts_page.py"
).read_text(encoding="utf-8")

_ADAPTERS = {
    name: pathlib.Path(
        f"src/subtitles_extractor/infrastructure/tts/{name}_tts_adapter.py"
    ).read_text(encoding="utf-8")
    for name in ("vieneu", "gemini", "edge")
}


def test_edge_only_fields_really_edge_only() -> None:
    # Nguồn sự thật: mỗi field trong tập phải XUẤT HIỆN ở Edge và VẮNG ở 2 engine kia.
    for field in _EDGE_ONLY_FIELDS:
        assert field in _ADAPTERS["edge"], f"{field} không có trong Edge"
        assert field not in _ADAPTERS["vieneu"], f"{field} thực ra VieNeu có dùng"
        assert field not in _ADAPTERS["gemini"], f"{field} thực ra Gemini có dùng"


def test_shared_fields_not_marked_edge_only() -> None:
    # Các tham số dùng chung KHÔNG được đánh dấu nhầm (sẽ khoá oan control hữu ích).
    for field in (
        "base_speed", "max_speed", "allow_audio_overlap", "max_overlap_ms",
        "skip_overlap_ms", "dialog_pause_ms", "normalize", "target_lufs",
        "voice_clarity", "retry_count", "gap_threshold_s",
    ):
        assert field not in _EDGE_ONLY_FIELDS


def test_controls_disabled_for_non_edge_engines() -> None:
    assert "def _sync_edge_only_controls(self, engine_id: str) -> None:" in _SRC
    assert "is_edge = engine_id == _ENGINE_EDGE" in _SRC
    assert "widget.setEnabled(is_edge)" in _SRC
    # Gọi mỗi khi đổi engine.
    assert "self._sync_edge_only_controls(engine_id)" in _SRC


def test_tooltip_hint_applied() -> None:
    assert _EDGE_ONLY_HINT.strip().startswith("—")
    assert "chỉ áp dụng cho Edge TTS" in _EDGE_ONLY_HINT
    assert "tip + _EDGE_ONLY_HINT" in _SRC  # thêm hậu tố khi không phải Edge


def test_debug_config_marks_inapplicable_fields() -> None:
    assert "(không áp dụng cho engine này)" in _SRC
    assert "if not is_edge and attr in _EDGE_ONLY_FIELDS:" in _SRC


def test_all_six_visible_widgets_covered() -> None:
    # 6 control người dùng nhìn thấy trực tiếp phải nằm trong danh sách vô hiệu hoá.
    for widget in (
        "self._strategy", "self._high_quality", "self._double_pass",
        "self._elastic_timing", "self._comfort_ratio", "self._min_pause",
    ):
        assert widget in _SRC.split("def _sync_edge_only_controls")[1].split("def ")[0]
