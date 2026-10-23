"""[v3.23.438] Test cờ tốc độ đọc (CPS) — khoanh dòng phụ đề đọc không kịp."""
from __future__ import annotations
from types import SimpleNamespace as NS
from subtitles_extractor.application.services.translation_diagnostics import detect_quality_flags


def _ev(i, t, s, e):
    return NS(index=i, text=t, start_sec=s, end_sec=e)


def test_flags_high_cps_reliable_timing() -> None:
    src = [_ev(1, "你好", 0, 2)]
    dst = [_ev(1, "Câu này rất dài nên chắc chắn đọc không kịp trong hai giây ngắn ngủi", 0, 2)]
    flags = detect_quality_flags(src, dst)
    assert len(flags["reading_speed"]) == 1
    assert flags["reading_speed"][0]["index"] == 1
    assert flags["counts"]["reading_speed"] == 1


def test_ignores_short_duration_ocr_artifact() -> None:
    src = [_ev(1, "短", 4, 4.3)]
    dst = [_ev(1, "Dòng này timing chỉ 0.3 giây nên là lỗi OCR phải bỏ qua", 4, 4.3)]
    flags = detect_quality_flags(src, dst)
    assert flags["reading_speed"] == []


def test_normal_speed_not_flagged() -> None:
    src = [_ev(1, "你好", 0, 3)]
    dst = [_ev(1, "Xin chào bạn nhé", 0, 3)]  # 16c/3s ~ 5 CPS
    flags = detect_quality_flags(src, dst)
    assert flags["reading_speed"] == []


def test_empty_text_skipped() -> None:
    src = [_ev(1, "你好", 0, 2)]
    dst = [_ev(1, "", 0, 2)]
    flags = detect_quality_flags(src, dst)
    assert flags["reading_speed"] == []
