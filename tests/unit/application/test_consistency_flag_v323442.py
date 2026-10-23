"""[v3.23.442] Test cờ nhất quán trong tập (nguồn giống, dịch khác — bỏ qua dấu câu)."""
from __future__ import annotations
from types import SimpleNamespace as NS
from subtitles_extractor.application.services.translation_diagnostics import detect_quality_flags


def _ev(i, t):
    return NS(index=i, text=t, start_sec=0.0, end_sec=2.0)


def test_flags_real_difference():
    src = [_ev(1, "苏大小姐的事"), _ev(2, "苏大小姐的事")]
    dst = [_ev(1, "Tô đại tiểu thư"), _ev(2, "Sô đại tiểu thư")]
    flags = detect_quality_flags(src, dst)
    assert flags["counts"]["consistency"] == 1


def test_ignores_punctuation_only_difference():
    src = [_ev(1, "救救我女儿啊"), _ev(2, "救救我女儿啊")]
    dst = [_ev(1, "Cứu con tôi với."), _ev(2, "Cứu con tôi với!")]
    flags = detect_quality_flags(src, dst)
    assert flags["consistency"] == []


def test_ignores_case_only_difference():
    src = [_ev(1, "我求求你们啊"), _ev(2, "我求求你们啊")]
    dst = [_ev(1, "Tôi cầu xin các người"), _ev(2, "tôi cầu xin các người")]
    flags = detect_quality_flags(src, dst)
    assert flags["consistency"] == []


def test_consistent_translation_not_flagged():
    src = [_ev(1, "野田智子啊"), _ev(2, "野田智子啊")]
    dst = [_ev(1, "Dã Điền Trí Tử"), _ev(2, "Dã Điền Trí Tử")]
    flags = detect_quality_flags(src, dst)
    assert flags["consistency"] == []


def test_short_source_skipped():
    # nguồn <4 ký tự → bỏ qua (tránh nhiễu thán từ)
    src = [_ev(1, "嗯"), _ev(2, "嗯")]
    dst = [_ev(1, "Ừ"), _ev(2, "Vâng")]
    flags = detect_quality_flags(src, dst)
    assert flags["consistency"] == []
