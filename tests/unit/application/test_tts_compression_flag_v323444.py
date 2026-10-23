"""[v3.23.444] Test cờ nén TTS (mặt tiếng nói của text-quá-dài-so-thời-lượng)."""
from __future__ import annotations
from types import SimpleNamespace as NS
from subtitles_extractor.application.services.translation_diagnostics import detect_quality_flags


def _ev(i, t, s, e):
    return NS(index=i, text=t, start_sec=s, end_sec=e)


def test_flags_heavy_compression():
    # 60 ký tự / 1.5s → TTS cần ~3.2s → nén ~2.1x (>1.6)
    src = [_ev(1, "你好", 0, 1.5)]
    dst = [_ev(1, "Câu này dài sáu mươi ký tự nên TTS phải nén rất mạnh nha bạn", 0, 1.5)]
    flags = detect_quality_flags(src, dst)
    assert flags["counts"]["tts_compression"] == 1
    assert flags["tts_compression"][0]["ratio"] > 1.6


def test_natural_pace_not_flagged():
    # 20 ký tự / 2s → TTS cần ~1.3s → nén <1 (thoải mái)
    src = [_ev(1, "你好", 0, 2)]
    dst = [_ev(1, "Xin chào các bạn nhé", 0, 2)]
    flags = detect_quality_flags(src, dst)
    assert flags["tts_compression"] == []


def test_short_duration_excluded():
    # timing OCR lỗi <0.8s → không tính (v440 giãn giờ xử)
    src = [_ev(1, "你好", 0, 0.3)]
    dst = [_ev(1, "Trời ạ thật là quá đáng mà", 0, 0.3)]
    flags = detect_quality_flags(src, dst)
    assert flags["tts_compression"] == []


def test_empty_skipped():
    src = [_ev(1, "你好", 0, 2)]
    dst = [_ev(1, "", 0, 2)]
    flags = detect_quality_flags(src, dst)
    assert flags["tts_compression"] == []
