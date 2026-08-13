"""[v3.23.148] Test NHẤT QUÁN is_dialog giữa scheduler <-> placement (khoảng lặng hội thoại).

Bug: is_dialog (quyết định chèn dialog_pause) được tính KHÁC NHAU ở 3 nơi:
- scheduler ``pause_for``: không skip, không strip tag -> "[Nam:] - Chào" bị coi KHÔNG
  hội thoại -> lịch không dành chỗ cho khoảng lặng -> audio dài hơn lịch -> cắt oan.
- placement (nơi chèn pause thật): có strip nhưng không skip -> "(cười) - Chào" khi bật
  skip ngoặc bị MẤT khoảng lặng hội thoại.
Fix: cả hai nơi dùng đúng văn bản ÂM THANH THẬT (skip + strip_speaker_tag) như Pass 1/2.
"""

from __future__ import annotations

from types import SimpleNamespace

from subtitles_extractor.infrastructure.tts.edge_tts_adapter import (
    EdgeTTSAdapter,
    _preprocess_tts_text,
    _skip_from_request,
)


def _request(**overrides) -> SimpleNamespace:
    base = dict(
        dialog_pause_ms=300, max_drift_s=1.5, base_speed=1.0, max_speed=2.0,
        clean_tags=True, lead_in_s=0.0, last_line_max_extend_s=0.0,
        timing_strategy="smooth",  # tất định: dur = pause + speech/base
        skip_paren=True, skip_square=False, skip_curly=False,
        skip_music_pair=False, skip_music_line=False,
        allow_audio_overlap=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _schedule_one(text: str) -> tuple[float, float, float]:
    events = [SimpleNamespace(text=text, start_sec=0.0, end_sec=2.0)]
    request = _request()
    schedule = EdgeTTSAdapter._schedule_timeline(events, {0: 1.0}, request)
    return schedule[0]


# ── _preprocess_tts_text: is_dialog trên văn bản âm thanh thật ───────────


def test_speaker_tag_then_dash_is_dialog() -> None:
    req = _request()
    _, is_dialog = _preprocess_tts_text(
        "[Nam:] - Xin chào", True, _skip_from_request(req), strip_speaker_tag=True
    )
    assert is_dialog is True


def test_paren_then_dash_is_dialog_when_skip_paren() -> None:
    req = _request()
    _, is_dialog = _preprocess_tts_text(
        "(cười) - Xin chào", True, _skip_from_request(req), strip_speaker_tag=True
    )
    assert is_dialog is True


# ── scheduler: lịch phải DÀNH CHỖ cho pause đúng như placement sẽ chèn ───


def test_schedule_reserves_pause_for_tagged_dialog() -> None:
    start, end, _speed = _schedule_one("[Nam:] - Xin chào")
    # dur = dialog_pause (0.3s) + speech (1.0s / base 1.0) = 1.3s.
    assert abs((end - start) - 1.3) < 0.01


def test_schedule_reserves_pause_for_paren_dialog() -> None:
    start, end, _speed = _schedule_one("(cười) - Xin chào")
    assert abs((end - start) - 1.3) < 0.01


def test_schedule_no_pause_for_plain_line() -> None:
    start, end, _speed = _schedule_one("Xin chào mọi người")
    assert abs((end - start) - 1.0) < 0.01
