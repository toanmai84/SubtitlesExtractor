"""[v3.23.440] Test giãn thời gian hiển thị phụ đề theo tốc độ đọc."""
from __future__ import annotations
from subtitles_extractor.application.services.reading_time_extender import (
    compute_reading_end, extend_events_for_reading,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.domain.value_objects.confidence import Confidence


def _ev(idx, text, s, e):
    return SubtitleEvent(index=idx, text=text, interval=TimeInterval(s, e),
                         confidence=Confidence(1.0), frame_count=1)


def test_extends_into_gap():
    # 40 ký tự trong 1s = 40 CPS; gap tới 15s → giãn về 20 CPS (cần 2s)
    end = compute_reading_end(40, 10.0, 11.0, 15.0, target_cps=20)
    assert abs(end - 12.0) < 0.01


def test_respects_next_start_guard():
    end = compute_reading_end(40, 10.0, 11.0, 11.5, target_cps=20)
    assert abs(end - 11.4) < 0.01  # chừa 0.1s


def test_already_readable_untouched():
    assert compute_reading_end(10, 0.0, 2.0, 100.0, target_cps=20) == 2.0


def test_small_gain_skipped():
    assert compute_reading_end(22, 0.0, 1.0, 1.15, target_cps=20) == 1.0


def test_max_duration_cap():
    end = compute_reading_end(300, 0.0, 1.0, 100.0, target_cps=20, max_duration_sec=7.0)
    assert abs(end - 7.0) < 0.01


def test_disabled_when_target_zero():
    events = [_ev(1, "x" * 40, 0.0, 1.0), _ev(2, "y", 10.0, 11.0)]
    out, n = extend_events_for_reading(events, target_cps=0)
    assert n == 0 and out[0].end_sec == 1.0


def test_extend_events_end_to_end():
    events = [_ev(1, "x" * 40, 0.0, 1.0), _ev(2, "y", 10.0, 11.0)]
    out, n = extend_events_for_reading(events, target_cps=20)
    assert n == 1
    assert out[0].end_sec > 1.0  # dòng 1 được giãn
    assert out[1].end_sec == 11.0  # dòng 2 không đổi
    assert events[0].end_sec == 1.0  # KHÔNG mutate gốc


def test_last_event_extends_to_max():
    events = [_ev(1, "x" * 100, 0.0, 1.0)]  # câu cuối, không có next
    out, n = extend_events_for_reading(events, target_cps=20, max_duration_sec=7.0)
    assert n == 1 and abs(out[0].end_sec - 5.0) < 0.01  # 100c/20cps=5s
