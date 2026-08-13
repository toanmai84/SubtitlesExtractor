"""Unit test cho logic hiệu năng UI (v3.14.1, Nhóm 4 — phần thuần, không cần Qt).

Phủ: serialize/deserialize Visual Cues (Auto-Save), throttle theo khoảng thời
gian (Editor Throttle), và context manager khoá vẽ lại (UI Freeze Prevention).
"""

from __future__ import annotations

from subtitles_extractor.application.services.visual_cue_serializer import (
    deserialize_visual_cues,
    serialize_visual_cues,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import VisualCue
from subtitles_extractor.presentation.utils.ui_performance import (
    IntervalThrottle,
    batched_widget_update,
)


class TestVisualCueSerializer:
    def test_round_trip(self) -> None:
        cues = [
            VisualCue(line_no=1, speaker="Lâm Côn", addressee="Nữ tỳ", scene="bình tĩnh"),
            VisualCue(line_no=2, speaker="Diệp Thiên", addressee="", scene="tức giận"),
        ]
        restored = deserialize_visual_cues(serialize_visual_cues(cues))
        assert restored == cues

    def test_deserialize_empty_and_garbage(self) -> None:
        assert deserialize_visual_cues("") == []
        assert deserialize_visual_cues(None) == []
        assert deserialize_visual_cues("{not json") == []
        assert deserialize_visual_cues('{"a":1}') == []  # không phải list

    def test_deserialize_skips_bad_rows_keeps_good(self) -> None:
        raw = '[{"id":0,"spk":"x"},{"spk":"no-id"},{"id":3,"spk":"ok"}]'
        cues = deserialize_visual_cues(raw)
        assert len(cues) == 1 and cues[0].line_no == 3

    def test_deserialize_sorts_by_line_no(self) -> None:
        raw = '[{"id":5,"spk":"e"},{"id":2,"spk":"b"}]'
        cues = deserialize_visual_cues(raw)
        assert [c.line_no for c in cues] == [2, 5]

    def test_unicode_preserved(self) -> None:
        cues = [VisualCue(line_no=1, speaker="師尊", addressee="弟子", scene="đại điện")]
        assert deserialize_visual_cues(serialize_visual_cues(cues)) == cues


class TestIntervalThrottle:
    def test_first_call_runs(self) -> None:
        throttle = IntervalThrottle(min_interval_ms=100.0, clock=lambda: 0.0)
        assert throttle.should_run() is True

    def test_blocks_within_interval(self) -> None:
        now = {"t": 0.0}
        throttle = IntervalThrottle(min_interval_ms=100.0, clock=lambda: now["t"])
        assert throttle.should_run() is True   # t=0
        now["t"] = 0.05                          # +50ms < 100ms
        assert throttle.should_run() is False
        now["t"] = 0.12                          # +120ms ≥ 100ms kể từ lần chạy
        assert throttle.should_run() is True

    def test_reset_forces_next_run(self) -> None:
        now = {"t": 0.0}
        throttle = IntervalThrottle(min_interval_ms=100.0, clock=lambda: now["t"])
        throttle.should_run()
        throttle.reset()
        assert throttle.should_run() is True


class TestBatchedWidgetUpdate:
    class _FakeWidget:
        def __init__(self) -> None:
            self.history: list[bool] = []

        def setUpdatesEnabled(self, enabled: bool) -> None:
            self.history.append(enabled)

    def test_disables_then_enables(self) -> None:
        widget = self._FakeWidget()
        with batched_widget_update(widget):
            assert widget.history == [False]
        assert widget.history == [False, True]

    def test_re_enables_on_exception(self) -> None:
        widget = self._FakeWidget()
        try:
            with batched_widget_update(widget):
                raise ValueError("lỗi giữa chừng")
        except ValueError:
            pass
        assert widget.history == [False, True]  # vẫn bật lại
