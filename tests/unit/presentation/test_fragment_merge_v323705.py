"""[v3.23.705] Đọc liền cụm câu: gộp các mảnh câu CÙNG người nói mà giọng đọc không kịp.

Đo trên lượt chạy thật 3664 dòng (VieNeu): 329 cụm, số dòng cần đọc > 1,5× giảm 717 → 369.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from subtitles_extractor.infrastructure.tts.tts_fit import (
    FRAGMENT_MAX_LINES,
    FRAGMENT_MAX_SYLLABLES,
    plan_fragment_merges,
)


def _ev(start: float, end: float, text: str) -> SimpleNamespace:
    return SimpleNamespace(start_sec=start, end_sec=end, text=text)


#: Ca thật (tập 40): mảnh 0,36 s nối liền câu sau.
_REAL = [
    _ev(136.515, 137.435, "Điên rồi sao?"),
    _ev(137.555, 137.915, "Mới lần hai"),
    _ev(137.955, 139.595, "mà đã giáng năm mươi đạo cùng lúc sao?"),
    _ev(141.475, 142.115, "Diệp Nhiên!"),
]


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


class TestPlan:
    def test_real_fragment_is_merged_with_its_continuation(self) -> None:
        assert plan_fragment_merges(_REAL) == [(1, 2)]

    def test_different_speakers_are_never_merged(self) -> None:
        assert plan_fragment_merges(_REAL, ["A", "A", "B", "B"]) == []
        assert plan_fragment_merges(_REAL, ["A", "B", "B", "B"]) == [(1, 2)]

    def test_finished_sentence_is_not_merged_into_next(self) -> None:
        events = [_ev(0.0, 0.36, "Mới lần hai."), _ev(0.4, 2.0, "mà đã giáng năm mươi đạo cùng lúc sao?")]
        assert plan_fragment_merges(events) == []

    def test_gap_breaks_the_group(self) -> None:
        events = [_ev(0.0, 0.36, "Mới lần hai"), _ev(0.9, 2.5, "mà đã giáng năm mươi đạo cùng lúc sao?")]
        assert plan_fragment_merges(events) == []

    def test_comfortable_lines_are_left_alone(self) -> None:
        events = [_ev(0.0, 2.0, "Hôm nay trời đẹp,"), _ev(2.0, 4.0, "ta đi dạo thôi.")]
        assert plan_fragment_merges(events) == []

    def test_no_gain_no_merge(self) -> None:
        # Cả hai mảnh đều quá nhanh: gộp không giúp gì → không đụng.
        events = [
            _ev(0.0, 0.4, "một hai ba bốn năm sáu"), _ev(0.4, 0.8, "bảy tám chín mười mười một."),
            _ev(0.8, 3.0, "Câu sau."),
        ]
        assert plan_fragment_merges(events) == []

    def test_limits_and_no_overlap(self) -> None:
        events = [_ev(i * 0.4, i * 0.4 + 0.4, "ba bốn năm sáu" if i % 2 else "một,") for i in range(40)]
        plans = plan_fragment_merges(events)
        assert plans
        previous_end = -1
        for first, last in plans:
            assert first > previous_end and 2 <= last - first + 1 <= FRAGMENT_MAX_LINES
            words = sum(len(events[p].text.split()) for p in range(first, last + 1))
            assert words <= FRAGMENT_MAX_SYLLABLES
            previous_end = last

    def test_empty_and_degenerate_input(self) -> None:
        assert plan_fragment_merges([]) == []
        assert plan_fragment_merges([_ev(0.0, 0.1, "một hai ba bốn")]) == []
        assert plan_fragment_merges([_ev(0.0, 0.0, ""), _ev(0.0, 0.0, "")]) == []


def _vm(events):  # noqa: ANN001, ANN202
    from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
    from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
    from subtitles_extractor.presentation.view_models.tts_page_view_model import TTSPageViewModel

    vm = TTSPageViewModel(SimpleNamespace())
    vm.set_source_events([
        SubtitleEvent(index=i + 1, interval=TimeInterval(e.start_sec, e.end_sec), text=e.text)
        for i, e in enumerate(events)
    ])
    return vm


class TestViewModel:
    def test_merge_is_one_undo_step_and_returns_final_positions(self, app: QApplication) -> None:
        events = _REAL + [_ev(e.start_sec + 100.0, e.end_sec + 100.0, e.text) for e in _REAL]
        vm = _vm(events)
        assert vm.fragment_merge_plan() == [(1, 2), (5, 6)]
        merged = vm.merge_fast_fragments()
        assert merged == [1, 4]
        texts = [e.text for e in vm.source_events]
        assert len(texts) == 6
        assert texts[1] == texts[4] == "Mới lần hai mà đã giáng năm mươi đạo cùng lúc sao?"
        assert vm.source_events[1].start_sec == pytest.approx(137.555)
        assert vm.source_events[1].end_sec == pytest.approx(139.595)
        assert vm.fragment_merge_plan() == []
        assert vm.undo()
        assert [e.text for e in vm.source_events] == [e.text for e in events]
        assert not vm.can_undo

    def test_nothing_to_merge(self, app: QApplication) -> None:
        vm = _vm([_ev(0.0, 2.0, "Hôm nay trời đẹp,"), _ev(2.0, 4.0, "ta đi dạo thôi.")])
        assert vm.merge_fast_fragments() == [] and not vm.can_undo

    def test_large_film_merges_quickly(self, app: QApplication) -> None:
        events = []
        for block in range(300):
            events += [_ev(e.start_sec + block * 10.0, e.end_sec + block * 10.0, e.text) for e in _REAL]
        vm = _vm(events)
        started = time.perf_counter()
        merged = vm.merge_fast_fragments()
        assert len(merged) == 300 and len(vm.source_events) == 900
        assert time.perf_counter() - started < 20.0


class TestPage:
    def test_button_counts_groups_and_merges(self, app: QApplication, tmp_path) -> None:  # noqa: ANN001
        from pathlib import Path

        from subtitles_extractor.composition.container import ApplicationContainer
        from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
        from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
        from subtitles_extractor.infrastructure.settings.json_file_repository import JsonFileRepository
        from subtitles_extractor.presentation.pages import translate_page
        from subtitles_extractor.presentation.pages.tts_page import TTSPage

        container = ApplicationContainer(
            settings_repository=JsonFileRepository(file_path=tmp_path / "c.json"),
            i18n_data_dir=Path(translate_page.__file__).resolve().parents[2] / "data",
            user_data_dir=tmp_path,
        )
        page = TTSPage(container)
        try:
            page._view_model.set_source_events([
                SubtitleEvent(index=i + 1, interval=TimeInterval(e.start_sec, e.end_sec), text=e.text)
                for i, e in enumerate(_REAL)
            ])
            controller = page._line_edits
            button = controller.action_buttons()["merge_fragments"]
            app.processEvents()
            assert controller.merge_fast_fragments() == 1
            assert len(page._view_model.source_events) == 3
            app.processEvents()
            assert not button.isEnabled()          # hết cụm cần gộp
            assert controller.merge_fast_fragments() == 0
        finally:
            page.close()
