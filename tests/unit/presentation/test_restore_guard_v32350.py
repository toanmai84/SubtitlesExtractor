"""Test [v3.23.50] bảo vệ nạp lại khi phụ đề nguồn đã thay đổi (tránh gán sai dòng)."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PySide6.QtWidgets import QApplication

from subtitles_extractor.composition.bootstrap import bootstrap_for_gui
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationStageKind


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _vm(c, events):
    from subtitles_extractor.presentation.view_models.translate_page_view_model import (
        TranslatePageViewModel,
    )
    vm = TranslatePageViewModel(c)
    vm.set_source_events(events)
    return vm


def _small_src():
    return [
        SubtitleEvent(index=1, text="a", interval=TimeInterval(0.0, 1.0)),
        SubtitleEvent(index=2, text="b", interval=TimeInterval(1.0, 2.0)),
    ]


def _save_two_lines(vm):
    vm._last_translate_video_path = "/fake/guard_test.mp4"
    vm._stage_outputs = {
        TranslationStageKind.LITERAL: [
            SubtitleEvent(index=1, text="A", interval=TimeInterval(0.0, 1.0)),
            SubtitleEvent(index=2, text="B", interval=TimeInterval(1.0, 2.0)),
        ],
    }
    vm._persist_stage_outputs()


def test_skip_when_source_changed_drastically(app) -> None:
    c = bootstrap_for_gui()
    _save_two_lines(_vm(c, _small_src()))
    # Nguồn mới 50 dòng → lệch lớn → bỏ qua.
    big = [
        SubtitleEvent(index=i, text=str(i), interval=TimeInterval(i, i + 1))
        for i in range(1, 51)
    ]
    vm = _vm(c, big)
    assert vm.restore_stage_outputs_for_video("/fake/guard_test.mp4") == 0


def test_restore_when_source_matches(app) -> None:
    c = bootstrap_for_gui()
    _save_two_lines(_vm(c, _small_src()))
    vm = _vm(c, _small_src())
    assert vm.restore_stage_outputs_for_video("/fake/guard_test.mp4") == 1


def test_restore_tolerates_small_diff(app) -> None:
    c = bootstrap_for_gui()
    _save_two_lines(_vm(c, _small_src()))
    # Thêm 1 dòng (lệch nhỏ, trong ngưỡng cho phép tối thiểu 5) → vẫn nạp.
    src3 = _small_src() + [
        SubtitleEvent(index=3, text="c", interval=TimeInterval(2.0, 3.0))
    ]
    vm = _vm(c, src3)
    assert vm.restore_stage_outputs_for_video("/fake/guard_test.mp4") == 1
