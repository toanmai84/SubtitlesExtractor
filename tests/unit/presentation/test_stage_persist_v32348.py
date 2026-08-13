"""Test [v3.23.48] lưu bền vững + nạp lại kết quả từng giai đoạn vào session."""

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


def _src():
    return [
        SubtitleEvent(index=1, text="你好", interval=TimeInterval(0.0, 1.0)),
        SubtitleEvent(index=2, text="再见", interval=TimeInterval(1.0, 2.0)),
    ]


def test_persist_then_restore(app) -> None:
    from subtitles_extractor.presentation.view_models.translate_page_view_model import (
        TranslatePageViewModel,
    )
    c = bootstrap_for_gui()
    vm = TranslatePageViewModel(c)
    vm.set_source_events(_src())
    vm._last_translate_video_path = "/fake/stage_persist.mp4"
    vm._stage_outputs = {
        TranslationStageKind.LITERAL: [
            SubtitleEvent(index=1, text="Xin chào", interval=TimeInterval(0.0, 1.0)),
            SubtitleEvent(index=2, text="Tạm biệt", interval=TimeInterval(1.0, 2.0)),
        ],
        TranslationStageKind.STYLE: [
            SubtitleEvent(index=1, text="Chào bạn", interval=TimeInterval(0.0, 1.0)),
            SubtitleEvent(index=2, text="Hẹn gặp", interval=TimeInterval(1.0, 2.0)),
        ],
    }
    vm._persist_stage_outputs()

    # VM mới (mô phỏng mở lại app), nạp lại.
    vm2 = TranslatePageViewModel(c)
    vm2.set_source_events(_src())
    n = vm2.restore_stage_outputs_for_video("/fake/stage_persist.mp4")
    assert n == 2
    comp = vm2.stage_comparison()
    names = [name for name, _ in comp]
    assert names == ["Dịch thô", "Tinh chỉnh"]
    assert comp[0][1] == ["Xin chào", "Tạm biệt"]
    assert comp[1][1] == ["Chào bạn", "Hẹn gặp"]


def test_restore_no_data(app) -> None:
    from subtitles_extractor.presentation.view_models.translate_page_view_model import (
        TranslatePageViewModel,
    )
    c = bootstrap_for_gui()
    vm = TranslatePageViewModel(c)
    vm.set_source_events(_src())
    assert vm.restore_stage_outputs_for_video("/fake/never_translated.mp4") == 0


def test_restore_empty_path(app) -> None:
    from subtitles_extractor.presentation.view_models.translate_page_view_model import (
        TranslatePageViewModel,
    )
    c = bootstrap_for_gui()
    vm = TranslatePageViewModel(c)
    assert vm.restore_stage_outputs_for_video("") == 0
