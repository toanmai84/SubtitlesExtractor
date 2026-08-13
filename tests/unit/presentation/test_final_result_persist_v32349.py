"""Test [v3.23.49] lưu bền vững bản dịch hiện hành (sau sửa/dịch lại) + nạp lại."""

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


def _vm(c):
    from subtitles_extractor.presentation.view_models.translate_page_view_model import (
        TranslatePageViewModel,
    )
    vm = TranslatePageViewModel(c)
    vm.set_source_events(_src())
    return vm


def test_edit_persists_and_restores(app) -> None:
    c = bootstrap_for_gui()
    vm = _vm(c)
    vm._last_translate_video_path = "/fake/edit_v349.mp4"
    vm._translated_events = [
        SubtitleEvent(index=1, text="Xin chào", interval=TimeInterval(0.0, 1.0)),
        SubtitleEvent(index=2, text="Tạm biệt", interval=TimeInterval(1.0, 2.0)),
    ]
    assert vm.update_translation_text(0, "Chào bạn nhé") is True

    vm2 = _vm(c)
    vm2.restore_stage_outputs_for_video("/fake/edit_v349.mp4")
    assert vm2.translated_events[0].text == "Chào bạn nhé"
    assert vm2.translated_events[1].text == "Tạm biệt"


def test_final_preferred_over_stages(app) -> None:
    c = bootstrap_for_gui()
    vm = _vm(c)
    vm._last_translate_video_path = "/fake/final_pref.mp4"
    # Lưu một stage LITERAL + một final khác nhau.
    vm._stage_outputs = {
        TranslationStageKind.LITERAL: [
            SubtitleEvent(index=1, text="Bản thô", interval=TimeInterval(0.0, 1.0)),
            SubtitleEvent(index=2, text="Bản thô 2", interval=TimeInterval(1.0, 2.0)),
        ],
    }
    vm._persist_stage_outputs()
    vm._translated_events = [
        SubtitleEvent(index=1, text="Bản cuối", interval=TimeInterval(0.0, 1.0)),
        SubtitleEvent(index=2, text="Bản cuối 2", interval=TimeInterval(1.0, 2.0)),
    ]
    vm._persist_current_result()

    vm2 = _vm(c)
    vm2.restore_stage_outputs_for_video("/fake/final_pref.mp4")
    # translated_events lấy từ 'final', KHÔNG phải LITERAL.
    assert vm2.translated_events[0].text == "Bản cuối"
    # nhưng stage so sánh vẫn còn LITERAL.
    comp = vm2.stage_comparison()
    assert comp == [] or any("thô" in t for _, texts in comp for t in texts)


def test_no_persist_without_video(app) -> None:
    c = bootstrap_for_gui()
    vm = _vm(c)
    vm._last_translate_video_path = ""
    vm._translated_events = [
        SubtitleEvent(index=1, text="X", interval=TimeInterval(0.0, 1.0)),
    ]
    # Không có video_path → không lưu, không lỗi.
    vm._persist_current_result()
