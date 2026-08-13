"""Test [v3.23.55] tích hợp Translation Memory vào view model (tập 1 → tập 2)."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PySide6.QtWidgets import QApplication

from subtitles_extractor.composition.bootstrap import bootstrap_for_gui
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _vm(c):
    from subtitles_extractor.presentation.view_models.translate_page_view_model import (
        TranslatePageViewModel,
    )
    return TranslatePageViewModel(c)


def test_accumulate_then_retrieve_across_episodes(app) -> None:
    c = bootstrap_for_gui()
    # Tập 1: dịch + tích luỹ.
    vm1 = _vm(c)
    vm1.set_source_events([
        SubtitleEvent(index=1, text="林恒走进来", interval=TimeInterval(0.0, 1.0)),
    ])
    vm1._last_translate_video_path = "/data/PhimBoTest55/EP01.mp4"
    vm1._translated_events = [
        SubtitleEvent(index=1, text="Lâm Hằng đi vào", interval=TimeInterval(0.0, 1.0)),
    ]
    vm1._accumulate_translation_memory()

    # Tập 2: truy hồi (cùng thư mục PhimBoTest55).
    vm2 = _vm(c)
    block = vm2.retrieve_memory_for_lines(
        "/data/PhimBoTest55/EP02.mp4", ["林恒又走进来了"]
    )
    assert "Lâm Hằng" in block


def test_no_memory_for_unknown_series(app) -> None:
    c = bootstrap_for_gui()
    vm = _vm(c)
    block = vm.retrieve_memory_for_lines(
        "/data/SeriesChuaTungDich99/EP01.mp4", ["你好"]
    )
    assert block == ""


def test_no_video_path_no_memory(app) -> None:
    c = bootstrap_for_gui()
    vm = _vm(c)
    assert vm.retrieve_memory_for_lines("", ["你好"]) == ""
