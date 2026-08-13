"""Test [v3.23.47] so sánh kết quả các giai đoạn dịch."""

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


def _ev(text):
    return SubtitleEvent(index=1, text=text, interval=TimeInterval(0.0, 1.0))


def test_comparison_needs_two_stages(app) -> None:
    from subtitles_extractor.presentation.view_models.translate_page_view_model import (
        TranslatePageViewModel,
    )
    vm = TranslatePageViewModel(bootstrap_for_gui())
    vm._stage_outputs = {TranslationStageKind.LITERAL: [_ev("Xin chào")]}
    assert vm.stage_comparison() == []  # 1 giai đoạn → không so sánh


def test_comparison_ordered_by_pipeline(app) -> None:
    from subtitles_extractor.presentation.view_models.translate_page_view_model import (
        TranslatePageViewModel,
    )
    vm = TranslatePageViewModel(bootstrap_for_gui())
    # Cố ý đưa LOCALIZE trước LITERAL để kiểm sắp xếp.
    vm._stage_outputs = {
        TranslationStageKind.LOCALIZE: [_ev("Chào nhé")],
        TranslationStageKind.LITERAL: [_ev("Xin chào")],
        TranslationStageKind.STYLE: [_ev("Chào bạn")],
    }
    comp = vm.stage_comparison()
    names = [name for name, _ in comp]
    assert names == ["Dịch thô", "Tinh chỉnh", "Bản địa hoá"]
    assert comp[0][1] == ["Xin chào"]
    assert comp[2][1] == ["Chào nhé"]


def test_comparison_empty_when_no_stages(app) -> None:
    from subtitles_extractor.presentation.view_models.translate_page_view_model import (
        TranslatePageViewModel,
    )
    vm = TranslatePageViewModel(bootstrap_for_gui())
    assert vm.stage_comparison() == []
