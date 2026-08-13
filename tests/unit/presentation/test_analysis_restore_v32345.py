"""Test [v3.23.45] tự nạp lại phân tích ngữ cảnh đã lưu khi chọn lại video."""

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


def _events():
    return [
        SubtitleEvent(index=1, text="你好", interval=TimeInterval(0.0, 1.0)),
        SubtitleEvent(index=2, text="再见", interval=TimeInterval(1.0, 2.0)),
    ]


def test_restore_after_save(app) -> None:
    from subtitles_extractor.presentation.view_models.translate_page_view_model import (
        TranslatePageViewModel,
    )
    from subtitles_extractor.application.services.translation_session_hashing import (
        hash_analysis_input,
    )
    c = bootstrap_for_gui()
    vm = TranslatePageViewModel(c)
    vm.set_source_events(_events())

    video = "/fake/movie_restore_test.mp4"
    lang = "Vietnamese"
    src_lines = [e.text for e in _events()]
    input_hash = hash_analysis_input(src_lines, lang)
    # Lưu phân tích vào session store (giả lập đã phân tích trước đó).
    c.translation_session_store.save_analysis(
        video, characters="林恒 (Lâm Hằng)", overview="Tiên hiệp",
        source_lang="zh", target_lang=lang, input_hash=input_hash,
        glossary="灵力 => linh lực", visual_cues='[{"id":1,"spk":"Lâm Hằng"}]',
    )

    received = []
    vm.analysis_restored.connect(lambda r: received.append(r))
    ok = vm.try_restore_saved_analysis(video, lang)
    assert ok is True
    assert len(received) == 1
    assert received[0].characters == "林恒 (Lâm Hằng)"
    assert received[0].glossary == "灵力 => linh lực"
    assert received[0].visual_cues == '[{"id":1,"spk":"Lâm Hằng"}]'


def test_no_restore_when_input_changed(app) -> None:
    from subtitles_extractor.presentation.view_models.translate_page_view_model import (
        TranslatePageViewModel,
    )
    c = bootstrap_for_gui()
    vm = TranslatePageViewModel(c)
    vm.set_source_events(_events())
    # Lưu với hash của đầu vào KHÁC → không khớp → không khôi phục.
    c.translation_session_store.save_analysis(
        "/fake/other.mp4", characters="X", overview="Y",
        source_lang="zh", target_lang="Vietnamese", input_hash="HASH_KHAC",
    )
    assert vm.try_restore_saved_analysis("/fake/other.mp4", "Vietnamese") is False


def test_no_restore_without_video(app) -> None:
    from subtitles_extractor.presentation.view_models.translate_page_view_model import (
        TranslatePageViewModel,
    )
    c = bootstrap_for_gui()
    vm = TranslatePageViewModel(c)
    vm.set_source_events(_events())
    assert vm.try_restore_saved_analysis("", "Vietnamese") is False
