"""[v3.23.653] Đồng bộ người nói chọn tay ở trang TTS về Visual Cues của trang Dịch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from subtitles_extractor.application.services.dubbing_line_edits import SHARED_VOICE_SPEAKER
from subtitles_extractor.application.services.visual_cue_serializer import deserialize_visual_cues
from subtitles_extractor.composition.bootstrap import bootstrap_for_gui
from subtitles_extractor.domain.ports.subtitle_translator_port import CUE_ORIGIN_USER
from subtitles_extractor.presentation.view_models.tts_page_view_model import TTSPageViewModel
from tests.unit.presentation.test_dubbing_line_edits_v323598 import (  # noqa: F401
    HUSBAND,
    P495,
    P496,
    WIFE,
    _events,
    _source,
    app,
    page,
)


def _prepared_vm() -> TTSPageViewModel:
    vm = TTSPageViewModel(SimpleNamespace())
    vm.set_source_events(_events())
    vm.prepare_dubbing(_source())
    return vm


def test_corrections_map_tts_lines_to_source_ids(app: QApplication) -> None:  # noqa: F811
    vm = _prepared_vm()
    vm.set_speaker_override([P495], HUSBAND)
    vm.set_speaker_override([P496], SHARED_VOICE_SPEAKER)   # "giọng chung" không phải một người nói
    assert vm.speaker_corrections() == {P495 + 1: HUSBAND}
    assert vm.dubbing_source_spans == _source().source_spans


def test_no_corrections_before_cast_is_loaded(app: QApplication) -> None:  # noqa: F811
    vm = TTSPageViewModel(SimpleNamespace())
    vm.set_source_events(_events())
    vm.set_speaker_override([P495], HUSBAND)
    assert vm.speaker_corrections() == {}


def test_translate_page_applies_and_saves(app: QApplication) -> None:  # noqa: F811
    from subtitles_extractor.presentation.pages.translate_page import TranslatePage

    page = TranslatePage(bootstrap_for_gui())
    saved: list = []
    page._view_model.save_context_edits = lambda path, **k: saved.append((path, k))
    page._source_video_path = "/tmp/ep1.mp4"
    page._visual_cues_edit.setPlainText(_source().visual_cues)
    assert page.apply_user_speakers({P495 + 1: HUSBAND}) == 1
    cues = {c.line_no: c for c in deserialize_visual_cues(page._visual_cues_edit.toPlainText())}
    assert cues[P495 + 1].speaker == HUSBAND and cues[P495 + 1].origin == CUE_ORIGIN_USER
    assert saved and saved[-1][0] == "/tmp/ep1.mp4"
    assert page.apply_user_speakers({P495 + 1: HUSBAND}) == 0

    page._visual_cues_edit.setPlainText("[{")
    assert page.apply_user_speakers({1: WIFE}) == -1
    page._visual_cues_edit.setPlainText(_source().visual_cues)
    page._view_model._is_busy = True
    try:
        assert page.apply_user_speakers({1: HUSBAND}) == -2
    finally:
        page._view_model._is_busy = False


def test_reanalysis_result_keeps_user_speaker(app: QApplication) -> None:  # noqa: F811
    from subtitles_extractor.presentation.pages.translate_page import TranslatePage

    page = TranslatePage(bootstrap_for_gui())
    page._view_model.save_context_edits = lambda *_a, **_k: None
    page._source_video_path = "/tmp/ep1.mp4"
    page._visual_cues_edit.setPlainText(_source().visual_cues)
    page.apply_user_speakers({P495 + 1: HUSBAND})
    page._on_visual_cues_topped_up(_source().visual_cues)       # bản AI cũ quay lại sau lượt quét bổ sung
    cues = {c.line_no: c for c in deserialize_visual_cues(page._visual_cues_edit.toPlainText())}
    assert cues[P495 + 1].speaker == HUSBAND


def test_main_window_refuses_when_translate_page_has_another_subtitle(app: QApplication) -> None:  # noqa: F811
    from subtitles_extractor.presentation.main_window import MainWindow

    results: list[int] = []
    applied: list[dict] = []
    spans = _source().source_spans
    events = [SimpleNamespace(start_sec=s, end_sec=e) for s, e in spans]
    fake = SimpleNamespace(
        _translate_page=SimpleNamespace(
            _view_model=SimpleNamespace(source_events=events),
            apply_user_cues=lambda c: applied.append(c) or len(c),
        ),
        _tts_page=SimpleNamespace(
            _view_model=SimpleNamespace(dubbing_source_spans=spans),
            show_speaker_sync_result=results.append,
        ),
    )
    MainWindow._sync_tts_speakers(fake, {6: HUSBAND})
    assert applied == [{6: (HUSBAND, None)}] and results == [1]

    fake._tts_page._view_model.dubbing_source_spans = spans[:-1]   # trang Dịch đã đổi phụ đề
    MainWindow._sync_tts_speakers(fake, {6: HUSBAND})
    assert results[-1] == -3 and len(applied) == 1


def test_tts_page_emits_corrections_once(page) -> None:  # noqa: F811
    received: list[dict] = []
    page.speaker_corrections_ready.connect(received.append)
    page._view_model.set_source_events(_events())
    page._view_model.prepare_dubbing(_source())
    page._view_model.set_speaker_override([P495], HUSBAND)
    assert page._speaker_sync_timer.isActive()        # gom lần sửa, không gửi ngay từng phím
    page._emit_speaker_corrections()
    page._emit_speaker_corrections()                  # không đổi gì: không gửi lại
    assert received == [{P495 + 1: (HUSBAND, None)}]
    page.show_speaker_sync_result(-2)                 # trang Dịch bận → lần sau gửi lại
    page._emit_speaker_corrections()
    assert len(received) == 2


def test_choosing_auto_again_releases_user_cue_v323655(page) -> None:  # noqa: F811
    """[v3.23.655] Chọn lại "Tự động" ở trang TTS → cue người dùng ở trang Dịch trả về nhãn AI, bỏ dấu người dùng."""
    from subtitles_extractor.application.services.visual_cue_serializer import CUE_AUTO, apply_user_cues

    received: list[dict] = []
    page.speaker_corrections_ready.connect(received.append)
    page._view_model.set_source_events(_events())
    page._view_model.prepare_dubbing(_source())
    page._view_model.set_speaker_override([P495], HUSBAND)
    page._emit_speaker_corrections()
    page.show_speaker_sync_result(1)                  # trang Dịch đã ghi
    page._view_model.set_speaker_override([P495], None)   # chọn lại "Tự động"
    page._emit_speaker_corrections()
    assert received[-1] == {P495 + 1: (CUE_AUTO, None)}

    ai_raw = _source().visual_cues
    user_raw, _ = apply_user_cues(ai_raw, received[0])
    back_raw, changed = apply_user_cues(user_raw, received[-1])
    before = {c.line_no: c for c in deserialize_visual_cues(ai_raw)}
    after = {c.line_no: c for c in deserialize_visual_cues(back_raw)}
    assert changed == [P495 + 1]
    assert after[P495 + 1].speaker == before[P495 + 1].speaker and after[P495 + 1].origin == ""
    page.show_speaker_sync_result(1)
    page._emit_speaker_corrections()                  # đã trả về: không gửi lại
    assert len(received) == 2
