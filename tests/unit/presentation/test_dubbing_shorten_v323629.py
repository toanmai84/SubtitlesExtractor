"""[v3.23.629] Trang TTS: nút "Rút gọn dòng quá dài (AI)" (xem trước → áp dụng → TTS lại)."""  # noqa: RUF002

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from subtitles_extractor.application.use_cases.shorten_for_speech import ShortenedLine
from tests.unit.presentation.test_dubbing_line_edits_v323598 import (  # noqa: F401
    LINES,
    _events,
    app,
    page,
)


def _tight_events():
    """Hai dòng 6 âm tiết trong 0,5 s sát câu sau → bị dự đoán phải nén > 1,8×."""
    from dataclasses import replace

    from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

    events = _events()
    events[2] = replace(events[2], text="Nghịch đồ vô liêm sỉ lắm đấy",
                        interval=TimeInterval(events[2].start_sec, events[2].start_sec + 0.5))
    events[3] = replace(events[3], interval=TimeInterval(events[2].start_sec + 0.55,
                                                          events[2].start_sec + 3.0))
    return events


class TestShortenButton:
    def test_button_counts_misfits_and_needs_a_provider(self, page) -> None:  # noqa: F811
        controller = page._line_edits
        page._view_model.set_source_events(_events())
        controller._update_buttons()
        assert not controller._btn_shorten.isEnabled()
        page._view_model.set_source_events(_tight_events())
        controller._update_buttons()
        assert controller._btn_shorten.isEnabled() and "(1)" in controller._btn_shorten.text()

    def test_without_an_api_key_the_user_is_told_why(self, page) -> None:  # noqa: F811
        page._view_model.set_source_events(_tight_events())
        page.set_shorten_provider(lambda: ("", "m", "Vietnamese"))
        assert page._line_edits.shorten_lines() == 1
        assert "khoá API Gemini" in page._stage_lbl.text()

    def test_preview_then_apply_edits_the_lines(self, page, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
        from subtitles_extractor.presentation.pages import tts_line_edits

        page._view_model.set_source_events(_tight_events())
        original = page._view_model.source_events[2].text
        proposals = [ShortenedLine(2, original, "Nghịch đồ vô liêm sỉ", 7, 5)]
        seen: dict = {}

        def fake_preview(_parent, _title, _intro, headers, lines):
            seen.update(headers=headers, lines=list(lines))
            return list(lines)

        monkeypatch.setattr(tts_line_edits, "ask_shorten_preview", fake_preview)
        assert page._line_edits.apply_shortened(proposals) == 1
        assert seen["lines"] == proposals and len(seen["headers"]) == 4
        assert page._view_model.source_events[2].text == "Nghịch đồ vô liêm sỉ"
        assert page._view_model.line_edits().texts[2].original == original      # "Bỏ sửa tay" hoàn lại
        assert page._table.item(2, 2).text() == "Nghịch đồ vô liêm sỉ"
        assert 2 in page._view_model.edited_positions
        assert page._view_model.can_undo and page._view_model.undo()
        assert page._view_model.source_events[2].text == original

    def test_declining_the_preview_changes_nothing(self, page, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
        from subtitles_extractor.presentation.pages import tts_line_edits

        page._view_model.set_source_events(_tight_events())
        monkeypatch.setattr(tts_line_edits, "ask_shorten_preview", lambda *_a: [])
        before = page._view_model.source_events[2].text
        assert page._line_edits.apply_shortened([ShortenedLine(2, before, "x y", 7, 2)]) == 0
        assert page._view_model.source_events[2].text == before
        assert page._line_edits.apply_shortened([]) == 0
        assert "không đề xuất" in page._stage_lbl.text() or "giữ nguyên" in page._stage_lbl.text()

    def test_dialog_returns_only_ticked_rows(self, app) -> None:  # noqa: F811
        from PySide6.QtCore import Qt

        from subtitles_extractor.presentation.widgets.shorten_preview_dialog import (
            ShortenPreviewDialog,
        )

        lines = [ShortenedLine(0, "a b c d e", "a b c", 5, 3), ShortenedLine(4, "f g h i j", "f g h", 5, 3)]
        dialog = ShortenPreviewDialog(None, "t", "i", ["#", "a", "b", "c"], lines)
        dialog.table.item(1, 0).setCheckState(Qt.CheckState.Unchecked)
        assert dialog.accepted_lines() == [lines[0]]


class TestLineContext:
    def test_source_text_speaker_and_addressee_follow_the_alignment(self, page) -> None:  # noqa: F811
        from dataclasses import replace

        from tests.unit.presentation.test_dubbing_line_edits_v323598 import _source

        base = _source()
        texts = tuple(row[6] for row in LINES)
        page.set_dubbing_source_provider(lambda: replace(base, source_texts=texts))
        page._dubbing_panel.set_dubbing_enabled(True)
        page._view_model.set_source_events(_events())
        page._dubbing.load_cast()
        source, speaker, addressee = page._view_model.line_context(5)
        assert source == "阿秀，你幫我，幫幫我.."  # noqa: RUF001
        assert speaker == "秦可秀" and addressee == "甄劍聲"
