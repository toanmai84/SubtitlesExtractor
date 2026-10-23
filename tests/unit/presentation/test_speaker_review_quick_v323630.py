"""[v3.23.630] Hộp thoại "Duyệt nhanh người nói" + nút/menu trên bảng dòng lồng tiếng.

Cảnh thật dòng 490-499 phim "灭门惨案II": dòng 495 nhãn vợ nhưng câu gốc gọi chính cô là "阿秀"
(người nói là chồng). Trước bản này sửa dòng đó = chuột phải → menu 24 vai → chọn, lặp hàng
trăm lần (log thật: 861/3649 dòng sửa tay).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from subtitles_extractor.application.services.speaker_review_flow import ReviewLine
from subtitles_extractor.application.services.speaker_review_session import (
    SpeakerReviewSession,
)
from subtitles_extractor.presentation.pages import tts_line_edits as line_edits_module
from subtitles_extractor.presentation.widgets.speaker_review_dialog import (
    SpeakerReviewDialog,
)
from subtitles_extractor.application.services.dubbing_cast import DubbingSource
from tests.unit.presentation.test_dubbing_line_edits_v323598 import (  # noqa: F401
    HUSBAND,
    LINES,
    P495,
    WIFE,
    _events,
    _row_of,
    _source,
    _trigger,
    app,
    page,
)


def _source_with_texts() -> DubbingSource:
    """Như ``_source`` nhưng có CÂU GỐC — cần để phát hiện "gọi tên chính người nói"."""
    base = _source()
    return DubbingSource(
        base.characters, base.visual_cues, base.source_spans,
        tuple(row[6] for row in LINES),
    )

TEXTS = {
    key: key for key in line_edits_module.REVIEW_TEXT_KEYS
}
TEXTS.update({"progress": "{k}/{n}", "run": "run+{n}", "pending": "đổi {n}"})


def _review_lines() -> list[ReviewLine]:
    speakers = [row[4] for row in LINES]
    return [
        ReviewLine(i, row[1], row[2], row[7], row[6], speakers[i], row[5])
        for i, row in enumerate(LINES)
    ]


def _dialog(queue=(P495, 7), **kw) -> SpeakerReviewDialog:
    session = SpeakerReviewSession(_review_lines(), queue)
    return SpeakerReviewDialog(
        None, session, TEXTS, lambda name: name,
        [(WIFE, WIFE), (HUSBAND, HUSBAND), ("Bác sĩ", "醫生")],
        kw.get("reason_of", lambda position: "gọi tên" if position == P495 else ""),
    )


class TestDialog:
    def test_shows_the_scene_around_the_current_line(self, app: QApplication) -> None:
        dialog = _dialog()
        assert dialog.progress_label.text() == "1/2"
        assert dialog.table.rowCount() == 7                    # ±3 dòng
        assert dialog.table.item(3, 0).text() == str(P495 + 1)  # dòng đang duyệt ở giữa
        assert dialog.table.item(3, 2).text() == WIFE
        assert dialog.reason_label.text() == "gọi tên" and not dialog.reason_label.isHidden()
        labels = [button.text() for button in dialog.candidate_buttons]
        assert len(labels) == 2 and {WIFE, HUSBAND} == {t.split(" · ")[1].lstrip("✓ ") for t in labels}

    def test_candidate_button_picks_speaker_and_moves_on(self, app: QApplication) -> None:
        dialog = _dialog()
        index = next(
            i for i, name in enumerate(dialog._session.candidates()) if name == HUSBAND
        )
        assert dialog.choose_candidate(index) == [P495]
        assert dialog.result() == {P495: HUSBAND}
        assert dialog.progress_label.text() == "2/2"
        assert dialog.pending_label.text() == "đổi 1"
        assert dialog.undo_button.isEnabled()

    def test_keyboard_digits_space_arrows_and_undo(self, app: QApplication) -> None:
        dialog = _dialog()
        dialog.show()
        dialog.activateWindow()
        names = dialog._session.candidates()
        digit = Qt.Key(Qt.Key.Key_1 + names.index(HUSBAND))
        QTest.keyClick(dialog, digit)
        assert dialog.result() == {P495: HUSBAND}
        QTest.keyClick(dialog, Qt.Key.Key_Space)             # giữ nguyên dòng 7
        assert dialog._session.finished
        QTest.keyClick(dialog, Qt.Key.Key_Left)              # về dòng 7
        assert dialog._session.position == 7
        QTest.keyClick(dialog, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        assert dialog.result() == {} and dialog._session.position == P495
        QTest.keyClick(dialog, Qt.Key.Key_0)                 # giọng chung
        assert dialog.result() == {P495: ""}
        dialog.close()

    def test_whole_run_checkbox(self, app: QApplication) -> None:
        dialog = _dialog(queue=(2, P495))
        assert dialog.run_box.isEnabled() and dialog.run_box.text() == "run+1"
        dialog.run_box.setChecked(True)
        dialog.choose_speaker(WIFE)
        assert dialog.result() == {2: WIFE, 3: WIFE}

    def test_run_checkbox_disabled_for_a_lone_line(self, app: QApplication) -> None:
        dialog = _dialog(queue=(0,))
        assert not dialog.run_box.isEnabled()
        dialog.run_box.setChecked(True)
        assert dialog.choose_speaker(WIFE) == [0]

    def test_other_role_combo(self, app: QApplication) -> None:
        dialog = _dialog()
        dialog._on_other(3)                                  # "Bác sĩ"
        assert dialog.result() == {P495: "醫生"}
        assert dialog.other_box.currentIndex() == 0

    def test_all_lines_scope_and_finish_state(self, app: QApplication) -> None:
        dialog = _dialog(queue=(P495,))
        dialog.all_lines_box.setChecked(True)
        assert dialog.progress_label.text() == f"{P495 + 1}/{len(LINES)}"
        dialog.keep()
        dialog.all_lines_box.setChecked(False)
        assert dialog._session.finished
        assert not dialog.keep_button.isEnabled() and dialog.done_button.isEnabled()
        dialog.previous()
        assert dialog._session.position == P495

    def test_changed_speaker_is_marked_in_the_context_table(self, app: QApplication) -> None:
        dialog = _dialog(queue=(P495, 7))
        dialog.choose_speaker(HUSBAND)
        dialog.previous()
        row = next(
            r for r in range(dialog.table.rowCount())
            if dialog.table.item(r, 0).text() == str(P495 + 1)
        )
        assert dialog.table.item(row, 2).text() == HUSBAND
        assert dialog.table.item(row, 2).toolTip() == "changed_tip"


class TestController:
    def _open(self, page, monkeypatch, answer):
        page.set_dubbing_source_provider(_source_with_texts)
        page._dubbing_panel.set_dubbing_enabled(True)
        page._view_model.set_source_events(_events())
        seen: dict[str, object] = {}

        def fake(parent, session, texts, display_name, roles, reason_of):
            seen.update(
                session=session, texts=texts, roles=roles,
                reason=reason_of(P495), position=session.position,
                total=session.total,
            )
            return answer

        monkeypatch.setattr(line_edits_module, "ask_speaker_review", fake)
        return page._line_edits, seen

    def test_button_opens_the_flagged_queue_and_applies_the_answer(
        self, page, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        controller, seen = self._open(page, monkeypatch, {P495: HUSBAND})
        assert controller._btn_quick_review.isEnabled()
        assert controller.quick_review() == 1
        assert seen["position"] == P495 and seen["total"] == 1
        assert "阿秀" in seen["reason"] and "Tần Khả Tú" in seen["reason"]   # lý do hiện trên hộp thoại
        assert {key for key in line_edits_module.REVIEW_TEXT_KEYS} == set(seen["texts"])
        assert all(seen["texts"].values())                    # mọi chuỗi đều có trong strings
        assert page._view_model.speaker_overrides == {P495: HUSBAND}
        assert page._view_model.effective_line_speakers()[P495] == HUSBAND

    def test_cancel_changes_nothing(self, page, monkeypatch: pytest.MonkeyPatch) -> None:
        controller, _seen = self._open(page, monkeypatch, None)
        assert controller.quick_review() == 0
        assert page._view_model.speaker_overrides == {}
        controller, _seen = self._open(page, monkeypatch, {})
        assert controller.quick_review() == 0

    def test_applies_one_batch_per_speaker(self, page, monkeypatch: pytest.MonkeyPatch) -> None:
        controller, _seen = self._open(page, monkeypatch, {2: WIFE, 3: WIFE, P495: HUSBAND})
        assert controller.quick_review() == 3
        assert page._view_model.speaker_overrides == {2: WIFE, 3: WIFE, P495: HUSBAND}

    def test_start_position_reviews_every_line_from_there(
        self, page, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        controller, seen = self._open(page, monkeypatch, None)
        controller.quick_review(start_position=8)
        assert seen["position"] == 8 and seen["session"].all_lines

    def test_no_flagged_lines_falls_back_to_all_lines(
        self, page, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        controller, seen = self._open(page, monkeypatch, None)
        controller.apply_speaker([P495], HUSBAND)           # hết dòng nghi sai
        controller.quick_review()
        assert seen["session"].all_lines and seen["position"] == 0

    def test_without_a_cast_it_says_so(self, page, monkeypatch: pytest.MonkeyPatch) -> None:
        page.set_dubbing_source_provider(lambda: line_edits_module.__dict__.get("nope") or _empty())
        page._view_model.set_source_events(_events())
        called: list[int] = []
        monkeypatch.setattr(
            line_edits_module, "ask_speaker_review",
            lambda *a, **k: called.append(1) or None,
        )
        assert page._line_edits.quick_review() == 0 and not called

    def test_select_run_selects_the_adjacent_same_speaker_lines(
        self, page, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        controller, _seen = self._open(page, monkeypatch, None)
        assert controller.select_run(2) == 2                 # dòng 2-3 cùng chồng
        assert page._selected_event_indices() == [2, 3]

    def test_context_menu_has_the_new_entries(self, page, monkeypatch: pytest.MonkeyPatch) -> None:
        controller, _seen = self._open(page, monkeypatch, {P495: HUSBAND})
        menu = controller.build_context_menu([P495])
        texts = [action.text() for action in menu.actions()]
        assert any(t.startswith("Duyệt nhanh người nói từ dòng này") for t in texts)
        assert any(t.startswith("Chọn cả đoạn liền kề") for t in texts)
        _trigger(menu, "Duyệt nhanh người nói từ dòng này")
        assert page._view_model.speaker_overrides == {P495: HUSBAND}


def _empty():
    from subtitles_extractor.application.services.dubbing_cast import DubbingSource

    return DubbingSource()
