"""[v3.23.609] Rà lần hai trang Dịch: các lỗi còn lại.

* Bản dịch mở lại phiên THIẾU dòng (phụ đề có thêm dòng sau khi dịch) → bảng ghép câu gốc
  với bản dịch theo VỊ TRÍ nên từ dòng thiếu trở đi lệch hết, "Dịch lại các dòng đã
  chọn" ghi kết quả vào dòng khác.
* Trang Dịch và trang Biên tập/TTS dùng CHUNG đối tượng phụ đề: trang Biên tập đánh số
  lại (xoá một dòng) là đổi luôn chỉ số câu gốc của trang Dịch; sửa một dòng ở trang Dịch
  đổi luôn dòng đã gửi sang trang Biên tập mà trang đó không hay biết (không đánh dấu đã
  sửa, hoàn tác sai).
* Dòng sửa tay / dịch lại trên BẢNG KẾT QUẢ bị mất khi bấm Dịch lại: từ 3.23.608 dịch lại
  chỉ mất vài giây (dùng bộ nhớ đệm) nên đây là thao tác thường gặp — sửa một giai đoạn
  rồi bấm Dịch là mất mọi dòng đã sửa trên bảng, không báo gì.
* Bấm "Tải model": model đang chọn (vd gõ tay) không có trong danh sách trả về thì bị
  đổi ngầm sang model đầu danh sách — đổi luôn hạn mức và bộ nhớ đệm.
* Hộp thoại hạn mức: nhập sai một ô là mất hết mọi dòng vừa nhập (hộp thoại đã đóng).
* Dịch cả bộ: giữa hai tập vẫn bấm được Dịch / Nạp phụ đề (kết quả ghi vào tệp của tập
  đang chạy), còn nút Huỷ thì bị khoá; huỷ đúng lúc chuyển tập vẫn dịch tiếp tập kế.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from subtitles_extractor.application.services.subtitle_editor_service import (
    SubtitleEditorService,
)
from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesResponse,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationStageKind,
)
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

LITERAL = TranslationStageKind.LITERAL


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _events(prefix: str, count: int = 6) -> list[SubtitleEvent]:
    return [
        SubtitleEvent(index=i + 1, text=f"{prefix}{i}", interval=TimeInterval(i, i + 0.9))
        for i in range(count)
    ]


def _container(tmp: Path):
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import (
        JsonFileRepository,
    )
    from subtitles_extractor.presentation.pages import translate_page

    return ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp / "c.json"),
        i18n_data_dir=Path(translate_page.__file__).resolve().parents[2] / "data",
        user_data_dir=tmp,
    )


def _view_model(tmp: Path):
    from subtitles_extractor.presentation.view_models.translate_page_view_model import (
        TranslatePageViewModel,
    )

    return TranslatePageViewModel(_container(tmp))


def _page(tmp: Path):
    from subtitles_extractor.presentation.pages import translate_page

    page = translate_page.TranslatePage(_container(tmp))
    page._persist_inputs = lambda: None  # không ghi QSettings thật của máy
    page._show_info = page._show_success = page._show_warning = lambda *_a: None
    return page


def _finish(view_model, texts: list[str], target: str = "Vietnamese") -> None:
    """Giả một lượt dịch vừa xong với kết quả ``texts`` (theo thứ tự câu gốc)."""
    view_model._last_translate_context = TranslationContext(target_lang=target)
    source = view_model.source_events
    events = [
        SubtitleEvent(index=ev.index, text=text, interval=ev.interval)
        for ev, text in zip(source, texts, strict=True)
    ]
    view_model._on_worker_finished(
        TranslateSubtitlesResponse(events=events, completed_stages=[LITERAL])
    )


class TestRestoredTranslationStaysAligned:
    def test_missing_lines_do_not_shift_the_rest(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        view_model = _view_model(tmp_path)
        video = str(tmp_path / "phim.mp4")
        view_model.set_source_events(_events("句", 5))
        view_model._last_translate_video_path = video
        _finish(view_model, [f"dịch {i}" for i in range(5)])
        # Phụ đề được bổ sung một dòng ở GIỮA (OCR sót) sau khi dịch.
        source = _events("句", 5)
        inserted = SubtitleEvent(index=3, text="新", interval=TimeInterval(2.5, 2.9))
        renumbered = [source[0], source[1], inserted, *source[2:]]
        renumbered = [
            SubtitleEvent(index=pos, text=ev.text, interval=ev.interval)
            for pos, ev in enumerate(renumbered, start=1)
        ]
        view_model.set_source_events(renumbered)
        view_model.restore_stage_outputs_for_video(video)
        translated = view_model.translated_events
        if translated:
            assert [ev.index for ev in translated] == [ev.index for ev in renumbered]

    def test_lines_added_at_the_end_keep_their_rows(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        view_model = _view_model(tmp_path)
        video = str(tmp_path / "phim.mp4")
        view_model.set_source_events(_events("句", 20))
        view_model._last_translate_video_path = video
        _finish(view_model, [f"dịch {i}" for i in range(20)])
        view_model.set_source_events(_events("句", 21))  # thêm một dòng cuối
        view_model.restore_stage_outputs_for_video(video)
        translated = view_model.translated_events
        assert [ev.index for ev in translated] == list(range(1, 22))
        assert translated[0].text == "dịch 0"
        assert translated[20].text == ""


class TestEventsAreNotSharedWithOtherPages:
    def test_editor_renumbering_does_not_change_translate_source(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        view_model = _view_model(tmp_path)
        editor = SubtitleEditorService()
        editor.load(_events("句", 4))
        view_model.set_source_events(editor.snapshot_state().events)  # "Lấy từ Biên tập"
        editor.delete(0)
        assert [ev.index for ev in view_model.source_events] == [1, 2, 3, 4]

    def test_editing_after_sending_to_editor_does_not_touch_the_editor_copy(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        view_model = _view_model(tmp_path)
        view_model.set_source_events(_events("句", 3))
        _finish(view_model, ["a", "b", "c"])
        sent: list[list[SubtitleEvent]] = []
        view_model.set_push_to_editor_callback(sent.append)
        assert view_model.push_to_editor()
        view_model.update_translation_text(1, "B đã sửa")
        assert [ev.text for ev in sent[0]] == ["a", "b", "c"]

    def test_returned_lists_cannot_change_the_view_model(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        view_model = _view_model(tmp_path)
        view_model.set_source_events(_events("句", 2))
        _finish(view_model, ["a", "b"])
        view_model.translated_events[0].text = "đổi ngầm"
        view_model.source_events[0].index = 99
        assert view_model.translated_events[0].text == "a"
        assert view_model.source_events[0].index == 1


class TestManualEditsSurviveRetranslation:
    def test_edit_on_the_result_table_is_kept_after_translating_again(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        view_model = _view_model(tmp_path)
        video = str(tmp_path / "phim.mp4")
        view_model.set_source_events(_events("句", 3))
        view_model._last_translate_video_path = video
        _finish(view_model, ["a", "b", "c"])
        assert view_model.update_translation_text(1, "B chốt", target_lang="Vietnamese")
        _finish(view_model, ["a", "b", "c"])  # bấm Dịch lại (sau khi sửa một giai đoạn)
        texts = [ev.text for ev in view_model.translated_events]
        assert texts == ["a", "B chốt", "c"]
        assert view_model.final_edit_positions() == {1}

    def test_edits_are_restored_with_the_session_and_kept_on_the_next_run(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        video = str(tmp_path / "phim.mp4")
        first = _view_model(tmp_path)
        first.set_source_events(_events("句", 3))
        first._last_translate_video_path = video
        _finish(first, ["a", "b", "c"])
        first.update_translation_text(2, "C chốt", target_lang="Vietnamese")

        reopened = _view_model(tmp_path)
        reopened.set_source_events(_events("句", 3))
        reopened.restore_stage_outputs_for_video(video)
        assert reopened.final_edit_positions() == {2}
        _finish(reopened, ["a2", "b2", "c2"])
        assert [ev.text for ev in reopened.translated_events] == ["a2", "b2", "C chốt"]

    def test_edit_for_another_language_is_not_applied(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        view_model = _view_model(tmp_path)
        view_model.set_source_events(_events("句", 2))
        _finish(view_model, ["a", "b"])
        view_model.update_translation_text(0, "A chốt", target_lang="Vietnamese")
        _finish(view_model, ["hello", "world"], target="English")
        assert [ev.text for ev in view_model.translated_events] == ["hello", "world"]

    def test_discarding_an_edit_returns_the_latest_machine_text(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        view_model = _view_model(tmp_path)
        view_model.set_source_events(_events("句", 2))
        _finish(view_model, ["a", "b"])
        view_model.update_translation_text(0, "A chốt", target_lang="Vietnamese")
        _finish(view_model, ["a mới", "b"])
        assert view_model.discard_final_edits([0]) == 1
        assert view_model.translated_events[0].text == "a mới"
        assert view_model.final_edit_positions() == set()

    def test_retranslated_lines_are_kept_too(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        view_model = _view_model(tmp_path)
        view_model.set_source_events(_events("句", 3))
        _finish(view_model, ["a", "b", "c"])
        view_model._retranslate_indices = [2]
        view_model._retranslate_target_lang = "Vietnamese"
        view_model._on_retranslate_done(
            TranslateSubtitlesResponse(
                events=[SubtitleEvent(index=1, text="c tốt hơn",
                                      interval=TimeInterval(2, 2.9))],
                completed_stages=[LITERAL],
            )
        )
        _finish(view_model, ["a", "b", "c"])
        assert view_model.translated_events[2].text == "c tốt hơn"


class TestPageControls:
    def test_loading_models_keeps_a_model_missing_from_the_list(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        page = _page(tmp_path)
        page._stage_literal.model_combo.setCurrentText("gemini-3.8-flash-preview-0915")
        page._analyze_model_combo.setCurrentText("gemini-3.8-flash")
        page._on_models_ready(["gemini-3.1-flash-lite", "gemini-3.5-flash"])
        assert (
            page._stage_literal.build_config().model_name
            == "gemini-3.8-flash-preview-0915"
        )
        assert page._analyze_model_combo.currentText() == "gemini-3.8-flash"
        page.close()

    def test_quota_dialog_keeps_the_rows_after_a_typo(
        self, app: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        page = _page(tmp_path)
        saved: list[dict] = []
        page._view_model.save_quota_config = saved.append
        opened: list[int] = []

        def _exec(dialog) -> int:
            from PySide6.QtWidgets import QTableWidget

            table = dialog.findChild(QTableWidget)
            opened.append(table.rowCount())
            if len(opened) == 1:
                table.item(0, 0).setText("gemini-3.8-flash")
                table.item(0, 1).setText("mười")  # gõ nhầm
                return QDialog.DialogCode.Accepted
            assert table.item(0, 0).text() == "gemini-3.8-flash"
            table.item(0, 1).setText("10")
            return QDialog.DialogCode.Accepted

        monkeypatch.setattr(QDialog, "exec", _exec)
        page._on_edit_quota_clicked()
        assert len(opened) == 2
        assert saved and saved[0]["gemini-3.8-flash"]["rpm"] == 10
        page.close()

    def test_batch_queue_locks_manual_actions_and_allows_cancel(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        page = _page(tmp_path)
        page._view_model.set_source_events(_events("句", 3))
        page._batch_items = [object(), object()]
        page._update_action_states()
        assert not page._btn_run.isEnabled()
        assert not page._btn_load_file.isEnabled()
        assert not page._btn_pull_editor.isEnabled()
        assert page._btn_cancel.isEnabled()
        page._batch_items = []
        page._update_action_states()
        assert page._btn_run.isEnabled()
        page.close()

    def test_cancel_between_episodes_does_not_start_the_next_one(
        self, app: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        page = _page(tmp_path)
        started: list[bool] = []
        page._start_translation_from_ui = lambda: started.append(True) or True
        monkeypatch.setattr(QMessageBox, "information", lambda *_a, **_k: None)
        page._batch_items = [object(), object()]
        page._on_cancel_clicked()  # bấm Huỷ lúc đang chuyển tập (không có lượt chạy)
        page._start_batch_episode()
        assert started == []
        assert page._batch_items == []
        page.close()


def test_final_edits_are_saved_in_the_session(tmp_path: Path, app: QApplication) -> None:
    view_model = _view_model(tmp_path)
    video = str(tmp_path / "phim.mp4")
    view_model.set_source_events(_events("句", 2))
    view_model._last_translate_video_path = video
    _finish(view_model, ["a", "b"])
    view_model.update_translation_text(1, "B chốt", target_lang="Vietnamese")
    session = view_model._container.translation_session_store.get(video)
    rows = json.loads(session.stage("final_edits").lines_json)
    assert rows[0]["index"] == 2 and rows[0]["text"] == "B chốt"
    assert rows[0]["auto"] == "b"


def test_edited_rows_are_marked_and_can_be_discarded(
    app: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from PySide6.QtWidgets import QInputDialog

    page = _page(tmp_path)
    page._view_model.set_source_events(_events("句", 3))
    _finish(page._view_model, ["a", "b", "c"])
    monkeypatch.setattr(
        QInputDialog, "getMultiLineText", lambda *_a, **_k: ("B chốt", True)
    )
    page._on_edit_translation_row(1)
    assert page._table.item(1, 0).text() == "2 ✏"
    assert "Bỏ sửa tay" in page._table.item(1, 3).toolTip()
    assert page._view_model.discard_final_edits([1]) == 1
    assert page._table.item(1, 0).text() == "2"
    assert page._table.item(1, 3).text() == "b"
    page.close()
