"""[v3.23.608] Trang Dịch: "Quản lý giai đoạn" và nút xoá bộ nhớ đệm dịch.

Lỗi thật của nút "Xoá bộ nhớ đệm dịch" (≤ v3.23.607): trang chỉ gửi ngôn ngữ đích, thiếu
ngôn ngữ gốc (luôn có sau khi phân tích ngữ cảnh) → tính ra khoá KHÁC khoá lúc dịch →
bấm xoá, báo "đã xoá" mà file vẫn còn, lần dịch tới vẫn dùng lại bản cũ.
"""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationStageKind,
)
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

LITERAL = TranslationStageKind.LITERAL
STYLE = TranslationStageKind.STYLE
LINES = 6


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _events() -> list[SubtitleEvent]:
    return [
        SubtitleEvent(index=i, text=f"句{i}", interval=TimeInterval(i, i + 0.9))
        for i in range(1, LINES + 1)
    ]


def _page():
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import (
        JsonFileRepository,
    )
    from subtitles_extractor.presentation.pages import translate_page

    tmp = Path(tempfile.mkdtemp())
    container = ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp / "c.json"),
        i18n_data_dir=Path(translate_page.__file__).resolve().parents[2] / "data",
        user_data_dir=tmp,
    )
    page = translate_page.TranslatePage(container)
    for panel, enabled in (
        (page._stage_preprocess, False),
        (page._stage_literal, True),
        (page._stage_style, True),
        (page._stage_localize, False),
    ):
        panel.enable_check.setChecked(enabled)
    page._lang_combo.setCurrentText("Vietnamese")
    page._source_lang_edit.setText("Chinese")
    page._visual_cues_check.setChecked(False)
    page._view_model.set_source_events(_events())
    return page, container


def _translate_into_cache(page, container) -> None:
    """Một lượt dịch thật (bộ dịch giả) với ĐÚNG cấu hình trên trang."""
    from subtitles_extractor.application.use_cases.translate_subtitles import (
        TranslateSubtitlesUseCase,
    )
    from subtitles_extractor.infrastructure.translation import (
        gemini_translation_adapter,
    )

    def _batch(*, batch, is_literal=False, **_kwargs):
        if is_literal:
            return [replace(line, text=f"dịch {line.index}") for line in batch]
        return [replace(line, text=f"mượt {line.index}") for line in batch]

    adapter = gemini_translation_adapter.GeminiSubtitleTranslator(api_key="KEY")
    adapter._ensure_available = lambda: None  # type: ignore[method-assign]
    adapter._translate_single_batch = _batch  # type: ignore[method-assign]
    settings = page._stage_cache_settings()
    request = page._view_model._translation_request(
        settings["stages"],
        page._view_model._effective_context(settings["context"]),
        video_path=settings["video_path"],
        enable_visual_cues=bool(settings["enable_visual_cues"]),
    )
    TranslateSubtitlesUseCase(
        translator=adapter, checkpoint_dir=container.stage_checkpoint_dir
    ).execute(request)


def test_clear_cache_button_deletes_the_file_used_by_translation(
    app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    from subtitles_extractor.application.use_cases.translate_subtitles import (
        TranslateSubtitlesRequest,
        _compute_checkpoint_key,
        _StageCheckpoint,
    )

    page, container = _page()
    stages = page._collect_enabled_stages()
    request = TranslateSubtitlesRequest(
        events=_events(),
        stages=stages,
        context=TranslationContext(target_lang="Vietnamese", source_lang="Chinese"),
    )
    checkpoint_dir = container.user_data_dir / "translation_checkpoints"
    lines = [
        replace(line, text=f"dịch {line.index}") for line in _lines_from(request)
    ]
    _StageCheckpoint(checkpoint_dir, _compute_checkpoint_key(request)).save_stage(
        LITERAL, lines
    )
    monkeypatch.setattr(
        QMessageBox, "question", lambda *_a, **_k: QMessageBox.StandardButton.Yes
    )
    page._on_clear_cache_clicked()
    assert list(checkpoint_dir.glob("*.json")) == []
    page.close()


def _lines_from(request):
    from subtitles_extractor.application.use_cases.translate_subtitles import (
        TranslateSubtitlesUseCase,
    )

    return TranslateSubtitlesUseCase._events_to_lines(request.events)


def test_manage_stages_button_opens_the_dialog(
    app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    from subtitles_extractor.presentation.widgets import stage_cache_dialog

    page, container = _page()
    _translate_into_cache(page, container)
    page._update_action_states()
    assert page._btn_stage_cache.isEnabled()
    opened: list[object] = []

    def _fake_exec(dialog) -> int:
        opened.append(dialog._overview)
        return 0

    monkeypatch.setattr(stage_cache_dialog.StageCacheDialog, "exec", _fake_exec)
    # Không ghi cấu hình trang vào QSettings thật của máy (ảnh hưởng test khác).
    monkeypatch.setattr(page, "_persist_inputs", lambda: None)
    page._on_stage_cache_clicked()
    assert len(opened) == 1
    overview = opened[0]
    assert overview.stored_kinds == [LITERAL, STYLE]
    page.close()


def test_dialog_edit_save_apply_delete(
    app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    from subtitles_extractor.presentation.widgets.stage_cache_dialog import (
        StageCacheDialog,
        StageEditorDialog,
    )

    page, container = _page()
    _translate_into_cache(page, container)
    view_model = page._view_model
    session = view_model.open_stage_cache(**page._stage_cache_settings())
    assert session is not None

    dialog = StageCacheDialog(session)
    assert dialog._table.rowCount() == 4
    literal_row = 1
    assert "Dùng lại nguyên" in dialog._table.item(literal_row, 5).text()
    assert dialog._table.item(literal_row, 1).text() == f"{LINES} dòng"

    editor = StageEditorDialog(session, LITERAL)
    assert editor._table.rowCount() == LINES
    assert editor._table.item(2, 2).text() == "句3"
    assert editor._table.item(2, 3).text() == "句3"
    editor._table.item(2, 4).setText("sửa tay 3")
    assert editor._pending == {3: "sửa tay 3"}
    assert editor._btn_save.isEnabled()
    assert editor._on_save()
    assert editor._pending == {}
    assert "Tinh chỉnh" in editor._status.text()
    # Bảng so sánh giai đoạn theo kịp nội dung vừa sửa.
    assert view_model._stage_outputs[LITERAL][2].text == "sửa tay 3"
    editor._only_edited.setChecked(True)
    hidden = sum(editor._table.isRowHidden(row) for row in range(LINES))
    assert hidden == LINES - 1
    editor.close()

    dialog.refresh()
    style_next = dialog._table.item(2, 5).text()
    assert "Giữ 5/6 dòng" in style_next

    results: list[list[str]] = []
    view_model.result_ready.connect(
        lambda _src, translated: results.append([event.text for event in translated])
    )
    session.apply_stage(STYLE)
    assert results and results[-1] == [f"mượt {i}" for i in range(1, LINES + 1)]

    monkeypatch.setattr(QMessageBox, "exec", lambda _self: QMessageBox.StandardButton.Yes)
    dialog._table.selectRow(2)
    dialog._on_delete()
    assert dialog._table.item(2, 1).text() == "Chưa có"
    dialog.close()
    page.close()


def test_editor_revert_selected_and_unsaved_prompt(
    app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    from subtitles_extractor.presentation.widgets.stage_cache_dialog import (
        StageEditorDialog,
    )

    page, container = _page()
    _translate_into_cache(page, container)
    session = page._view_model.open_stage_cache(**page._stage_cache_settings())
    session.save_edits(LITERAL, {1: "chốt 1"})
    editor = StageEditorDialog(session, LITERAL)
    assert editor._btn_revert_all.isEnabled()
    editor._table.selectRow(0)
    editor._on_revert_selected()
    assert editor._pending == {1: "dịch 1"}
    monkeypatch.setattr(
        QMessageBox, "question", lambda *_a, **_k: QMessageBox.StandardButton.Cancel
    )
    editor.close()
    assert editor._pending  # huỷ đóng → còn nguyên dòng chưa lưu
    monkeypatch.setattr(
        QMessageBox, "question", lambda *_a, **_k: QMessageBox.StandardButton.Save
    )
    editor.close()
    rows = session.editor_rows(LITERAL)
    assert not rows[0].edited and rows[0].text == "dịch 1"
    page.close()


def test_retranslating_selected_lines_does_not_fill_the_stage_cache(
    app: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Bộ nhớ đệm nay được giữ sau khi xong: tập dòng con (dịch lại vài dòng) không được
    thành file riêng, kẻo chiếm chỗ bộ nhớ đệm của các phim."""
    from types import SimpleNamespace

    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.domain.ports.subtitle_translator_port import (
        TranslationStageConfig,
    )
    from subtitles_extractor.infrastructure.settings.json_file_repository import (
        JsonFileRepository,
    )
    from subtitles_extractor.presentation.view_models import translate_page_view_model

    container = ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp_path / "c.json"),
        i18n_data_dir=tmp_path,
        user_data_dir=tmp_path,
    )
    assert container.make_translate_subtitles_use_case(
        "KEY", 1, use_stage_cache=False
    )._checkpoint_dir is None
    assert container.make_translate_subtitles_use_case("KEY", 1)._checkpoint_dir == (
        container.stage_checkpoint_dir
    )

    captured: dict[str, object] = {}

    class _Worker:
        def __init__(self, **kwargs: object) -> None:
            self.progress_changed = SimpleNamespace(connect=lambda _f: None)
            self.finished_ok = self.cancelled = self.failed = self.finished = (
                self.progress_changed
            )

        def start(self) -> None:
            captured["started"] = True

    def _factory(*args: object, **kwargs: object) -> object:
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(translate_page_view_model, "TranslateWorker", _Worker)
    monkeypatch.setattr(
        container, "make_subtitle_translator",
        lambda *_a, **_k: SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setattr(container, "make_translate_subtitles_use_case", _factory)
    view_model = translate_page_view_model.TranslatePageViewModel(container)
    view_model.set_source_events(_events())
    view_model._translated_events = _events()
    assert view_model.retranslate_lines(
        "KEY", 1, [1, 2],
        [TranslationStageConfig(kind=LITERAL, model_name="gemini-3.8-flash")],
        TranslationContext(target_lang="Vietnamese"),
    )
    assert captured["kwargs"] == {"use_stage_cache": False}


def test_editor_refuses_a_blank_line(app: QApplication) -> None:
    """[v3.23.609] Xoá trắng một ô: không nhận, ô trở về câu trước, có lời nhắc."""
    from subtitles_extractor.presentation.widgets.stage_cache_dialog import (
        StageEditorDialog,
    )

    page, container = _page()
    _translate_into_cache(page, container)
    session = page._view_model.open_stage_cache(**page._stage_cache_settings())
    editor = StageEditorDialog(session, LITERAL)
    editor._table.item(0, 4).setText("   ")
    assert editor._pending == {}
    assert editor._table.item(0, 4).text() == "dịch 1"
    assert "không để trống" in editor._status.text()
    editor.close()
    page.close()
