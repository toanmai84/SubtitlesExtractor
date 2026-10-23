"""[v3.23.609] Lỗi tìm được khi rà toàn bộ trang Dịch.

* Nạp tệp phụ đề / lấy từ trang Biên tập: video của dự án mở TRƯỚC vẫn được coi là video
  của phụ đề mới → nạp phân tích và bản dịch của phim cũ, ghi phiên vào video cũ, đặt tên
  file xuất theo video cũ; còn video cạnh tệp phụ đề thì gắn SAU khi đã nạp lại phiên nên
  phân tích đã lưu của chính nó không bao giờ được nạp.
* Nạp phụ đề mới khi đang đính video: ô "Phân tích hình ảnh" còn bật + đánh dấu dù ô
  đính video đã tắt và video đã gỡ → bấm Dịch bị chặn "Chưa chọn video ngữ cảnh".
* Dịch KHÔNG đính video: không lưu gì vào phiên (khoá phiên là video đính kèm) → mở lại
  mất bản dịch, Bộ nhớ dịch phim bộ không tích luỹ.
* Nạp lại phiên ghép bản dịch theo số thứ tự khi số dòng chỉ lệch ≤ 20% — phụ đề khác
  cùng cỡ vẫn nhận bản dịch của phim kia; chỉ số dòng không bắt đầu từ 1 bị lệch một dòng.
* Thay phụ đề nguồn mà "So sánh giai đoạn" vẫn hiện các giai đoạn của phim cũ.
* Dịch cả bộ: nạp lại phiên (``result_ready``) bị coi là "dịch xong tập" → ghi bản dịch cũ
  thành tệp của tập và nhảy tập; huỷ làm hàng đợi treo; cảnh báo cấu hình làm hàng đợi
  đứng im; mọi tập dùng chung ngữ cảnh/video; ước lượng request đọc ô không tồn tại;
  tổng kết khi hết hạn mức báo thiếu một tập.
* Thông báo "Dịch hoàn tất" hiện cả khi chỉ mở lại phiên.
"""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationStageKind,
)
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

LITERAL = TranslationStageKind.LITERAL
STYLE = TranslationStageKind.STYLE


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _events(prefix: str, count: int = 6, first_index: int = 1) -> list[SubtitleEvent]:
    return [
        SubtitleEvent(
            index=first_index + i, text=f"{prefix}{i}", interval=TimeInterval(i, i + 0.9)
        )
        for i in range(count)
    ]


def _write_srt(path: Path, events: list[SubtitleEvent]) -> Path:
    blocks = []
    for position, event in enumerate(events, start=1):
        blocks.append(
            f"{position}\n00:00:0{position},000 --> 00:00:0{position},900\n{event.text}\n"
        )
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


def _page(tmp: Path | None = None):
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import (
        JsonFileRepository,
    )
    from subtitles_extractor.presentation.pages import translate_page

    tmp = tmp or Path(tempfile.mkdtemp())
    container = ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp / "c.json"),
        i18n_data_dir=Path(translate_page.__file__).resolve().parents[2] / "data",
        user_data_dir=tmp,
    )
    page = translate_page.TranslatePage(container)
    page._persist_inputs = lambda: None  # không ghi QSettings thật của máy
    for panel, enabled in (
        (page._stage_preprocess, False), (page._stage_literal, True),
        (page._stage_style, False), (page._stage_localize, False),
    ):
        panel.enable_check.setChecked(enabled)
    page._lang_combo.setCurrentText("Vietnamese")
    return page, container


def _save_translation(container, video: str, events: list[SubtitleEvent]) -> None:
    """Lưu phiên kiểu bản cũ (không vân tay câu gốc) cho ``video``."""
    import json

    from subtitles_extractor.infrastructure.database import (
        sqlite_translation_session_store as session_store,
    )

    lines = json.dumps(
        [{"index": ev.index, "text": f"dịch {ev.text}"} for ev in events],
        ensure_ascii=False,
    )
    store = container.translation_session_store
    for stage_id in ("final", "literal"):
        store.save_stage(
            video,
            session_store.StageResult(stage_id=stage_id, input_hash="", lines_json=lines),
        )


def _save_analysis(
    container, video: str, events: list[SubtitleEvent], roster: str
) -> None:
    from subtitles_extractor.application.services.translation_session_hashing import (
        hash_analysis_input,
    )

    container.translation_session_store.save_analysis(
        video,
        characters=roster, overview="Tóm tắt", source_lang="Chinese",
        target_lang="Vietnamese",
        input_hash=hash_analysis_input([ev.text for ev in events], "Vietnamese"),
    )


def _pick_file(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *_a, **_k: (str(path), "")
    )


class TestSourceVideoOfLoadedSubtitles:
    def test_file_of_another_film_does_not_get_the_previous_project(
        self, app: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        page, container = _page(tmp_path)
        old_video = str(tmp_path / "old" / "phim_cu.mp4")
        _save_translation(container, old_video, _events("旧"))
        page.suggest_source_path(old_video)  # dự án mở trước đó
        _pick_file(monkeypatch, _write_srt(tmp_path / "phim_moi.srt", _events("新")))
        page._on_load_file_clicked()
        assert page._view_model.translated_events == []
        assert page._table.rowCount() == 0
        page.close()

    def test_sibling_video_analysis_is_restored(
        self, app: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        page, container = _page(tmp_path)
        events = _events("新")
        srt = _write_srt(tmp_path / "tap1.srt", events)
        video = tmp_path / "tap1.mp4"
        video.write_bytes(b"")
        _save_analysis(container, str(video), events, "Lâm Dật (nam)")
        _pick_file(monkeypatch, srt)
        page._on_load_file_clicked()
        assert page._characters_edit.toPlainText() == "Lâm Dật (nam)"
        page.close()

    def test_pull_from_editor_uses_the_editor_video(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        page, container = _page(tmp_path)
        events = _events("新")
        video = str(tmp_path / "tap2.mp4")
        _save_analysis(container, video, events, "Tô Mộc (nữ)")
        page.set_editor_event_provider(lambda: list(events))
        page.set_editor_video_provider(lambda: video)
        page._show_info = lambda *_a: None
        page._on_pull_editor_clicked()
        assert page._characters_edit.toPlainText() == "Tô Mộc (nữ)"
        page.close()

    def test_unreadable_file_keeps_the_current_subtitles_and_context(
        self, app: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        page, _container = _page(tmp_path)
        page._view_model.set_source_events(_events("旧"))
        page._characters_edit.setPlainText("Giữ nguyên")
        empty = tmp_path / "rong.srt"
        empty.write_text("", encoding="utf-8")
        _pick_file(monkeypatch, empty)
        page._on_load_file_clicked()
        assert page._characters_edit.toPlainText() == "Giữ nguyên"
        assert len(page._view_model.source_events) == 6
        page.close()


class TestVideoOptionsAfterNewFile:
    def test_new_file_does_not_block_translation(
        self, app: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        page, _container = _page(tmp_path)
        page._api_key_edit.setText("KEY")
        page._attach_video_check.setChecked(True)
        page._video_path = str(tmp_path / "x.mp4")
        assert page._visual_cues_check.isChecked()
        _pick_file(monkeypatch, _write_srt(tmp_path / "khac.srt", _events("新")))
        page._on_load_file_clicked()
        assert not page._visual_cues_check.isChecked()
        assert not page._visual_cues_check.isEnabled()
        warnings: list[str] = []
        page._show_warning = lambda title, *_a: warnings.append(title)
        calls: list[dict] = []
        monkeypatch.setattr(
            page._view_model, "start_translation",
            lambda **kwargs: calls.append(kwargs) or True,
        )
        page._on_run_clicked()
        assert warnings == []
        assert len(calls) == 1
        page.close()


class TestSessionWithoutAttachedVideo:
    def test_translation_is_saved_under_the_subtitles_video(
        self, app: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from subtitles_extractor.application.use_cases.translate_subtitles import (
            TranslateSubtitlesResponse,
        )
        from subtitles_extractor.presentation.view_models import (
            translate_page_view_model as vm_module,
        )

        page, container = _page(tmp_path)
        video = str(tmp_path / "tap3.mp4")
        page.suggest_source_path(video)
        page._view_model.set_source_events(_events("新"))
        page._api_key_edit.setText("KEY")
        assert not page._attach_video_check.isChecked()

        class _Worker:
            def __init__(self, **_kwargs: object) -> None:
                signal = SimpleNamespace(connect=lambda _f: None)
                self.progress_changed = self.finished_ok = self.cancelled = signal
                self.failed = self.finished = self.video_refs_ready = signal

            def start(self) -> None:
                return None

        monkeypatch.setattr(vm_module, "TranslateWorker", _Worker)
        monkeypatch.setattr(
            container, "make_subtitle_translator",
            lambda *_a, **_k: SimpleNamespace(is_available=lambda: True),
        )
        monkeypatch.setattr(
            container, "make_translate_subtitles_use_case", lambda *_a, **_k: object()
        )
        page._on_run_clicked()
        view_model = page._view_model
        view_model._worker = None
        translated = [
            replace(ev, text=f"dịch {ev.text}") for ev in view_model.source_events
        ]
        view_model._on_worker_finished(
            TranslateSubtitlesResponse(events=translated, completed_stages=[LITERAL])
        )
        session = container.translation_session_store.get(video)
        assert session is not None and session.stage("final") is not None
        page.close()


class TestSessionRestoreMatchesSubtitles:
    def _view_model(self, tmp_path: Path):
        page, container = _page(tmp_path)
        return page, page._view_model, container

    def test_translation_of_other_subtitles_is_not_restored(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        page, view_model, _container = self._view_model(tmp_path)
        video = str(tmp_path / "v.mp4")
        view_model.set_source_events(_events("甲"))
        view_model._last_translate_video_path = video
        view_model._translated_events = [
            replace(ev, text="dịch") for ev in view_model.source_events
        ]
        view_model._persist_current_result()
        view_model.set_source_events(_events("乙"))
        view_model.restore_stage_outputs_for_video(video)
        assert view_model.translated_events == []
        page.close()

    def test_a_few_typo_fixes_still_restore(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        page, view_model, _container = self._view_model(tmp_path)
        video = str(tmp_path / "v.mp4")
        source = _events("句", count=20)
        view_model.set_source_events(source)
        view_model._last_translate_video_path = video
        view_model._translated_events = [
            replace(ev, text=f"dịch {ev.index}") for ev in source
        ]
        view_model._persist_current_result()
        fixed = list(source)
        fixed[3] = replace(fixed[3], text="句3 (sửa chính tả)")
        view_model.set_source_events(fixed)
        view_model.restore_stage_outputs_for_video(video)
        assert [ev.text for ev in view_model.translated_events][:2] == [
            "dịch 1", "dịch 2"
        ]
        page.close()

    def test_indices_not_starting_at_one_are_matched_by_index(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        page, view_model, container = self._view_model(tmp_path)
        video = str(tmp_path / "v0.mp4")
        source = _events("零", first_index=0)
        _save_translation(container, video, source)
        view_model.set_source_events(source)
        view_model.restore_stage_outputs_for_video(video)
        assert [ev.text for ev in view_model.translated_events] == [
            f"dịch {ev.text}" for ev in source
        ]
        page.close()

    def test_new_subtitles_drop_the_old_stage_comparison(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        page, view_model, _container = self._view_model(tmp_path)
        view_model.set_source_events(_events("甲"))
        view_model._stage_outputs = {
            LITERAL: list(view_model.source_events),
            STYLE: list(view_model.source_events),
        }
        assert view_model.stage_comparison()
        view_model.set_source_events(_events("乙"))
        assert view_model.stage_comparison() == []
        page.close()


def _item(folder: Path, name: str, lines: int = 6):
    from subtitles_extractor.application.services.batch_translate_plan import (
        TranslateItem,
        TranslateItemStatus,
    )

    video = folder / f"{name}.mp4"
    video.write_bytes(b"")
    source = _write_srt(folder / f"{name}.original.srt", _events(name, lines))
    return TranslateItem(
        video_path=video, source_path=source,
        output_path=folder / f"{name}.translate.vi.srt", line_count=lines,
        status=TranslateItemStatus.READY,
    )


class TestBatchTranslate:
    def _batch_page(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        page, container = _page(tmp_path)
        page._api_key_edit.setText("KEY")
        summaries: list[str] = []
        monkeypatch.setattr(
            QMessageBox, "information", lambda _p, _t, text, *_a: summaries.append(text)
        )
        persisted: list[int] = []
        page._persist_batch_translation = lambda: persisted.append(page._batch_index) or True   # [v3.23.650] trả bool: ghi được tệp
        return page, container, summaries, persisted

    def test_restored_session_is_not_a_finished_episode(
        self, app: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        page, _container, _summaries, persisted = self._batch_page(tmp_path, monkeypatch)
        page._batch_items = [_item(tmp_path, "t1"), _item(tmp_path, "t2")]
        page._batch_index = 0
        page._view_model.set_source_events(_events("t1"))
        page._view_model.result_ready.emit(
            page._view_model.source_events, page._view_model.source_events
        )
        QApplication.processEvents()
        assert persisted == []
        assert page._batch_index == 0
        page._batch_items = []
        page.close()

    def test_cancel_ends_the_queue(
        self, app: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        page, _container, summaries, _persisted = self._batch_page(tmp_path, monkeypatch)
        page._batch_items = [_item(tmp_path, "t1"), _item(tmp_path, "t2")]
        page._batch_index = 0
        page._view_model.translation_cancelled.emit()
        assert page._batch_items == []
        assert summaries and summaries[0].startswith("Đã huỷ")
        page.close()

    def test_settings_problem_stops_the_queue_with_a_summary(
        self, app: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        page, _container, summaries, _persisted = self._batch_page(tmp_path, monkeypatch)
        page._api_key_edit.setText("")
        page._show_warning = lambda *_a: None
        page._batch_items = [_item(tmp_path, "t1"), _item(tmp_path, "t2")]
        page._batch_index = 0
        page._run_next_batch_translate()
        for _ in range(5):
            QApplication.processEvents()
        assert page._batch_items == []
        assert summaries and "t1.mp4" in summaries[0]
        page.close()

    def test_each_episode_gets_its_own_video_and_context(
        self, app: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        page, container, _summaries, _persisted = self._batch_page(tmp_path, monkeypatch)
        first = _item(tmp_path, "t1")
        _save_analysis(container, str(first.video_path), _events("t1"), "Riêng tập 1")
        page._visual_cues_edit.setPlainText('[{"line": 3, "speaker": "Phim cũ"}]')
        page._characters_edit.setPlainText("Nhân vật phim cũ")
        calls: list[dict] = []
        monkeypatch.setattr(
            page._view_model, "start_translation",
            lambda **kwargs: calls.append(kwargs) or True,
        )
        page._batch_items = [first, _item(tmp_path, "t2")]
        page._batch_index = 0
        page._run_next_batch_translate()
        for _ in range(5):
            QApplication.processEvents()
        assert page._visual_cues_edit.toPlainText() == ""
        assert page._characters_edit.toPlainText() == "Riêng tập 1"
        assert calls and calls[0]["session_path"] == str(first.video_path)
        page._batch_items = []
        page.close()

    def test_quota_stop_counts_the_finished_episode(
        self, app: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        page, _container, summaries, persisted = self._batch_page(tmp_path, monkeypatch)
        monkeypatch.setattr(
            page._view_model, "start_translation", lambda **_kwargs: True
        )
        page._batch_items = [
            _item(tmp_path, "t1"), _item(tmp_path, "t2"), _item(tmp_path, "t3")
        ]
        page._batch_index = 0
        page._batch_done = 0
        page._view_model.translation_finished.emit()  # tập 1 xong
        for _ in range(5):
            QApplication.processEvents()
        assert persisted == [0]
        page._view_model.error_occurred.emit("Hết hạn mức ngày (quota) của model.")
        assert summaries and "Đã dịch xong 1/3" in summaries[0]
        assert "Còn 2 tập chưa dịch" in summaries[0]
        page.close()

    def test_estimate_counts_every_enabled_stage_and_model(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        from subtitles_extractor.application.services.batch_translate_plan import (
            estimate_requests_by_model,
        )

        plan = [_item(tmp_path, "t1", lines=120), _item(tmp_path, "t2", lines=120)]
        assert estimate_requests_by_model(
            plan, [("gemini-3.8-flash", 50), ("gemini-3.5-flash-lite", 60),
                   ("gemini-3.8-flash", 40)]
        ) == {"gemini-3.8-flash": 2 * 3 + 2 * 3, "gemini-3.5-flash-lite": 2 * 2}
        page, _container = _page(tmp_path)
        page._stage_style.enable_check.setChecked(True)
        # [v3.23.635] Mặc định giai đoạn tinh chỉnh nay cũng là 3.1-flash-lite → chọn tường minh một
        # model khác để kiểm ước tính TÁCH theo model.
        page._stage_style.model_combo.setCurrentText("gemini-3.6-flash")
        page._stage_literal.batch_spin.setValue(200)
        page._stage_literal.model_combo.setCurrentText("gemini-3.1-flash-lite")
        text = page._batch_request_estimate(plan, page._collect_enabled_stages())
        # Lô 200 của model lite bị kẹp còn 60 → 2 request/tập.
        assert "gemini-3.1-flash-lite: ~4 request" in text
        assert page._stage_style.model_combo.currentText() in text
        page.close()


class TestFinishedToast:
    def test_restoring_a_session_does_not_announce_a_translation(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        page, _container = _page(tmp_path)
        toasts: list[str] = []
        page._show_success = lambda title, *_a: toasts.append(title)
        page._view_model.set_source_events(_events("甲"))
        page._view_model.result_ready.emit(
            page._view_model.source_events, page._view_model.source_events
        )
        assert toasts == []
        page._view_model.translation_finished.emit()
        assert toasts == ["Dịch hoàn tất"]
        page.close()


def test_view_model_reports_translation_finished_once(
    app: QApplication, tmp_path: Path
) -> None:
    from subtitles_extractor.application.use_cases.translate_subtitles import (
        TranslateSubtitlesResponse,
    )

    page, _container = _page(tmp_path)
    view_model = page._view_model
    view_model.set_source_events(_events("甲"))
    finished: list[int] = []
    view_model.translation_finished.connect(lambda: finished.append(1))
    view_model._worker = None
    view_model._on_worker_finished(
        TranslateSubtitlesResponse(
            events=list(view_model.source_events), completed_stages=[LITERAL]
        )
    )
    assert finished == [1]
    page.close()


class TestAnalysisWithoutAttachedVideo:
    def test_analysis_is_saved_under_the_subtitles_video(
        self, app: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from subtitles_extractor.domain.ports.subtitle_translator_port import (
            SubtitleContextAnalysis,
        )
        from subtitles_extractor.presentation.workers import analyze_context_worker

        page, container = _page(tmp_path)
        video = str(tmp_path / "tap4.mp4")
        page.suggest_source_path(video)
        events = _events("新")
        page._view_model.set_source_events(events)
        page._api_key_edit.setText("KEY")

        class _Worker:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                signal = SimpleNamespace(connect=lambda _f: None)
                self.finished_ok = self.video_refs_ready = self.cancelled = signal
                self.failed = self.progress_message = self.finished = signal

            def start(self) -> None:
                return None

        monkeypatch.setattr(analyze_context_worker, "AnalyzeContextWorker", _Worker)
        monkeypatch.setattr(
            container, "make_subtitle_translator",
            lambda *_a, **_k: SimpleNamespace(is_available=lambda: True),
        )
        monkeypatch.setattr(
            container, "make_analyze_context_use_case", lambda *_a, **_k: object()
        )
        page._on_analyze_context_clicked()
        page._view_model._on_analyze_finished(
            SubtitleContextAnalysis(
                source_lang="Chinese", characters="Lâm Dật (nam)", overview="Tóm tắt"
            )
        )
        session = container.translation_session_store.get(video)
        assert session is not None
        assert session.analysis_characters == "Lâm Dật (nam)"
        page.close()


def test_translator_not_ready_stops_the_whole_batch(
    app: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Lỗi cấu hình chung (bộ dịch chưa sẵn sàng) áp cho MỌI tập — dừng hàng đợi thay vì
    lần lượt làm hỏng từng tập."""
    page, _container = _page(tmp_path)
    summaries: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "information", lambda _p, _t, text, *_a: summaries.append(text)
    )
    page._batch_items = [
        _item(tmp_path, "t1"), _item(tmp_path, "t2"), _item(tmp_path, "t3")
    ]
    page._batch_index = 0
    page._view_model.error_occurred.emit(
        "Bộ dịch chưa sẵn sàng. Kiểm tra đã cài 'google-genai' và nhập đúng API Key."
    )
    assert page._batch_items == []
    assert summaries and "Thất bại:\n• t1.mp4" in summaries[0]
    assert "t2.mp4" not in summaries[0]
    page.close()


def test_retranslating_selected_lines_keeps_their_own_speakers(
    app: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dịch lại dòng 4 và 6: khâu dịch đánh số tập con 1..2, nên cues phải theo — trước
    đây dòng 4 nhận người nói của dòng 1 của phim, dòng 6 nhận của dòng 2."""
    from subtitles_extractor.application.services.visual_cue_serializer import (
        deserialize_visual_cues,
        serialize_visual_cues,
    )
    from subtitles_extractor.domain.ports.subtitle_translator_port import (
        TranslationContext,
        TranslationStageConfig,
        VisualCue,
    )
    from subtitles_extractor.presentation.view_models import (
        translate_page_view_model as vm_module,
    )

    page, container = _page(tmp_path)
    view_model = page._view_model
    view_model.set_source_events(_events("句"))
    view_model._translated_events = list(view_model.source_events)
    cues = serialize_visual_cues([
        VisualCue(line_no=1, speaker="Ông A", addressee="Bà B"),
        VisualCue(line_no=4, speaker="Cô C", addressee="Anh D"),
        VisualCue(line_no=6, speaker="Anh D", addressee="Cô C"),
    ])
    captured: dict[str, object] = {}

    class _Worker:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            signal = SimpleNamespace(connect=lambda _f: None)
            self.progress_changed = self.finished_ok = self.cancelled = signal
            self.failed = self.finished = self.video_refs_ready = signal

        def start(self) -> None:
            return None

    monkeypatch.setattr(vm_module, "TranslateWorker", _Worker)
    monkeypatch.setattr(
        container, "make_subtitle_translator",
        lambda *_a, **_k: SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setattr(
        container, "make_translate_subtitles_use_case", lambda *_a, **_k: object()
    )
    assert view_model.retranslate_lines(
        "KEY", 1, [3, 5],
        [TranslationStageConfig(kind=LITERAL, model_name="gemini-3.8-flash")],
        TranslationContext(target_lang="Vietnamese", visual_cues=cues),
    )
    request = captured["request"]
    sent = deserialize_visual_cues(request.context.visual_cues)
    assert [(cue.line_no, cue.speaker) for cue in sent] == [(1, "Cô C"), (2, "Anh D")]
    page.close()


def test_stage_used_as_translation_is_saved_without_attached_video(
    app: QApplication, tmp_path: Path
) -> None:
    """"Dùng làm bản dịch hiện hành" trong Quản lý giai đoạn phải lưu vào phiên của video
    phụ đề kể cả khi không đính video (trước đây chỉ lưu khi đính video)."""
    from tests.unit.presentation.test_stage_cache_ui_v323608 import _translate_into_cache

    page, container = _page(tmp_path)
    video = str(tmp_path / "tap5.mp4")
    page.suggest_source_path(video)
    page._view_model.set_source_events(_events("新"))
    _translate_into_cache(page, container)
    session = page._view_model.open_stage_cache(**page._stage_cache_settings())
    assert session is not None
    session.apply_stage(LITERAL)
    saved = container.translation_session_store.get(video)
    assert saved is not None and saved.stage("final") is not None
    page.close()


def test_batch_cannot_start_while_a_translation_runs(
    app: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    page, _container = _page(tmp_path)
    asked: list[int] = []
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", lambda *_a, **_k: asked.append(1) or ""
    )
    page._view_model._set_busy(True)
    assert not page._btn_batch.isEnabled()
    page._on_batch_translate_clicked()
    assert asked == []
    page._view_model._set_busy(False)
    assert page._btn_batch.isEnabled()
    page.close()
