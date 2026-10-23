"""[v3.23.650] Crash/kẹt trên trang Lồng tiếng (TTS) do rà soát độc lập: bấm "Rút gọn" hai lần, nạp giọng lỗi lạ."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

_PRELUDE = textwrap.dedent(
    """
    import os, sys, tempfile, time
    from pathlib import Path
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
    from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import JsonFileRepository
    from subtitles_extractor.presentation.pages import tts_page as m

    ApplicationContainer.preload_ocr_engine_async = lambda self: None
    tmp = Path(tempfile.mkdtemp())
    ini = tmp / "page.ini"
    m.QSettings = lambda *a: QSettings(str(ini), QSettings.Format.IniFormat)
    container = ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp / "config.json"),
        i18n_data_dir=Path(m.__file__).resolve().parents[2] / "data", user_data_dir=tmp)
    page = m.TTSPage(container)
    page.show(); app.processEvents()

    def pump(ms):
        end = time.time() + ms / 1000
        while time.time() < end:
            app.processEvents(); time.sleep(0.01)

    def events(n):
        return [SubtitleEvent(index=i + 1, text=f"Câu số {i} xin chào", interval=TimeInterval(i * 2.0, i * 2.0 + 1.5))
                for i in range(n)]
    """
)


def _run_editor(body: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    """Trang Chỉnh sửa với 40 dòng (dùng lại prelude của test_editor_crash_v323650)."""
    from tests.unit.presentation import test_editor_crash_v323650 as _ed   # noqa: PLC0415

    return _ed._run(body, timeout)


def _run(body: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    src = str(Path(__file__).resolve().parents[3] / "src")
    return subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", _PRELUDE + textwrap.dedent(body)],
        capture_output=True, text=True, timeout=timeout, check=False,
        env=dict(os.environ, PYTHONPATH=src, QT_QPA_PLATFORM="offscreen"),
    )


def test_second_shorten_click_while_first_runs_does_not_abort_v323650() -> None:
    result = _run(
        """
        from subtitles_extractor.application.use_cases import shorten_for_speech as sfs
        def slow_execute(self, *a, **k):
            time.sleep(1.5); return []
        sfs.ShortenForSpeechUseCase.execute = slow_execute
        container.make_subtitle_translator = lambda *a, **k: object()
        page.set_shorten_provider(lambda: ("KEY", "model", "Vietnamese"))
        page._view_model.set_source_events(events(5))
        page._start_shorten([]); pump(150)
        first = page._shorten_worker
        page._start_shorten([])                  # lần bấm thứ hai khi lần đầu còn chạy
        print("SAME=" + str(page._shorten_worker is first))
        pump(2500)
        print("DONE")
        """
    )
    assert result.returncode == 0, f"exit={result.returncode}\n{result.stderr[-1500:]}"
    assert "SAME=True" in result.stdout and "DONE" in result.stdout


def test_voice_loader_unexpected_error_restores_ui_v323650() -> None:
    result = _run(
        """
        class BadAdapter:
            supports_line_cache = False
            def is_available(self): return True
            def get_engine_name(self): return "Bad"
            def list_voice_catalog(self, lang): return []
            def list_speakers(self, lang): raise TypeError("API drift")   # không thuộc (Runtime|OS|Value)Error
        container.make_vieneu_tts_adapter = lambda *a, **k: BadAdapter()
        page._engine_avail = {"edge": True, "vieneu": True}
        page._engine_combo.setCurrentIndex(page._engine_combo.findData("vieneu"))
        pump(1200)
        print("COMBO_ENABLED=" + str(page._vieneu_voice.isEnabled()))
        print("LOADER_RUNNING=" + str(page._voice_loader_running()))
        page._update_action_states()             # trước đây ném RuntimeError (C++ object already deleted)
        print("DONE")
        """
    )
    assert result.returncode == 0, f"exit={result.returncode}\n{result.stderr[-1500:]}"
    assert "COMBO_ENABLED=True" in result.stdout and "LOADER_RUNNING=False" in result.stdout and "DONE" in result.stdout


def test_shortened_text_not_applied_to_a_shifted_line_v323650() -> None:
    """Xoá dòng trong lúc AI rút gọn: câu rút gọn không được ghi nhầm vào dòng kế bên."""
    result = _run(
        """
        from subtitles_extractor.application.use_cases.shorten_for_speech import ShortenedLine
        import subtitles_extractor.presentation.pages.tts_line_edits as le
        page._view_model.set_source_events(events(4))
        before = page._view_model.source_events[0].text
        lines = [ShortenedLine(position=0, before=before, after="ngắn", syllables_before=9, syllables_after=2)]
        page._view_model.delete_lines({0})                       # dòng 0 biến mất, dòng cũ 1 dồn lên vị trí 0
        le.ask_shorten_preview = lambda win, t, i, h, ls: list(ls)   # chọn hết những gì được đưa ra
        applied = page._line_edits.apply_shortened(lines)
        print("APPLIED=%d" % applied)
        print("TEXT0=" + page._view_model.source_events[0].text)
        """
    )
    assert result.returncode == 0, f"exit={result.returncode}\n{result.stderr[-1500:]}"
    assert "APPLIED=0" in result.stdout
    assert "TEXT0=Câu số 1 xin chào" in result.stdout


def test_publish_batch_item_that_cannot_start_is_counted_as_failure_not_stall_v323650() -> None:
    result = _run(
        """
        from types import SimpleNamespace
        from PySide6.QtWidgets import QMessageBox
        from subtitles_extractor.presentation.pages.publish_page import PublishPage
        QMessageBox.information = staticmethod(lambda *a, **k: None)
        pub = PublishPage(container)
        items = [SimpleNamespace(video_path=Path(f"/tmp/ep{i}.mp4"), subtitle_path=None, audio_path=None,
                                 output_path=Path(f"/tmp/out{i}.mp4")) for i in range(2)]
        pub._batch_items = items; pub._batch_index = 0; pub._batch_failures = []; pub._batch_cancelled = False
        pub._build_request = lambda: None                     # thiếu đầu vào → không bắt đầu được
        pub._start_render()
        pump(300)
        print("FAILURES=%d" % len(pub._batch_failures) if pub._batch_items else "FINISHED")
        print("ITEMS=%d" % len(pub._batch_items))
        """
    )
    assert result.returncode == 0, f"exit={result.returncode}\n{result.stderr[-1500:]}"
    assert "FINISHED" in result.stdout and "ITEMS=0" in result.stdout


def test_tts_cancel_during_batch_clears_queue_v323650() -> None:
    result = _run(
        """
        from types import SimpleNamespace
        from PySide6.QtWidgets import QMessageBox
        shown = []
        QMessageBox.information = staticmethod(lambda *a, **k: shown.append(a[2] if len(a) > 2 else ""))
        items = [SimpleNamespace(video_path=Path(f"/tmp/ep{i}.mp4"), subtitle_path=None, output_path=Path(f"/tmp/o{i}.wav"))
                 for i in range(3)]
        page._batch_items = items; page._batch_index = 0; page._batch_failures = []; page._batch_cancelled = False
        page._view_model.tts_cancelled.emit()                  # người dùng huỷ → VM báo tts_cancelled
        print("ITEMS=%d" % len(page._batch_items))
        print("SUMMARY=%d" % len(shown))
        """
    )
    assert result.returncode == 0, f"exit={result.returncode}\n{result.stderr[-1500:]}"
    assert "ITEMS=0" in result.stdout and "SUMMARY=1" in result.stdout


def test_tts_batch_stops_when_generation_cannot_start_v323650() -> None:
    result = _run(
        """
        from types import SimpleNamespace
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information = staticmethod(lambda *a, **k: None)
        items = [SimpleNamespace(video_path=Path(f"/tmp/ep{i}.mp4"), subtitle_path=None, output_path=Path(f"/tmp/o{i}.wav"))
                 for i in range(3)]
        page._batch_items = items; page._batch_index = 0; page._batch_failures = []; page._batch_cancelled = False
        page._on_generate = lambda: None                       # thoát sớm: engine chưa cài / thiếu khoá…
        page._generate_for_batch(); pump(200)
        print("ITEMS=%d" % len(page._batch_items))
        """
    )
    assert result.returncode == 0, f"exit={result.returncode}\n{result.stderr[-1500:]}"
    assert "ITEMS=0" in result.stdout


def test_reset_database_sqlite_error_still_emits_signal_v323650() -> None:
    result = _run(
        """
        import sqlite3
        from subtitles_extractor.presentation.view_models.settings_page_view_model import SettingsPageViewModel
        class _C:
            def reset_database(self): raise sqlite3.OperationalError("database is locked")
        vm = SettingsPageViewModel.__new__(SettingsPageViewModel)
        SettingsPageViewModel.__init__(vm, container)
        vm._container = _C()
        got = []
        vm.database_reset.connect(got.append)
        vm.reset_database()
        print("EMITTED=" + str(got))
        """
    )
    assert result.returncode == 0, f"exit={result.returncode}\n{result.stderr[-1500:]}"
    assert "EMITTED=[0]" in result.stdout


def test_translate_result_table_refill_is_not_quadratic_v323650() -> None:
    """Điền lại bảng kết quả dịch đã đầy (dịch lại dòng, nạp lại phiên…): trước đây O(n²) (3649 hàng ≈ 100 s)."""
    result = _run(
        """
        from subtitles_extractor.presentation.pages.translate_page import TranslatePage
        tp = TranslatePage(container)
        tp.resize(1200, 800); tp.show(); pump(100)
        n = 1500
        src = events(n)
        tr = [SubtitleEvent(index=e.index, text="Bản dịch " + e.text, interval=e.interval) for e in src]
        tp._on_result_ready(src, tr); pump(50)
        t0 = time.perf_counter()
        tp._on_result_ready(src, tr); pump(50)                  # lần hai: bảng đã đầy
        print("SECONDS=%.2f" % (time.perf_counter() - t0))
        print("ROWS=%d" % tp._table.rowCount())
        """,
        timeout=300,
    )
    assert result.returncode == 0, f"exit={result.returncode}\n{result.stderr[-1500:]}"
    seconds = float(result.stdout.split("SECONDS=")[1].split()[0])
    assert "ROWS=1500" in result.stdout
    assert seconds < 5.0, f"refill bảng đầy phải nhanh (đo {seconds:.1f}s)"


def test_translate_vm_survives_sqlite_error_in_memory_store_v323650() -> None:
    """DB khoá/đầy đĩa ("database is locked") không được làm view-model dịch ném ngoại lệ ra khỏi slot."""
    result = _run(
        """
        import sqlite3
        from subtitles_extractor.presentation.view_models.translate_page_view_model import TranslatePageViewModel
        vm = TranslatePageViewModel(container)
        class _Store:
            def list_series(self): raise sqlite3.OperationalError("database is locked")
            def clear_series(self, key): raise sqlite3.OperationalError("database is locked")
        vm._container = type("C", (), {"translation_memory_store": _Store()})()
        print("LIST=" + str(vm.list_memory_series()))
        print("CLEAR=" + str(vm.clear_memory_series("k")))
        """
    )
    assert result.returncode == 0, f"exit={result.returncode}\n{result.stderr[-1500:]}"
    assert "LIST=[]" in result.stdout and "CLEAR=False" in result.stdout


def test_translate_batch_export_failure_advances_exactly_once_v323650() -> None:
    result = _run(
        """
        from types import SimpleNamespace
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information = staticmethod(lambda *a, **k: None)
        from subtitles_extractor.presentation.pages.translate_page import TranslatePage
        tp = TranslatePage(container)
        items = [SimpleNamespace(video_path=Path(f"/tmp/ep{i}.mp4")) for i in range(3)]
        tp._batch_items = items; tp._batch_index = 0; tp._batch_done = 0; tp._batch_failures = []
        tp._batch_cancelled = False
        started = []
        tp._run_next_batch_translate = lambda: started.append(tp._batch_index)
        def failing_export(path, fmt):
            tp._view_model.error_occurred.emit("không ghi được tệp")   # như export_translation thật khi lỗi ghi
            return False
        tp._view_model.export_translation = failing_export
        tp._on_translation_finished(); pump(200)
        print("INDEX=%d" % tp._batch_index)
        print("DONE=%d" % tp._batch_done)
        print("FAILED=%s" % tp._batch_failures)
        print("STARTED=%s" % started)
        """
    )
    assert result.returncode == 0, f"exit={result.returncode}\n{result.stderr[-1500:]}"
    assert "INDEX=1" in result.stdout and "DONE=0" in result.stdout
    assert "ep0.mp4" in result.stdout and "STARTED=[1]" in result.stdout


def test_tts_batch_start_error_is_counted_once_and_batch_continues_v323650() -> None:
    result = _run(
        """
        from types import SimpleNamespace
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information = staticmethod(lambda *a, **k: None)
        items = [SimpleNamespace(video_path=Path(f"/tmp/ep{i}.mp4"), subtitle_path=None, output_path=Path(f"/tmp/o{i}.wav"))
                 for i in range(3)]
        page._batch_items = items; page._batch_index = 0; page._batch_failures = []; page._batch_cancelled = False
        started = []
        page._run_next_batch_tts = lambda: started.append(page._batch_index)
        def failing_generate():
            page._view_model.error_occurred.emit("engine không khả dụng")   # phát ĐỒNG BỘ như start_tts thật
        page._on_generate = failing_generate
        page._generate_for_batch(); pump(200)
        print("CANCELLED=%s" % page._batch_cancelled)
        print("FAILURES=%s" % page._batch_failures)
        print("INDEX=%d" % page._batch_index)
        """
    )
    assert result.returncode == 0, f"exit={result.returncode}\n{result.stderr[-1500:]}"
    assert "CANCELLED=False" in result.stdout and "INDEX=1" in result.stdout
    assert result.stdout.count("ep") == 1                      # chỉ ep0 bị tính lỗi


def test_queued_table_edit_follows_uid_when_rows_shift_v323650() -> None:
    """Sửa chữ trong ô, rồi trước khi vòng lặp sự kiện xử lý, dòng bị dời: sửa phải rơi vào ĐÚNG câu (theo uid)."""
    from textwrap import dedent  # noqa: F401

    result = _run_editor(
        """
        target = page._table_model.events[5]
        page._table_model.text_edit_requested.emit(5, "SỬA")      # như model.setData phát
        page._view_model.delete_event(0)                           # danh sách dời lên một hàng trước khi áp dụng
        pump(150)
        texts = [e.text for e in page._table_model.events]
        print("EDITED_AT=%d" % texts.index("SỬA"))
        print("OTHER_INTACT=" + str(texts.count("SỬA") == 1))
        """
    )
    assert result.returncode == 0, f"exit={result.returncode}\n{result.stderr[-1500:]}"
    assert "EDITED_AT=4" in result.stdout and "OTHER_INTACT=True" in result.stdout


def test_translate_selection_survives_table_refill_v323650() -> None:
    result = _run(
        """
        from subtitles_extractor.presentation.pages.translate_page import TranslatePage
        tp = TranslatePage(container)
        tp.resize(1200, 800); tp.show(); pump(100)
        src = events(30)
        tr = [SubtitleEvent(index=e.index, text="dịch " + e.text, interval=e.interval) for e in src]
        tp._on_result_ready(src, tr); pump(50)
        tp._table.selectRow(7); pump(50)
        tp._on_result_ready(src, tr); pump(50)
        print("SELECTED=%s" % sorted({i.row() for i in tp._table.selectionModel().selectedRows()}))
        """
    )
    assert result.returncode == 0, f"exit={result.returncode}\n{result.stderr[-1500:]}"
    assert "SELECTED=[7]" in result.stdout
