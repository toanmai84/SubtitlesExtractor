"""[v3.23.650] Đóng ứng dụng dừng được mọi luồng nền; huỷ trang khi luồng còn chạy không còn abort()."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, QThread  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from subtitles_extractor.presentation.workers import thread_shutdown as ts  # noqa: E402

_app = QApplication.instance() or QApplication([])


class _LoopThread(QThread):
    """Chạy tới khi được huỷ mềm."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.stop_flag = threading.Event()

    def cancel(self) -> None:
        self.stop_flag.set()

    def run(self) -> None:
        while not self.stop_flag.is_set():
            time.sleep(0.01)


class _StubbornThread(QThread):
    """Bỏ ngoài tai mọi yêu cầu dừng (như một lệnh gọi mạng/ffmpeg kẹt)."""

    def run(self) -> None:
        time.sleep(30)


class _Page(QObject):
    pass


def _wait_running(thread: QThread) -> None:
    thread.start()
    deadline = time.time() + 5
    while not thread.isRunning() and time.time() < deadline:
        time.sleep(0.005)


def test_cancel_then_stop_ends_cooperative_thread_without_terminate_v323650() -> None:
    page = _Page()
    worker = _LoopThread(page)
    page._concat_thread = worker           # type: ignore[attr-defined]
    page._concat_worker = worker           # type: ignore[attr-defined]
    _wait_running(worker)
    assert ts.request_cancel_workers([page]) >= 1
    found = ts.running_threads(page, [page])
    assert found == [worker]
    assert ts.stop_threads(found, wait_ms=3000) == 0
    assert not worker.isRunning()


def test_stubborn_thread_is_terminated_so_window_can_close_v323650() -> None:
    page = _Page()
    worker = _StubbornThread(page)
    _wait_running(worker)
    start = time.time()
    killed = ts.stop_threads(ts.running_threads(page), wait_ms=300, kill_wait_ms=2000)
    assert killed == 1 and not worker.isRunning()
    assert time.time() - start < 5


def test_python_held_parentless_thread_is_found_via_page_attribute_v323650() -> None:
    """Luồng rút gọn AI không có parent (chỉ Python giữ) vẫn phải được tìm thấy qua thuộc tính của trang."""
    page = _Page()
    worker = _LoopThread()                 # không parent
    page._shorten_worker = worker          # type: ignore[attr-defined]
    _wait_running(worker)
    assert worker in ts.running_threads(page, [page])
    ts.request_cancel_workers([page])
    ts.stop_threads(ts.running_threads(page, [page]))
    assert not worker.isRunning()


def test_deleting_owner_after_stop_does_not_abort_v323650() -> None:
    script = textwrap.dedent(
        """
        import os, time, threading
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtCore import QObject, QThread
        from PySide6.QtWidgets import QApplication
        app = QApplication([])
        from subtitles_extractor.presentation.workers import thread_shutdown as ts

        class T(QThread):
            def __init__(self, p):
                super().__init__(p); self.ev = threading.Event()
            def cancel(self): self.ev.set()
            def run(self):
                while not self.ev.is_set(): time.sleep(0.01)

        page = QObject()
        page._worker = T(page); page._worker.start()
        time.sleep(0.1)
        ts.request_cancel_workers([page])
        ts.stop_threads(ts.running_threads(page, [page]))
        del page
        print("OK")
        """
    )
    src = str(Path(__file__).resolve().parents[3] / "src")
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60,
                            env=dict(os.environ, PYTHONPATH=src), check=False)
    assert result.returncode == 0, result.stderr[-1000:]
    assert "OK" in result.stdout


def test_main_window_close_stops_publish_thread_instead_of_aborting_v323650() -> None:
    """Đóng cửa sổ chính khi trang Xuất bản đang render: trước đây abort() (exit 134) vì QThread con bị huỷ khi còn chạy."""
    script = textwrap.dedent(
        """
        import os, sys, tempfile, time, threading
        from pathlib import Path
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtCore import QThread
        from PySide6.QtWidgets import QApplication, QMessageBox
        app = QApplication([])
        QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
        from subtitles_extractor.composition.container import ApplicationContainer
        from subtitles_extractor.infrastructure.settings.json_file_repository import JsonFileRepository
        from subtitles_extractor.presentation.main_window import MainWindow
        from subtitles_extractor.presentation.pages import editor_page as ep
        ApplicationContainer.preload_ocr_engine_async = lambda self: None
        tmp = Path(tempfile.mkdtemp(prefix="mw650_"))
        container = ApplicationContainer(
            settings_repository=JsonFileRepository(file_path=tmp / "c.json"),
            i18n_data_dir=Path(ep.__file__).resolve().parents[2] / "data", user_data_dir=tmp)
        window = MainWindow(container)
        window.show(); app.processEvents()

        class Render(QThread):
            def __init__(self, p):
                super().__init__(p); self.ev = threading.Event()
            def cancel(self): self.ev.set()
            def run(self):
                while not self.ev.is_set(): time.sleep(0.01)

        publish = window._publish_page
        publish._thread = Render(publish)
        publish._worker = publish._thread
        publish._thread.start(); time.sleep(0.1)
        window.close(); app.processEvents()
        print("RUNNING=" + str(publish._thread.isRunning()))
        del window
        print("OK")
        """
    )
    src = str(Path(__file__).resolve().parents[3] / "src")
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120,
                            env=dict(os.environ, PYTHONPATH=src), check=False)
    assert result.returncode == 0, f"exit={result.returncode}\n{result.stderr[-1500:]}"
    assert "RUNNING=False" in result.stdout and "OK" in result.stdout


class _Worker(QObject):
    from PySide6.QtCore import Signal as _Signal

    finished = _Signal()

    def run(self) -> None:
        self.finished.emit()


def _wait_after_finish(connection) -> bool:
    """Worker xong ngay, rồi luồng GUI gọi thread.wait() (như closeEvent) — không bơm sự kiện."""
    thread = QThread()
    worker = _Worker()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit, connection)
    thread.start()
    time.sleep(0.2)
    stopped = thread.wait(1000)
    thread.quit()                      # dọn: không để QThread bị huỷ khi còn chạy
    thread.wait(2000)
    return stopped


def test_queued_quit_never_arrives_while_gui_thread_waits_but_direct_does_v323650() -> None:
    from PySide6.QtCore import Qt

    assert _wait_after_finish(Qt.ConnectionType.QueuedConnection) is False   # lý do closeEvent luôn chờ hết 8 s
    assert _wait_after_finish(Qt.ConnectionType.DirectConnection) is True


def test_view_models_connect_thread_quit_directly_v323650() -> None:
    import re

    root = Path(__file__).resolve().parents[3] / "src" / "subtitles_extractor" / "presentation"
    offenders = []
    for rel in ("view_models/extract_page_view_model.py", "view_models/editor_page_view_model.py",
                "view_models/debug_page_view_model.py"):
        text = (root / rel).read_text(encoding="utf-8")
        offenders += [f"{rel}: {m.group(0)}" for m in re.finditer(r"\.connect\((?:self\._async_thread|thread)\.quit\)", text)]
    assert not offenders, offenders


def test_no_worker_signal_is_connected_to_a_lambda_touching_widgets_v323650() -> None:
    """Lambda nối vào tín hiệu của worker chạy ở LUỒNG WORKER (PySide6) → đụng QLabel sai luồng → segfault ngẫu nhiên."""
    import re

    root = Path(__file__).resolve().parents[3] / "src" / "subtitles_extractor" / "presentation"
    offenders = []
    for rel in ("pages/settings_page.py", "pages/extract_page.py", "pages/tts_page.py", "pages/publish_page.py"):
        text = (root / rel).read_text(encoding="utf-8")
        for m in re.finditer(r"worker\.(?:progress|cancelled|finished|failed)\.connect\(\s*lambda[^\n]*", text):
            if "logger." not in m.group(0):                 # lambda chỉ ghi log thì an toàn ở mọi luồng
                offenders.append(f"{rel}: {m.group(0)[:80]}")
    assert not offenders, offenders


def _run_window_script(body: str, timeout: int = 120) -> "subprocess.CompletedProcess[str]":
    prelude = textwrap.dedent(
        """
        import os, sys, tempfile, time
        from pathlib import Path
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication, QMessageBox
        app = QApplication([])
        QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
        QMessageBox.information = staticmethod(lambda *a, **k: None)
        QMessageBox.warning = staticmethod(lambda *a, **k: None)
        from subtitles_extractor.composition.container import ApplicationContainer
        from subtitles_extractor.infrastructure.settings.json_file_repository import JsonFileRepository
        from subtitles_extractor.presentation.main_window import MainWindow
        from subtitles_extractor.presentation.pages import editor_page as ep
        from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
        from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
        ApplicationContainer.preload_ocr_engine_async = lambda self: None
        tmp = Path(tempfile.mkdtemp(prefix="mw650_"))
        container = ApplicationContainer(
            settings_repository=JsonFileRepository(file_path=tmp / "c.json"),
            i18n_data_dir=Path(ep.__file__).resolve().parents[2] / "data", user_data_dir=tmp)
        window = MainWindow(container)
        window.show(); app.processEvents()
        """
    )
    src = str(Path(__file__).resolve().parents[3] / "src")
    return subprocess.run([sys.executable, "-c", prelude + textwrap.dedent(body)], capture_output=True, text=True,
                          timeout=timeout, env=dict(os.environ, PYTHONPATH=src), check=False)


def test_extract_page_ignores_video_drop_and_start_while_busy_v323650() -> None:
    result = _run_window_script(
        """
        page = window._extract_page
        loaded = []
        page._view_model.load_video = lambda p: loaded.append(p)
        page._view_model._is_busy = True
        page._stt_install_thread = object()
        page._load_video(Path("/tmp/other.mp4"))
        page._on_start_clicked()
        print("LOADED=%d" % len(loaded))
        print("STT_KEPT=" + str(page._stt_install_thread is not None))
        page._on_video_loaded(type("M", (), dict(filename="a.mp4", width=1, height=1, fps=25.0, duration_str="0:01",
                                                 path=Path("/tmp/a.mp4")))())
        print("START_ENABLED=" + str(page._start_button.isEnabled()))
        """
    )
    assert result.returncode == 0, f"exit={result.returncode}\n{result.stderr[-1500:]}"
    assert "LOADED=0" in result.stdout and "STT_KEPT=True" in result.stdout and "START_ENABLED=False" in result.stdout


def test_keeping_edits_does_not_adopt_new_video_as_current_project_v323650() -> None:
    result = _run_window_script(
        """
        window._current_video_hash = "OLD"
        window._editor_page.load_events = lambda events, **k: False        # người dùng chọn "giữ bản đang sửa"
        video = tmp / "new.mp4"; video.write_bytes(b"x" * 2048)
        meta = type("M", (), dict(path=video))()
        type(window._extract_page).current_video = property(lambda self: meta)
        suggested = []
        window._tts_page.suggest_output_path = lambda p: suggested.append(p)
        window._translate_page.suggest_source_path = lambda p: suggested.append(p)
        ev = SubtitleEvent(index=1, text="x", interval=TimeInterval(0.0, 1.0), confidence=0.9, frame_count=1)
        window._on_extraction_completed([ev])
        print("HASH=" + window._current_video_hash)
        print("SUGGESTED=%d" % len(suggested))
        """
    )
    assert result.returncode == 0, f"exit={result.returncode}\n{result.stderr[-1500:]}"
    assert "HASH=OLD" in result.stdout and "SUGGESTED=0" in result.stdout


def test_idle_main_window_close_has_no_background_task_warning_v323650() -> None:
    """Luồng render sóng âm luôn chạy là hạ tầng, không phải "tác vụ nền": đóng app lúc rảnh không được hỏi xác nhận."""
    result = _run_window_script(
        """
        print("WARNINGS=%s" % window._collect_close_warnings())
        """
    )
    assert result.returncode == 0, f"exit={result.returncode}\n{result.stderr[-1500:]}"
    assert "WARNINGS=[]" in result.stdout


def test_close_warning_appears_when_a_page_holds_a_running_thread_v323650() -> None:
    result = _run_window_script(
        """
        import threading
        from PySide6.QtCore import QThread
        class T(QThread):
            def __init__(self, p):
                super().__init__(p); self.ev = threading.Event()
            def run(self):
                self.ev.wait(5)
        t = T(window._publish_page); window._publish_page._thread = t; t.start(); time.sleep(0.1)
        print("N=%d" % len(window._collect_close_warnings()))
        t.ev.set(); t.wait(3000)
        """
    )
    assert result.returncode == 0, f"exit={result.returncode}\n{result.stderr[-1500:]}"
    assert "N=1" in result.stdout
