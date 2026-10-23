"""[v3.23.613] Worker chạy trên QThread chỉ có MỘT chủ sở hữu — hết sập khi tua video.

Mẫu cũ dọn worker hai đường cùng lúc (``thread.finished → worker.deleteLater`` trong luồng
nền VÀ Python thả tham chiếu ở luồng giao diện) → xoá hai lần. Đo trên trang Trích xuất
(khung vẽ dự phòng khi không có MPV): tua liên tục 400 lần sập 4/5 lần chạy
(Segmentation fault / Bus error); gdb chỉ đúng ``QObject::~QObject`` trong luồng tua.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import QApplication

import subtitles_extractor
from subtitles_extractor.presentation.workers.thread_keeper import WorkerThreadKeeper

_SRC = Path(subtitles_extractor.__file__).parent


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


class _Worker(QObject):
    done = Signal()

    def run(self) -> None:
        self.done.emit()


class TestKeeper:
    def test_releases_after_thread_exits(self, app: QApplication) -> None:
        keeper = WorkerThreadKeeper()
        for _ in range(50):
            thread = QThread()
            worker = _Worker()
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.done.connect(thread.quit)
            keeper.keep(thread, worker)
            thread.start()
            del thread, worker  # nơi gọi không còn giữ gì
        for _ in range(200):
            QApplication.processEvents()
            if keeper.kept_count == 0:
                break
            QThread.msleep(10)
        assert keeper.kept_count == 0


def test_no_worker_delete_later_on_thread_finished() -> None:
    """Không còn nơi nào gắn ``worker.deleteLater`` vào ``thread.finished``."""
    offenders: list[str] = []
    for path in sorted((_SRC / "presentation").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "connect"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "finished"
                and node.args
                and isinstance(node.args[0], ast.Attribute)
                and node.args[0].attr == "deleteLater"
            ):
                continue
            emitter = ast.unparse(node.func.value.value)
            target = ast.unparse(node.args[0].value)
            if "thread" in emitter.lower() and target != emitter:
                where = f"{path.relative_to(_SRC)}:{node.lineno}"
                offenders.append(f"{where} {emitter} → {target}")
    assert not offenders, "\n".join(offenders)


_STRESS = textwrap.dedent(
    """
    import os, sys, tempfile, time
    from pathlib import Path
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    app = QApplication([])
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import (
        JsonFileRepository,
    )
    from subtitles_extractor.presentation.pages.extract_page import ExtractPage
    import subtitles_extractor
    tmp = Path(tempfile.mkdtemp())
    container = ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp / "c.json"),
        i18n_data_dir=Path(subtitles_extractor.__file__).parent / "data",
        user_data_dir=tmp,
    )
    page = ExtractPage(container)
    page._load_video(Path(sys.argv[1]))
    deadline = time.time() + 10
    while page.current_video is None and time.time() < deadline:
        app.processEvents(); time.sleep(0.02)
    assert page.current_video is not None, "không nạp được video thử"
    state = {"n": 0}
    def tick():
        if state["n"] >= 400:
            QTimer.singleShot(300, app.quit)
            return
        state["n"] += 1
        page._pending_seek_sec = (state["n"] % 9) / 10
        page._do_seek()
        QTimer.singleShot(1, tick)
    QTimer.singleShot(0, tick)
    app.exec()
    print("OK", state["n"], flush=True)
    os._exit(0)
    """
)


def test_rapid_seeking_does_not_crash(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.fail("Cần ffmpeg để dựng video thử")
    video = tmp_path / "seek.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc=s=160x90:d=1:r=10", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         str(video)],
        check=True, timeout=60,
    )
    script = tmp_path / "stress.py"
    script.write_text(_STRESS, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(_SRC.parent)}
    for attempt in range(3):
        result = subprocess.run(
            [sys.executable, str(script), str(video)],
            capture_output=True, text=True, timeout=120, env=env, check=False,
        )
        assert result.returncode == 0 and "OK 400" in result.stdout, (
            f"lần {attempt + 1}: mã thoát {result.returncode}\n{result.stderr[-1500:]}"
        )
