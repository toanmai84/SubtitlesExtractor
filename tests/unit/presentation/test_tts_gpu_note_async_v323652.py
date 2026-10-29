"""[v3.23.652] Dò khả năng GPU của VieNeu (``pip list``, tới 120 s) không được chạy trên luồng giao diện."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from subtitles_extractor.composition.container import ApplicationContainer
from subtitles_extractor.infrastructure.settings.json_file_repository import JsonFileRepository
from subtitles_extractor.infrastructure.tts import vieneu_gpu_plan
from subtitles_extractor.presentation.pages import tts_page as tts_page_module

DATA_DIR = Path(tts_page_module.__file__).resolve().parents[2] / "data"


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_page_build_does_not_wait_for_pip(qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ini_path = tmp_path / "tts_page.ini"
    monkeypatch.setattr(tts_page_module, "QSettings",
                        lambda *_a: QSettings(str(ini_path), QSettings.Format.IniFormat))
    release = threading.Event()
    calls: list[str] = []

    def slow_plan(installed=None):
        calls.append(threading.current_thread().name)
        release.wait(5)
        return vieneu_gpu_plan.GpuTtsPlan(
            status=vieneu_gpu_plan.GpuTtsStatus.READY, python_exe="py", missing_packages=(),
            install_commands=(), download_estimate_gb=0.0, note="",
        )

    monkeypatch.setattr(vieneu_gpu_plan, "build_gpu_tts_plan", slow_plan)
    container = ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp_path / "config.json"),
        i18n_data_dir=DATA_DIR, user_data_dir=tmp_path,
    )
    started = time.monotonic()
    page = tts_page_module.TTSPage(container)
    page._vieneu_force_cpu.setChecked(False)
    page._refresh_vieneu_gpu_note()
    assert time.monotonic() - started < 3  # trước đây chặn tới khi pip trả lời
    release.set()
    deadline = time.monotonic() + 5
    while page._gpu_plan_cache is None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    assert page._gpu_plan_cache is not None
    assert "GPU" in page._vieneu_gpu_note.text()
    assert all(name != threading.main_thread().name for name in calls)
    n = len(calls)
    page._vieneu_force_cpu.setChecked(True)
    page._vieneu_force_cpu.setChecked(False)
    assert len(calls) == n  # dùng kết quả đã nhớ, không dò lại
