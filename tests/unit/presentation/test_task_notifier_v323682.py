"""[v3.23.682] Báo việc dài xong khi người dùng đang ở cửa sổ khác (nháy thanh tác vụ + bong bóng khay)."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QSystemTrayIcon, QWidget

from subtitles_extractor.presentation.utils import task_notifier


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


class _FakeTray:
    def __init__(self) -> None:
        self.messages: list[tuple] = []

    def showMessage(self, *args) -> None:  # noqa: N802 — API Qt
        self.messages.append(args)


def test_notifies_only_when_window_inactive(app: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    alerts: list[object] = []
    tray = _FakeTray()
    monkeypatch.setattr(QApplication, "alert", staticmethod(lambda w, ms=0: alerts.append(w)))
    monkeypatch.setattr(task_notifier, "_tray", lambda _app: tray)
    widget = QWidget()
    monkeypatch.setattr(widget, "isActiveWindow", lambda: True)
    assert task_notifier.notify_task_finished(widget, "Xong", "12 câu") is False      # đang nhìn ứng dụng → im lặng
    monkeypatch.setattr(widget, "isActiveWindow", lambda: False)
    assert task_notifier.notify_task_finished(widget, "Trích xuất xong", "3657 câu") is True
    assert task_notifier.notify_task_finished(widget, "Lỗi", "x" * 1000, failed=True) is True
    assert alerts == [widget, widget]
    (t1, m1, icon1, _ms), (_t2, m2, icon2, _ms2) = tray.messages
    assert (t1, m1, icon1) == ("Trích xuất xong", "3657 câu", QSystemTrayIcon.MessageIcon.Information)
    assert icon2 == QSystemTrayIcon.MessageIcon.Warning and len(m2) == 300
    widget.deleteLater()


def test_no_tray_still_alerts_and_never_raises(app: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    alerts: list[object] = []
    monkeypatch.setattr(QApplication, "alert", staticmethod(lambda w, ms=0: alerts.append(w)))
    monkeypatch.setattr(QSystemTrayIcon, "isSystemTrayAvailable", staticmethod(lambda: False))
    widget = QWidget()
    monkeypatch.setattr(widget, "isActiveWindow", lambda: False)
    assert task_notifier.notify_task_finished(widget, "Xong", "") is True and alerts == [widget]
    assert task_notifier.notify_task_finished(None, "Xong", "") is False
    widget.deleteLater()


def test_pages_use_translated_titles() -> None:
    import json
    from pathlib import Path

    data_dir = Path(task_notifier.__file__).resolve().parents[2] / "data"
    for lang in ("vi", "en"):
        common = json.loads((data_dir / f"strings_{lang}.json").read_text("utf-8"))["common"]
        for key in ("task_done_extract", "task_failed_extract", "task_done_translate", "task_done_tts",
                    "task_done_publish", "task_failed_publish", "task_lines"):
            assert common[key]
        assert "{count}" in common["task_lines"]
