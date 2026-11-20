"""[v3.23.736] Giao diện hiện nguyên tên khoá chuỗi ("common.video_placeholder" giữa khung video trang Trích xuất).

Nguyên nhân: ``create_video_widget`` nhận ``translator`` tuỳ chọn; trang Trích xuất và Biên tập gọi KHÔNG truyền →
``MpvVideoWidget`` dùng ``NullTranslator`` (trả nguyên khoá). Bộ test chạy máy không có libmpv nên chỉ dựng
``VideoCanvas`` (không có chữ) — lỗi không bao giờ lộ. Quét lúc chạy: dựng cửa sổ chính với MPV "có mặt" (khởi tạo
lỗi để khỏi cần libmpv), duyệt MỌI widget của MỌI trang (cả trang đang ẩn), không chữ hiển thị nào được là tên khoá.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src"

SCRIPT = textwrap.dedent(
    """
    import json, os, sys, tempfile
    from pathlib import Path
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import (QAbstractButton, QApplication, QComboBox, QGroupBox, QLabel, QLineEdit,
                                   QMessageBox, QPlainTextEdit, QTabWidget, QTextEdit, QWidget)
    app = QApplication([])
    for name in ("question", "information", "warning", "critical"):
        setattr(QMessageBox, name, staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    import subtitles_extractor.presentation.widgets as widgets
    from subtitles_extractor.infrastructure.video import mpv_player_adapter
    widgets.is_mpv_available = lambda: True                 # máy người dùng: có libmpv → MpvVideoWidget
    def _no_mpv(*a, **k):
        raise OSError("libmpv không có trong môi trường test")
    mpv_player_adapter.MpvPlayerAdapter = _no_mpv
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import JsonFileRepository
    from subtitles_extractor.presentation.main_window import MainWindow
    from subtitles_extractor.presentation.pages import editor_page as ep
    ApplicationContainer.preload_ocr_engine_async = lambda self: None
    data = Path(ep.__file__).resolve().parents[2] / "data"
    def flat(d, p=""):
        out = set()
        for k, v in d.items():
            key = f"{p}.{k}" if p else k
            out |= flat(v, key) if isinstance(v, dict) else {key}
        return out
    keys = flat(json.loads((data / "strings_vi.json").read_text(encoding="utf-8")))
    tmp = Path(tempfile.mkdtemp(prefix="i18n736_"))
    container = ApplicationContainer(settings_repository=JsonFileRepository(file_path=tmp / "c.json"),
                                     i18n_data_dir=data, user_data_dir=tmp)
    window = MainWindow(container)
    found = []
    def texts(w):
        out = [w.toolTip(), w.windowTitle(), w.statusTip(), w.accessibleName()]
        if isinstance(w, (QLabel, QAbstractButton)): out.append(w.text())
        if isinstance(w, (QLineEdit, QPlainTextEdit, QTextEdit)): out.append(w.placeholderText())
        if isinstance(w, QGroupBox): out.append(w.title())
        if isinstance(w, QComboBox): out += [w.itemText(i) for i in range(w.count())]
        if isinstance(w, QTabWidget): out += [w.tabText(i) for i in range(w.count())]
        return out
    for w in [window, *window.findChildren(QWidget)]:
        for text in texts(w):
            for piece in (text or "").replace("\\n", " ").split():
                if piece.strip("…:()") in keys:
                    found.append(f"{type(w).__name__}: {text!r}")
    print("RESULT" + json.dumps(sorted(set(found)), ensure_ascii=False))
    window.close()
    """
)


def test_no_raw_i18n_key_visible_in_any_page() -> None:
    result = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=300,
                            env=dict(os.environ, PYTHONPATH=str(SRC), QT_QPA_PLATFORM="offscreen"), check=False)
    lines = [line for line in result.stdout.splitlines() if line.startswith("RESULT")]
    assert lines, result.stderr[-2000:]
    assert json.loads(lines[-1][len("RESULT"):]) == []


def test_video_widget_factory_requires_translator() -> None:
    import inspect

    from subtitles_extractor.presentation.widgets import create_video_widget

    assert inspect.signature(create_video_widget).parameters["translator"].default is inspect.Parameter.empty
