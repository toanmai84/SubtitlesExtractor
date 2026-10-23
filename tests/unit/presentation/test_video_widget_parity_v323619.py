"""[v3.23.619] Hai widget video (MPV và VideoCanvas dự phòng) phải có cùng API trang dùng.

Lỗi tìm được khi rà soát (mypy ``union-attr``):
* Trang Biên tập — nút "Tự sửa lỗi timeline" gọi ``_video_widget._schedule_geometry_update()``
  (chỉ MPV có) → máy không có libmpv (dùng VideoCanvas) sửa xong thì ném AttributeError.
* Trang Trích xuất — tua video khi MPV khởi tạo lỗi (``player()`` trả None) gọi
  ``set_frame`` trên MpvVideoWidget (không có) → AttributeError mỗi lần tua.
* ``MpvVideoWidget.player()`` không bắt ``OSError`` (python-mpv có nhưng thiếu libmpv).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QMessageBox

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval


@pytest.fixture()
def app() -> QApplication:
    return QApplication.instance() or QApplication([])




@pytest.mark.parametrize("method", ["set_frame", "schedule_overlay_refresh", "player"])
def test_both_video_widgets_share_page_api(app: QApplication, method: str) -> None:
    from subtitles_extractor.presentation.widgets.mpv_video_widget import MpvVideoWidget
    from subtitles_extractor.presentation.widgets.video_canvas import VideoCanvas

    assert callable(getattr(MpvVideoWidget, method, None)), f"MpvVideoWidget thiếu {method}"
    assert callable(getattr(VideoCanvas, method, None)), f"VideoCanvas thiếu {method}"


def test_mpv_widget_paints_frame_when_mpv_failed(
    app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    from subtitles_extractor.infrastructure.video import mpv_player_adapter
    from subtitles_extractor.presentation.widgets.mpv_video_widget import MpvVideoWidget

    def broken(*_a: object, **_k: object) -> None:
        raise OSError("cannot load library libmpv-2.dll")

    monkeypatch.setattr(mpv_player_adapter, "MpvPlayerAdapter", broken)
    widget = MpvVideoWidget(mpv_options={})
    assert widget.player() is None  # OSError được bắt, không crash
    image = QImage(64, 36, QImage.Format.Format_RGB32)
    image.fill(0xFF3366)
    widget.set_frame(image, 64, 36)
    assert widget._overlay._background_frame is not None
    widget.schedule_overlay_refresh()


def test_extract_seek_with_failed_mpv_does_not_crash(
    app: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from subtitles_extractor.infrastructure.video import mpv_player_adapter
    from subtitles_extractor.presentation.pages.extract_page import ExtractPage
    from subtitles_extractor.presentation.widgets.mpv_video_widget import MpvVideoWidget

    monkeypatch.setattr(
        mpv_player_adapter, "MpvPlayerAdapter",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("mpv init lỗi")),
    )
    page = ExtractPage(_container(tmp_path))
    page._canvas = MpvVideoWidget(mpv_options={})
    image = QImage(64, 36, QImage.Format.Format_RGB32)
    page._on_seek_frame_ready(image, 64, 36)  # trước đây: AttributeError set_frame
    page.close()


def test_editor_auto_fix_timeline_with_fallback_canvas(
    app: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from subtitles_extractor.presentation.pages.editor_page import EditorPage
    from subtitles_extractor.presentation.widgets.video_canvas import VideoCanvas

    page = EditorPage(_container(tmp_path))
    assert isinstance(page._video_widget, VideoCanvas)  # máy không có libmpv
    overlapping = [
        SubtitleEvent(index=1, text="一", interval=TimeInterval(0.0, 2.0)),
        SubtitleEvent(index=2, text="二", interval=TimeInterval(1.5, 3.0)),
    ]
    page.load_events(overlapping, confirm=False)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *_a, **_k: QMessageBox.StandardButton.Yes
    )
    page._on_auto_fix_timeline()  # trước đây: AttributeError _schedule_geometry_update
    events = page._view_model.current_events
    assert events[0].end_sec <= events[1].start_sec
    page.close()


def _container(tmp: Path) -> Any:
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import (
        JsonFileRepository,
    )
    from subtitles_extractor.presentation.pages import editor_page

    return ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp / "c.json"),
        i18n_data_dir=Path(editor_page.__file__).resolve().parents[2] / "data",
        user_data_dir=tmp,
    )


def test_deleting_waveform_widget_stops_its_threads() -> None:
    """Huỷ widget sóng âm không qua close_widget(): trước đây Qt abort() cả tiến trình
    ("QThread: Destroyed while thread is still running"). Chạy ở tiến trình con vì abort
    giết luôn tiến trình test."""
    import os
    import subprocess
    import sys

    code = (
        "import shiboken6\n"
        "from PySide6.QtWidgets import QApplication\n"
        "app = QApplication([])\n"
        "from subtitles_extractor.presentation.widgets.audio_waveform_widget import "
        "AudioWaveformWidget\n"
        "w = AudioWaveformWidget()\n"
        "assert w._active_render_worker.isRunning()\n"
        "shiboken6.delete(w)\n"
        "print('OK')\n"
    )
    src = Path(__file__).resolve().parents[3] / "src"
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(src)}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env=env
    )
    assert result.returncode == 0, result.stderr[-500:]
    assert "OK" in result.stdout


def test_pages_only_use_members_both_video_widgets_have() -> None:
    """Quét tĩnh: trang giữ widget từ ``create_video_widget`` chỉ được dùng thành viên mà CẢ
    HAI lớp đều có. Bấm thử giao diện tìm thêm: nút "⧉ Tách (F11)" truyền
    ``_video_widget._schedule_geometry_update`` cho QTimer → AttributeError với VideoCanvas."""
    import ast
    import re

    root = Path(__file__).resolve().parents[3] / "src" / "subtitles_extractor" / "presentation"

    def members(path: Path, cls: str) -> set[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls:
                names = {f.name for f in node.body if isinstance(f, ast.FunctionDef)}
                for sub in ast.walk(node):
                    if (isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name)
                            and sub.value.id == "self" and isinstance(sub.ctx, ast.Store)):
                        names.add(sub.attr)
                return names
        raise AssertionError(cls)

    mpv = members(root / "widgets" / "mpv_video_widget.py", "MpvVideoWidget")
    canvas = members(root / "widgets" / "video_canvas.py", "VideoCanvas")
    one_sided = mpv ^ canvas
    offenders = []
    for page in root.rglob("*.py"):
        source = page.read_text(encoding="utf-8")
        for holder in set(re.findall(r"self\.(\w+)\s*=\s*create_video_widget\(", source)):
            for match in re.finditer(rf"self\.{holder}\.(\w+)", source):
                if match.group(1) in one_sided:
                    line = source[: match.start()].count("\n") + 1
                    offenders.append(f"{page.name}:{line} {holder}.{match.group(1)}")
    assert offenders == [], offenders


def test_editor_detach_and_reattach_with_fallback_canvas(
    app: QApplication, tmp_path: Path
) -> None:
    from subtitles_extractor.presentation.pages.editor_page import EditorPage

    page = EditorPage(_container(tmp_path))
    page._toggle_detach()  # tách ra cửa sổ riêng
    page._toggle_detach()  # gộp lại → trước đây AttributeError khi gộp
    QApplication.processEvents()
    page.close()


def test_dispose_detached_window_closes_it_for_good(app: QApplication, tmp_path: Path) -> None:
    """Đóng ứng dụng khi video đang tách (F11): trước đây ứng dụng KHÔNG thoát (còn cửa sổ
    video), bấm X tiếp thì chạy ngầm không cửa sổ."""
    from subtitles_extractor.presentation.pages.editor_page import EditorPage

    page = EditorPage(_container(tmp_path))
    page._toggle_detach()
    window = page._detached_window
    assert window.isVisible()
    page.dispose_detached_window()
    QApplication.processEvents()
    assert page._detached_window is None
    assert page._video_widget.parent() is not None  # video đã về lại trang
    page.close()


def test_detached_window_closes_after_page_is_gone(app: QApplication, tmp_path: Path) -> None:
    import shiboken6

    from subtitles_extractor.presentation.pages.editor_page import EditorPage

    page = EditorPage(_container(tmp_path))
    page._toggle_detach()
    window = page._detached_window
    shiboken6.delete(page)
    window.close()  # trước đây: RuntimeError (gắn video vào trang đã huỷ) + bỏ qua lệnh đóng
    QApplication.processEvents()
    assert not window.isVisible()
    shiboken6.delete(window)
