"""[v3.23.613] Rà lỗi toàn ứng dụng — các lỗi giao diện chỉ lộ ra khi CHẠY thật.

Mỗi lỗi dưới đây làm một nút/thao tác chết im lặng (PySide6 nuốt exception trong slot,
chỉ in traceback ra stderr):

* Trình chỉnh sửa — "Tách dòng": ``QInputDialog.getDouble`` của PySide6 6.11 nhận
  ``minValue``/``maxValue``, KHÔNG nhận ``min``/``max`` → TypeError, hộp thoại không mở.
* Trình chỉnh sửa — mở dự án/đồng bộ sau trích xuất truyền đường dẫn dạng ``str``;
  ``load_video`` gọi ``path.resolve()`` → AttributeError, video không hiện, sóng âm
  không vẽ.
* Trình chỉnh sửa — Re-OCR với khoảng thời gian rỗng gọi ``self.error_occurred`` không
  tồn tại → AttributeError thay vì báo lỗi.
* Trình chỉnh sửa — log "bấm Xuất tệp" dùng ``%d`` với loguru → ghi chữ "%d" thô.
* Trang Trích xuất — cài WhisperX xong gọi ``_update_action_states`` không tồn tại →
  nút Phiên âm vẫn tắt và hộp thoại "đã sẵn sàng" không hiện.
* Trang Trích xuất — tua video khi đã chọn vùng ROI gọi ``self.current_roi`` không tồn
  tại → khung ROI không vẽ lại.
* Trang Trích xuất — 5 dòng trạng thái thay ``"{{n}}"`` trong khi chuỗi giao diện ghi
  ``"{n}"`` → người dùng thấy "{n}", "{label}", "{msg}" thô.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QItemSelectionModel
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QDialog, QInputDialog

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.roi import Roi
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


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


def _events() -> list[SubtitleEvent]:
    return [
        SubtitleEvent(index=1, text="Xin chào", interval=TimeInterval(1.0, 3.0)),
        SubtitleEvent(index=2, text="Tạm biệt", interval=TimeInterval(4.0, 6.0)),
    ]


@pytest.fixture
def editor(app: QApplication, tmp_path: Path) -> Iterator[Any]:
    from subtitles_extractor.presentation.pages.editor_page import EditorPage

    page = EditorPage(_container(tmp_path))
    yield page
    page._waveform_widget.close_widget()
    page.close()


@pytest.fixture
def extract(app: QApplication, tmp_path: Path) -> Iterator[Any]:
    from subtitles_extractor.presentation.pages.extract_page import ExtractPage

    page = ExtractPage(_container(tmp_path))
    yield page
    page.close()


def _select_row(page: Any, row: int) -> None:
    index = page._proxy_model.index(row, 0)
    page._table.selectionModel().select(
        index,
        QItemSelectionModel.SelectionFlag.ClearAndSelect
        | QItemSelectionModel.SelectionFlag.Rows,
    )


def _tiny_video(tmp_path: Path) -> Path:
    """Video 1 giây (hình + tiếng) dựng bằng ffmpeg."""
    target = tmp_path / "phim thử.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.fail("Cần ffmpeg để dựng video thử")
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=c=black:s=160x90:d=1",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         str(target)],
        check=True, timeout=60,
    )
    return target


def _binding_signature(function: Any) -> str:
    """Chuỗi tham số của hàm PySide6.

    [v3.23.619] Đọc ``__doc__`` trước; PySide6 6.12 bỏ chữ ký khỏi ``__doc__`` (rỗng) và
    ``inspect.signature`` cũng không có → đọc tệp chữ ký chính thức ``QtWidgets.pyi`` đi
    kèm binding. Trước đây test ném ``IndexError`` trên 6.12 dù ứng dụng chạy đúng.
    """
    doc = function.__doc__ or ""
    if "(" in doc:
        return doc.split("(", 1)[1].split(")", 1)[0]
    import PySide6

    stub = Path(PySide6.__file__).with_name("QtWidgets.pyi").read_text(encoding="utf-8")
    match = re.search(rf"def {function.__name__}\(([^)]*)\)", stub)
    assert match, f"Không tìm thấy chữ ký {function.__name__} trong QtWidgets.pyi"
    return match.group(1)


def _binding_parameters(function: Any) -> list[str]:
    """Tên tham số (theo thứ tự) của hàm PySide6."""
    return re.findall(r"(\w+):", _binding_signature(function))


def _positional_only(function: Any) -> list[str]:
    """Tham số đứng trước dấu ``/`` (chỉ truyền theo vị trí) — rỗng nếu không có."""
    signature = _binding_signature(function)
    if "/" not in signature:
        return []
    return re.findall(r"(\w+):", signature.split("/", 1)[0])


class TestEditorSplit:
    def test_split_dialog_opens_and_splits(
        self, editor: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Gọi đúng tên tham số của ``QInputDialog.getDouble`` bản đang cài.

        Không mở hộp thoại thật: vòng sự kiện lồng của nó xử lý luôn các đối tượng chờ
        huỷ của test khác trong cùng phiên.
        """
        from subtitles_extractor.presentation.pages import editor_page

        parameters = _binding_parameters(QInputDialog.getDouble)
        assert parameters[:4] == ["parent", "title", "label", "value"]

        positional_only = set(_positional_only(QInputDialog.getDouble))

        def strict_get_double(*args: Any, **kwargs: Any) -> tuple[float, bool]:
            unknown = set(kwargs) - set(parameters)
            if unknown or set(kwargs) & positional_only:
                raise TypeError(f"getDouble(): unsupported keyword {sorted(unknown)}")
            bound = {**dict(zip(parameters, args, strict=False)), **kwargs}
            assert bound["minValue"] < bound["value"] < bound["maxValue"]
            return float(bound["value"]), True

        monkeypatch.setattr(
            editor_page.QInputDialog, "getDouble", staticmethod(strict_get_double)
        )
        editor.load_events(_events(), confirm=False)
        _select_row(editor, 0)
        editor._on_split_clicked()
        events = editor._table_model.events
        assert len(events) == 3
        assert events[0].end_sec == pytest.approx(2.0, abs=1e-3)
        assert events[1].start_sec == pytest.approx(2.0, abs=1e-3)


class TestEditorLoadVideo:
    def test_accepts_string_path(self, editor: Any, tmp_path: Path) -> None:
        video = _tiny_video(tmp_path)
        editor.load_video(str(video))
        assert editor.get_current_video_path() == str(video)
        assert editor._video_stack.currentIndex() == 1



class _FakeEditor:
    def __init__(self, accept: bool) -> None:
        self.accept = accept
        self.videos: list[Any] = []

    def load_events(self, _events: list[SubtitleEvent]) -> bool:
        return self.accept

    def load_video(self, path: Any) -> None:
        self.videos.append(path)


def _open_project(accept: bool, tmp_path: Path) -> tuple[Any, _FakeEditor]:
    """Gọi ``MainWindow._open_project`` trên một cửa sổ giả (chỉ các trang cần dùng)."""
    from types import SimpleNamespace

    from subtitles_extractor.presentation.main_window import MainWindow

    editor = _FakeEditor(accept)
    suggested: list[str] = []
    window = SimpleNamespace(
        _editor_page=editor,
        _tts_page=SimpleNamespace(suggest_output_path=suggested.append),
        _translate_page=SimpleNamespace(suggest_source_path=suggested.append),
        _current_video_hash="dự án đang mở",
        switchTo=lambda _page: None,
        suggested=suggested,
    )
    record = SimpleNamespace(
        original_subtitle="1\n00:00:01,000 --> 00:00:02,000\nXin chào\n",
        video_path=str(tmp_path / "phim.mp4"),
        video_hash="dự án mới",
        video_name="phim.mp4",
    )
    MainWindow._open_project(window, record)  # type: ignore[arg-type]
    return window, editor


class TestOpenProject:
    def test_video_path_is_a_path(self, tmp_path: Path) -> None:
        """Đường dẫn lưu trong CSDL là chuỗi — trình chỉnh sửa nhận Path."""
        window, editor = _open_project(True, tmp_path)
        assert editor.videos == [tmp_path / "phim.mp4"]
        assert window._current_video_hash == "dự án mới"

    def test_keeping_current_edits_does_not_switch_project(self, tmp_path: Path) -> None:
        """Người dùng chọn GIỮ bản đang sửa → không đổi video, không đổi dự án hiện hành.

        Trước đây video và dự án hiện hành vẫn bị đổi sang dự án mới trong khi bảng phụ
        đề là của phim cũ: lồng tiếng/xuất bản sau đó ghi khâu vào nhầm dự án.
        """
        window, editor = _open_project(False, tmp_path)
        assert editor.videos == []
        assert window._current_video_hash == "dự án đang mở"
        assert window.suggested == []


class TestEditorReOcrBadRange:
    def test_empty_range_reports_error(
        self, editor: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from subtitles_extractor.presentation.pages import editor_page

        class _Dialog:
            def __init__(self, **_kwargs: Any) -> None: ...

            def exec(self) -> QDialog.DialogCode:
                return QDialog.DialogCode.Accepted

            def get_values(self) -> tuple[float, float, dict, None, None]:
                return 2.0, 2.0, {}, None, None  # đầu = cuối → khoảng rỗng

            def deleteLater(self) -> None: ...  # noqa: N802 — tên API Qt

        monkeypatch.setattr(editor_page, "AdvancedReOcrDialog", _Dialog)
        errors: list[str] = []
        monkeypatch.setattr(editor, "_show_error", errors.append)
        started: list[Any] = []
        monkeypatch.setattr(
            editor._view_model, "start_reocr", lambda **kw: started.append(kw)
        )
        editor.load_events(_events(), confirm=False)
        editor._current_video_path = tmp_path / "v.mp4"
        _select_row(editor, 0)
        editor._on_reocr_clicked()
        assert errors == [editor._translator.translate("editor.err_bad_range")]
        assert started == []


class TestEditorExportLog:
    def test_click_log_shows_real_values(
        self, editor: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from loguru import logger

        messages: list[str] = []
        sink = logger.add(lambda m: messages.append(m.record["message"]), level="DEBUG")
        monkeypatch.setattr(editor, "_do_export_clicked", lambda: None)
        try:
            editor.load_events(_events(), confirm=False)
            editor._on_export_clicked()
        finally:
            logger.remove(sink)
        clicked = [m for m in messages if "bấm Xuất tệp" in m]
        assert clicked == ["Editor: bấm Xuất tệp (events=2, is_busy=False)."]


def _fake_property(
    monkeypatch: pytest.MonkeyPatch, target: Any, name: str, value: Any
) -> None:
    monkeypatch.setattr(type(target), name, property(lambda _self: value))


class TestExtractSttRefresh:
    def test_refresh_after_install_enables_transcribe(
        self, extract: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from PySide6.QtWidgets import QMessageBox

        shown: list[str] = []
        monkeypatch.setattr(
            QMessageBox, "information", lambda *a, **_k: shown.append(a[1])
        )
        monkeypatch.setattr(extract._view_model, "is_stt_available", lambda: True)
        _fake_property(monkeypatch, extract._view_model, "video", object())
        extract._on_install_finished("/env/python")
        assert extract._btn_transcribe.isEnabled()
        assert extract._btn_install_stt.isHidden()
        assert extract._lbl_stt_hint.text() == extract._translator.translate(
            "extract.st_transcribe_hint"
        )
        assert shown == [extract._translator.translate("extract.dlg_wxdone_title")]

    def test_transcribe_stays_off_without_video(
        self, extract: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(extract._view_model, "is_stt_available", lambda: True)
        extract._refresh_stt_state()
        assert not extract._btn_transcribe.isEnabled()

    def test_transcribe_stays_off_while_busy(
        self, extract: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(extract._view_model, "is_stt_available", lambda: True)
        _fake_property(monkeypatch, extract._view_model, "video", object())
        _fake_property(monkeypatch, extract._view_model, "is_busy", True)
        extract._refresh_stt_state()
        assert not extract._btn_transcribe.isEnabled()


class TestExtractSeekRedrawsRoi:
    def test_committed_roi_is_redrawn(
        self, extract: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        roi = Roi(x=10, y=60, width=140, height=20)
        monkeypatch.setattr(type(extract._view_model), "roi", property(lambda _s: roi))
        drawn: list[Roi] = []
        monkeypatch.setattr(extract._canvas, "set_committed_roi", drawn.append)
        image = QImage(160, 90, QImage.Format.Format_RGB888)
        extract._on_seek_frame_ready(image, 160, 90)
        assert drawn == [roi]


class TestExtractStatusTexts:
    def test_install_progress_fills_label(self, extract: Any) -> None:
        extract._on_install_progress(40, "gói whisperx")
        assert extract._lbl_stt_hint.text() == extract._translator.translate(
            "extract.st_installing", label="gói whisperx"
        )
        assert "{" not in extract._lbl_stt_hint.text()

    def test_install_failure_fills_message(
        self, extract: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from PySide6.QtWidgets import QMessageBox

        monkeypatch.setattr(QMessageBox, "warning", lambda *_a, **_k: None)
        extract._on_install_failed("Hết dung lượng đĩa\nchi tiết pip")
        assert "Hết dung lượng đĩa" in extract._lbl_stt_hint.text()
        assert "{" not in extract._lbl_stt_hint.text()

    def test_install_failure_with_empty_message(
        self, extract: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from PySide6.QtWidgets import QMessageBox

        monkeypatch.setattr(QMessageBox, "warning", lambda *_a, **_k: None)
        extract._on_install_failed("")
        assert "{" not in extract._lbl_stt_hint.text()

    def test_transcribe_done_shows_count(self, extract: Any) -> None:
        extract._on_transcribe_finished(_events())
        assert extract._status_label.text() == extract._translator.translate(
            "extract.st_speech_done", n=2
        )
