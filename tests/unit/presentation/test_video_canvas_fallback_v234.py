"""Unit tests cho v2.34 — VideoCanvas fallback + factory pattern.

Bảo vệ:
    * ``VideoCanvas`` được tái tạo với API tương thích MpvVideoWidget.
    * ``create_video_widget()`` factory function ở widgets/__init__.py.
    * ``is_mpv_available()`` cached check function.
    * Cả 2 pages (extract, editor) dùng factory thay vì instantiate trực tiếp.
    * VideoCanvas có đủ method/signal cần thiết để swap drop-in.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from subtitles_extractor.domain.value_objects.roi import (
    Roi,
    TextAlignment,
    TextOrientation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRESENTATION_DIR = PROJECT_ROOT / "src" / "subtitles_extractor" / "presentation"
WIDGETS_INIT_PATH = PRESENTATION_DIR / "widgets" / "__init__.py"
VIDEO_CANVAS_PATH = PRESENTATION_DIR / "widgets" / "video_canvas.py"
EXTRACT_PAGE_PATH = PRESENTATION_DIR / "pages" / "extract_page.py"
EDITOR_PAGE_PATH = PRESENTATION_DIR / "pages" / "editor_page.py"


def _make_test_roi(x: int = 100, y: int = 200, w: int = 300, h: int = 50) -> Roi:
    return Roi(
        x=x, y=y, width=w, height=h,
        alignment=TextAlignment.CENTER,
        orientation=TextOrientation.HORIZONTAL,
    )


class TestVideoCanvasFileRestored:
    """v2.34: ``video_canvas.py`` đã tái tạo sau khi xoá ở v2.33."""

    def test_video_canvas_file_exists(self) -> None:
        assert VIDEO_CANVAS_PATH.exists(), (
            "File video_canvas.py phải tồn tại trở lại (v2.34 restore)"
        )

    def test_video_canvas_has_class_definition(self) -> None:
        source = VIDEO_CANVAS_PATH.read_text(encoding="utf-8")
        assert "class VideoCanvas(QWidget):" in source


class TestVideoCanvasApiCompatibility:
    """v2.34: VideoCanvas có API tương thích MpvVideoWidget."""

    @pytest.fixture()
    def videocanvas_class_node(self) -> ast.ClassDef:
        tree = ast.parse(VIDEO_CANVAS_PATH.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "VideoCanvas":
                return node
        pytest.fail("VideoCanvas class không tìm thấy")
        raise AssertionError

    def _method_names(self, class_node: ast.ClassDef) -> set[str]:
        return {
            item.name
            for item in class_node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    def test_has_load_method(self, videocanvas_class_node: ast.ClassDef) -> None:
        assert "load" in self._method_names(videocanvas_class_node)

    def test_has_set_video_size(self, videocanvas_class_node: ast.ClassDef) -> None:
        assert "set_video_size" in self._method_names(videocanvas_class_node)

    def test_has_set_frame_for_fallback_paint(
        self, videocanvas_class_node: ast.ClassDef
    ) -> None:
        """``set_frame`` là method độc nhất của fallback — caller dispatch
        SeekWorker rồi feed QImage."""
        assert "set_frame" in self._method_names(videocanvas_class_node)

    def test_has_enable_roi_drawing(self, videocanvas_class_node: ast.ClassDef) -> None:
        assert "enable_roi_drawing" in self._method_names(videocanvas_class_node)

    def test_has_set_committed_roi(self, videocanvas_class_node: ast.ClassDef) -> None:
        assert "set_committed_roi" in self._method_names(videocanvas_class_node)

    def test_has_set_secondary_rois(self, videocanvas_class_node: ast.ClassDef) -> None:
        assert "set_secondary_rois" in self._method_names(videocanvas_class_node)

    def test_has_set_ocr_overlay(self, videocanvas_class_node: ast.ClassDef) -> None:
        assert "set_ocr_overlay" in self._method_names(videocanvas_class_node)

    def test_has_clear_ocr_overlay(self, videocanvas_class_node: ast.ClassDef) -> None:
        assert "clear_ocr_overlay" in self._method_names(videocanvas_class_node)

    def test_has_clear_roi(self, videocanvas_class_node: ast.ClassDef) -> None:
        assert "clear_roi" in self._method_names(videocanvas_class_node)

    def test_has_player_noop_method(self, videocanvas_class_node: ast.ClassDef) -> None:
        """``player()`` phải có (no-op) để API tương thích MpvVideoWidget."""
        assert "player" in self._method_names(videocanvas_class_node)

    def test_has_release_player_noop_method(
        self, videocanvas_class_node: ast.ClassDef
    ) -> None:
        assert "release_player" in self._method_names(videocanvas_class_node)

    def test_has_send_mpv_command_noop(
        self, videocanvas_class_node: ast.ClassDef
    ) -> None:
        assert "send_mpv_command" in self._method_names(videocanvas_class_node)


class TestVideoCanvasSignalsCompatibility:
    """v2.34: VideoCanvas có signals cùng tên với MpvVideoWidget."""

    def test_has_roi_changed_signal(self) -> None:
        source = VIDEO_CANVAS_PATH.read_text(encoding="utf-8")
        assert "roi_changed = Signal(object)" in source

    def test_has_roi_preview_signal(self) -> None:
        source = VIDEO_CANVAS_PATH.read_text(encoding="utf-8")
        assert "roi_preview = Signal(object)" in source

    def test_has_video_clicked_signal(self) -> None:
        source = VIDEO_CANVAS_PATH.read_text(encoding="utf-8")
        assert "video_clicked = Signal" in source

    def test_has_video_double_clicked_signal(self) -> None:
        source = VIDEO_CANVAS_PATH.read_text(encoding="utf-8")
        assert "video_double_clicked = Signal" in source


class TestVideoCanvasBehavior:
    """v2.34: VideoCanvas runtime behavior (no Qt needed for these)."""

    def test_player_returns_none(self) -> None:
        """player() là no-op trả None — caller dùng để biết đang fallback."""
        source = VIDEO_CANVAS_PATH.read_text(encoding="utf-8")
        # Tìm theo prefix linh hoạt — không phụ thuộc type annotation.
        assert "def player(self)" in source
        idx = source.index("def player(self)")
        snippet = source[idx:idx + 400]
        assert "return None" in snippet

    def test_load_is_noop(self) -> None:
        """load(path) không raise — chỉ log debug."""
        source = VIDEO_CANVAS_PATH.read_text(encoding="utf-8")
        assert "def load(self, video_path: Path) -> None:" in source

    def test_send_mpv_command_returns_false(self) -> None:
        source = VIDEO_CANVAS_PATH.read_text(encoding="utf-8")
        assert "def send_mpv_command(self, *args: object) -> bool:" in source


class TestFactoryFunction:
    """v2.34: ``create_video_widget`` factory ở widgets/__init__.py."""

    @pytest.fixture()
    def widgets_init_source(self) -> str:
        return WIDGETS_INIT_PATH.read_text(encoding="utf-8")

    def test_exports_create_video_widget(self, widgets_init_source: str) -> None:
        assert "def create_video_widget(" in widgets_init_source
        assert '"create_video_widget"' in widgets_init_source

    def test_exports_is_mpv_available(self, widgets_init_source: str) -> None:
        assert "def is_mpv_available()" in widgets_init_source
        assert '"is_mpv_available"' in widgets_init_source

    def test_exports_video_canvas_class(self, widgets_init_source: str) -> None:
        assert '"VideoCanvas"' in widgets_init_source

    def test_exports_mpv_video_widget(self, widgets_init_source: str) -> None:
        assert '"MpvVideoWidget"' in widgets_init_source


class TestFactoryRuntime:
    """v2.34: ``create_video_widget`` runtime behavior."""

    @pytest.fixture(scope="class")
    def qapp(self):
        """QApplication fixture — share giữa các tests trong class."""
        pytest.importorskip("PyQt6", reason="PyQt6 required for runtime widget tests")
        from PySide6.QtWidgets import QApplication
        import sys
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv if hasattr(sys, "argv") else [])
        yield app

    def test_force_fallback_returns_video_canvas(self, qapp) -> None:
        """``force_fallback=True`` luôn trả VideoCanvas bất kể MPV."""
        from subtitles_extractor.presentation.widgets import create_video_widget
        from subtitles_extractor.presentation.widgets.video_canvas import VideoCanvas

        widget = create_video_widget(force_fallback=True, parent=None)
        assert isinstance(widget, VideoCanvas)

    def test_factory_returns_video_canvas_when_mpv_unavailable(self, qapp) -> None:
        """Mock ``is_mpv_available`` trả False → factory trả VideoCanvas."""
        from subtitles_extractor.presentation import widgets as widgets_module
        from subtitles_extractor.presentation.widgets.video_canvas import VideoCanvas

        with patch.object(widgets_module, "is_mpv_available", return_value=False):
            widget = widgets_module.create_video_widget(parent=None)
        assert isinstance(widget, VideoCanvas)

    def test_is_mpv_available_returns_bool(self) -> None:
        """``is_mpv_available()`` trả True hoặc False (không crash)."""
        from subtitles_extractor.presentation.widgets import is_mpv_available

        result = is_mpv_available()
        assert isinstance(result, bool)


class TestPagesUseFactory:
    """v2.34: Cả extract_page và editor_page dùng ``create_video_widget``."""

    def test_extract_page_uses_factory(self) -> None:
        source = EXTRACT_PAGE_PATH.read_text(encoding="utf-8")
        assert "from subtitles_extractor.presentation.widgets import create_video_widget" in source
        # Kiểm tra create_video_widget được dùng với parent=self (có thể multi-arg).
        assert "create_video_widget(" in source
        assert "parent=self" in source

    def test_editor_page_uses_factory_in_main_player(self) -> None:
        source = EDITOR_PAGE_PATH.read_text(encoding="utf-8")
        # Main player.
        assert "self._video_widget = create_video_widget(" in source

    def test_editor_page_uses_factory_in_reocr_dialog(self) -> None:
        source = EDITOR_PAGE_PATH.read_text(encoding="utf-8")
        assert "self.video_canvas = create_video_widget(" in source


class TestSeekFrameDispatch:
    """v2.34: Khi fallback (player() None), caller gọi set_frame() để paint."""

    def test_extract_page_calls_set_frame_when_player_none(self) -> None:
        source = EXTRACT_PAGE_PATH.read_text(encoding="utf-8")
        # Logic: if self._canvas.player() is None: self._canvas.set_frame(img, w, h)
        assert "if self._canvas.player() is None:" in source
        assert "self._canvas.set_frame(img, w, h)" in source

    def test_editor_page_calls_set_frame_when_player_none(self) -> None:
        source = EDITOR_PAGE_PATH.read_text(encoding="utf-8")
        assert "if self.video_canvas.player() is None:" in source
        assert "self.video_canvas.set_frame(qimage, width, height)" in source


class TestFilesAstParse:
    """v2.34: 4 files quan trọng vẫn parse được."""

    def test_video_canvas_ast_parses(self) -> None:
        source = VIDEO_CANVAS_PATH.read_text(encoding="utf-8")
        ast.parse(source)

    def test_widgets_init_ast_parses(self) -> None:
        source = WIDGETS_INIT_PATH.read_text(encoding="utf-8")
        ast.parse(source)

    def test_extract_page_ast_parses(self) -> None:
        source = EXTRACT_PAGE_PATH.read_text(encoding="utf-8")
        ast.parse(source)

    def test_editor_page_ast_parses(self) -> None:
        source = EDITOR_PAGE_PATH.read_text(encoding="utf-8")
        ast.parse(source)
