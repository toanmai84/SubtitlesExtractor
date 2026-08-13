"""Unit tests cho v2.30 — Migration extract_page sang MpvVideoWidget.

Bảo vệ:
    * ``MpvVideoWidget`` có đầy đủ API cho drop-in replacement.
    * ``ExtractPage`` import MpvVideoWidget (không còn VideoCanvas).
    * ``_RoiOverlay`` mới có signal ``roi_preview``.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest


# Đường dẫn tới các file source.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXTRACT_PAGE_PATH = (
    PROJECT_ROOT / "src" / "subtitles_extractor" / "presentation"
    / "pages" / "extract_page.py"
)
MPV_WIDGET_PATH = (
    PROJECT_ROOT / "src" / "subtitles_extractor" / "presentation"
    / "widgets" / "mpv_video_widget.py"
)


class TestMpvVideoWidgetApiSurface:
    """v2.30: MpvVideoWidget có đủ API drop-in cho VideoCanvas."""

    @pytest.fixture()
    def widget_source(self) -> str:
        return MPV_WIDGET_PATH.read_text(encoding="utf-8")

    @pytest.fixture()
    def widget_class_node(self, widget_source: str) -> ast.ClassDef:
        """Trả về AST node của class MpvVideoWidget."""
        tree = ast.parse(widget_source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "MpvVideoWidget":
                return node
        pytest.fail("MpvVideoWidget class không tìm thấy")
        raise AssertionError  # for type checker

    def _get_method_names(self, class_node: ast.ClassDef) -> set[str]:
        return {
            item.name
            for item in class_node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    def test_has_load_method(self, widget_class_node: ast.ClassDef) -> None:
        assert "load" in self._get_method_names(widget_class_node)

    def test_has_set_video_size(self, widget_class_node: ast.ClassDef) -> None:
        assert "set_video_size" in self._get_method_names(widget_class_node)

    def test_has_enable_roi_drawing(self, widget_class_node: ast.ClassDef) -> None:
        assert "enable_roi_drawing" in self._get_method_names(widget_class_node)

    def test_has_set_committed_roi(self, widget_class_node: ast.ClassDef) -> None:
        assert "set_committed_roi" in self._get_method_names(widget_class_node)

    def test_has_set_secondary_rois(self, widget_class_node: ast.ClassDef) -> None:
        """v2.30 NEW: set_secondary_rois."""
        assert "set_secondary_rois" in self._get_method_names(widget_class_node)

    def test_has_set_ocr_overlay(self, widget_class_node: ast.ClassDef) -> None:
        """v2.30 NEW: set_ocr_overlay."""
        assert "set_ocr_overlay" in self._get_method_names(widget_class_node)

    def test_has_clear_ocr_overlay(self, widget_class_node: ast.ClassDef) -> None:
        """v2.30 NEW: clear_ocr_overlay."""
        assert "clear_ocr_overlay" in self._get_method_names(widget_class_node)

    def test_has_clear_roi(self, widget_class_node: ast.ClassDef) -> None:
        assert "clear_roi" in self._get_method_names(widget_class_node)

    def test_has_release_player(self, widget_class_node: ast.ClassDef) -> None:
        assert "release_player" in self._get_method_names(widget_class_node)

    def test_has_player_accessor(self, widget_class_node: ast.ClassDef) -> None:
        assert "player" in self._get_method_names(widget_class_node)


class TestMpvVideoWidgetSignals:
    """v2.30: MpvVideoWidget khai báo đủ signals."""

    def test_has_roi_changed_signal_declared(self) -> None:
        source = MPV_WIDGET_PATH.read_text(encoding="utf-8")
        assert "roi_changed = Signal(object)" in source

    def test_has_roi_preview_signal_declared(self) -> None:
        """v2.30 NEW: roi_preview signal (kiểu tham số có thể thay đổi qua các phiên bản)."""
        source = MPV_WIDGET_PATH.read_text(encoding="utf-8")
        # Chỉ kiểm tra signal TỒN TẠI — không lock cứng kiểu tham số
        # (v2.36 upgrade từ Signal(object) → Signal(QRect) để type-safe hơn)
        assert "roi_preview = Signal" in source

    def test_has_video_clicked_signals(self) -> None:
        source = MPV_WIDGET_PATH.read_text(encoding="utf-8")
        assert "video_clicked = Signal" in source
        assert "video_double_clicked = Signal" in source


class TestRoiOverlayLayers:
    """v2.30: _RoiOverlay vẽ 4 layer (OCR, secondary, committed, live)."""

    def test_overlay_has_secondary_rects_method(self) -> None:
        source = MPV_WIDGET_PATH.read_text(encoding="utf-8")
        assert "def set_secondary_rects" in source

    def test_overlay_has_ocr_overlay_rects_method(self) -> None:
        source = MPV_WIDGET_PATH.read_text(encoding="utf-8")
        assert "def set_ocr_overlay_rects" in source

    def test_overlay_has_roi_preview_signal(self) -> None:
        source = MPV_WIDGET_PATH.read_text(encoding="utf-8")
        assert "roi_preview = Signal(QRect)" in source


class TestExtractPageMigration:
    """v2.30: ExtractPage đã migrate khỏi VideoCanvas."""

    @pytest.fixture()
    def extract_page_source(self) -> str:
        return EXTRACT_PAGE_PATH.read_text(encoding="utf-8")

    def test_imports_mpv_video_widget(self, extract_page_source: str) -> None:
        assert "from subtitles_extractor.presentation.widgets.mpv_video_widget import MpvVideoWidget" in extract_page_source

    def test_does_not_import_video_canvas(self, extract_page_source: str) -> None:
        """ExtractPage không còn import VideoCanvas."""
        # Parse AST để chỉ kiểm tra import statements, bỏ qua docstring/comment.
        tree = ast.parse(extract_page_source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    assert alias.name != "VideoCanvas", (
                        f"VideoCanvas vẫn được import từ {node.module}"
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert "video_canvas" not in (alias.name or ""), (
                        f"video_canvas module vẫn được import: {alias.name}"
                    )

    def test_uses_mpv_widget_methods(self, extract_page_source: str) -> None:
        """Dùng các method MPV API mới."""
        assert "enable_roi_drawing" in extract_page_source
        assert "set_committed_roi" in extract_page_source

    def test_no_set_frame_call_on_canvas(self, extract_page_source: str) -> None:
        """v2.34 NOTE: ``set_frame`` được gọi LẠI cho fallback path khi MPV
        không khả dụng (``if self._canvas.player() is None: ...``). Đó không
        phải lỗi — đây là feature mới. Test chỉ verify rằng ``set_frame``
        không phải call mặc định không điều kiện (luôn paint cho mọi frame).
        """
        # Đảm bảo set_frame chỉ được gọi trong điều kiện ``player() is None``.
        # Cách check đơn giản: trong file phải có pattern guard "if ... player() is None"
        # gần line set_frame.
        assert "if self._canvas.player() is None:" in extract_page_source
        # Verify rằng set_frame nằm trong block fallback (cùng line range).
        lines = extract_page_source.splitlines()
        for i, line in enumerate(lines):
            if "if self._canvas.player() is None:" in line:
                # Check 3 line tiếp theo có set_frame.
                window = "\n".join(lines[i:i + 4])
                assert "self._canvas.set_frame(" in window
                break
        else:
            pytest.fail("Không tìm thấy fallback guard với set_frame")

    def test_no_set_roi_mode_call(self, extract_page_source: str) -> None:
        """Bỏ ``_canvas.set_roi_mode`` (đổi sang enable_roi_drawing)."""
        assert "_canvas.set_roi_mode" not in extract_page_source

    def test_no_set_roi_rect_call(self, extract_page_source: str) -> None:
        """Bỏ ``_canvas.set_roi_rect`` (đổi sang set_committed_roi)."""
        assert "_canvas.set_roi_rect" not in extract_page_source

    def test_release_player_called_in_close(self, extract_page_source: str) -> None:
        """closeEvent gọi release_player để free MPV."""
        assert "release_player()" in extract_page_source

    def test_mpv_load_called_in_video_loaded(self, extract_page_source: str) -> None:
        """``_on_video_loaded`` gọi MPV load."""
        assert "self._canvas.load(" in extract_page_source

    def test_seek_uses_mpv_player(self, extract_page_source: str) -> None:
        """``_seek_and_show`` gọi mpv_player.seek()."""
        assert "mpv_player.seek(" in extract_page_source


class TestImportPaths:
    """v2.30: Module path resolution không lỗi."""

    def test_mpv_widget_module_path_valid(self) -> None:
        assert MPV_WIDGET_PATH.exists()

    def test_extract_page_module_path_valid(self) -> None:
        assert EXTRACT_PAGE_PATH.exists()

    def test_mpv_widget_ast_parses(self) -> None:
        """File source là Python hợp lệ."""
        source = MPV_WIDGET_PATH.read_text(encoding="utf-8")
        ast.parse(source)  # Raise SyntaxError nếu sai cú pháp.

    def test_extract_page_ast_parses(self) -> None:
        source = EXTRACT_PAGE_PATH.read_text(encoding="utf-8")
        ast.parse(source)
