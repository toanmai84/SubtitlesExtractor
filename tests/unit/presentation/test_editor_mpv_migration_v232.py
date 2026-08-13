"""Unit tests cho v2.32 — Migrate editor Re-OCR dialog sang MPV widget.

Bảo vệ:
    * ``AdvancedReOcrDialog`` đã import ``RoiMpvVideoWidget`` thay VideoCanvas.
    * Inline class playback widget được rename thành ``_RightPaneMpvVideoWidget``.
    * Không còn API cũ ``set_roi_rect``, ``set_frame``, ``set_roi_mode`` trong
      code (đã đổi sang ``set_committed_roi``, ``set_video_size``,
      ``enable_roi_drawing``).
    * MPV player được ``release`` khi dialog đóng.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EDITOR_PAGE_PATH = (
    PROJECT_ROOT / "src" / "subtitles_extractor" / "presentation"
    / "pages" / "editor_page.py"
)


@pytest.fixture()
def editor_source() -> str:
    return EDITOR_PAGE_PATH.read_text(encoding="utf-8")


@pytest.fixture()
def editor_tree(editor_source: str) -> ast.Module:
    return ast.parse(editor_source)


class TestEditorImportsMpvWidget:
    """v2.32: Editor import MpvVideoWidget từ widgets shared."""

    def test_imports_roi_mpv_widget_with_alias(self, editor_source: str) -> None:
        """Import as ``RoiMpvVideoWidget`` để tránh trùng tên class inline."""
        assert (
            "from subtitles_extractor.presentation.widgets.mpv_video_widget import"
            in editor_source
        )
        assert "MpvVideoWidget as RoiMpvVideoWidget" in editor_source

    def test_does_not_import_video_canvas(self, editor_tree: ast.Module) -> None:
        """v2.32: editor không còn import VideoCanvas."""
        for node in ast.walk(editor_tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    assert alias.name != "VideoCanvas", (
                        f"VideoCanvas vẫn import từ {node.module}"
                    )


class TestInlineClassRenamed:
    """v2.32 → v2.33: Inline playback class đã được xoá hoàn toàn.

    v2.32 chỉ rename ``MpvVideoWidget`` → ``_RightPaneMpvVideoWidget``. v2.33
    đã **xoá** hẳn class nội bộ này (cùng với ``_NativeVideoContainer`` và
    ``_ClickCatcherOverlay``) — main video player của ``EditorPage`` chuyển
    sang dùng shared ``RoiMpvVideoWidget``.
    """

    def test_right_pane_class_no_longer_exists(self, editor_source: str) -> None:
        """v2.33: ``_RightPaneMpvVideoWidget`` đã xoá khỏi định nghĩa class."""
        assert "class _RightPaneMpvVideoWidget(QWidget):" not in editor_source

    def test_no_duplicate_mpv_video_widget_class(self, editor_tree: ast.Module) -> None:
        """Không có class trùng tên ``MpvVideoWidget`` ở module-level."""
        for node in editor_tree.body:
            if isinstance(node, ast.ClassDef):
                # ``MpvVideoWidget`` chỉ được dùng như alias import, không
                # khai báo class trùng.
                assert node.name != "MpvVideoWidget", (
                    f"Class inline ``MpvVideoWidget`` vẫn tồn tại — chưa rename"
                )


class TestOldVideoCanvasApiRemoved:
    """v2.32: API VideoCanvas cũ không còn dùng trên ``video_canvas``."""

    def test_no_set_frame_call(self, editor_source: str) -> None:
        """v2.34 NOTE: ``set_frame`` được gọi LẠI cho fallback path khi MPV
        không khả dụng. Test verify rằng có guard ``if player() is None``."""
        # ``set_frame`` chỉ được gọi trong fallback block.
        assert "if self.video_canvas.player() is None:" in editor_source
        # Verify set_frame nằm trong fallback block.
        lines = editor_source.splitlines()
        for i, line in enumerate(lines):
            if "if self.video_canvas.player() is None:" in line:
                window = "\n".join(lines[i:i + 4])
                assert "self.video_canvas.set_frame(" in window
                break
        else:
            pytest.fail("Không tìm thấy fallback guard với set_frame")

    def test_no_set_roi_rect_call(self, editor_source: str) -> None:
        assert "video_canvas.set_roi_rect" not in editor_source

    def test_no_set_roi_mode_call(self, editor_source: str) -> None:
        assert "video_canvas.set_roi_mode" not in editor_source

    def test_no_roi_selected_signal(self, editor_source: str) -> None:
        """Signal mới là ``roi_changed``, không phải ``roi_selected``."""
        assert "video_canvas.roi_selected" not in editor_source


class TestNewMpvApiUsed:
    """v2.32: editor dùng API mới của ``RoiMpvVideoWidget``."""

    def test_uses_enable_roi_drawing(self, editor_source: str) -> None:
        assert "video_canvas.enable_roi_drawing" in editor_source

    def test_uses_set_committed_roi(self, editor_source: str) -> None:
        assert "video_canvas.set_committed_roi" in editor_source

    def test_uses_set_video_size(self, editor_source: str) -> None:
        assert "video_canvas.set_video_size" in editor_source

    def test_uses_roi_changed_signal(self, editor_source: str) -> None:
        assert "video_canvas.roi_changed" in editor_source

    def test_uses_video_canvas_load(self, editor_source: str) -> None:
        """``_init_player`` gọi ``video_canvas.load(path)`` cho MPV native."""
        assert "video_canvas.load(self.video_path)" in editor_source

    def test_uses_video_canvas_player_seek(self, editor_source: str) -> None:
        """Seek qua MPV player adapter (mpv_player.seek)."""
        assert "mpv_player.seek(" in editor_source


class TestMpvCleanupOnClose:
    """v2.32: ``_cleanup_services`` gọi ``release_player`` để free MPV decoder."""

    def test_cleanup_calls_release_player(self, editor_source: str) -> None:
        assert "video_canvas.release_player()" in editor_source


class TestRightPaneClassInstantiation:
    """v2.33: Main video player của ``EditorPage`` dùng ``RoiMpvVideoWidget``."""

    def test_main_video_widget_uses_roi_mpv_widget(self, editor_source: str) -> None:
        """v2.34: Main video player dùng ``create_video_widget`` factory thay
        cho instantiate trực tiếp — tự fallback VideoCanvas nếu MPV không OK."""
        # Sau v2.34, ``EditorPage._video_widget = create_video_widget(...)``.
        assert "self._video_widget = create_video_widget(" in editor_source

    def test_old_class_name_not_instantiated(self, editor_source: str) -> None:
        """Không còn instantiate ``_RightPaneMpvVideoWidget(`` (đã xoá v2.33)."""
        for line in editor_source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "_RightPaneMpvVideoWidget(" not in stripped, (
                f"Instantiation ``_RightPaneMpvVideoWidget(`` (đã xoá) vẫn còn: {line!r}"
            )


class TestSyntaxValid:
    """v2.32: File sau migrate vẫn parse được."""

    def test_editor_page_ast_parses(self) -> None:
        source = EDITOR_PAGE_PATH.read_text(encoding="utf-8")
        ast.parse(source)
