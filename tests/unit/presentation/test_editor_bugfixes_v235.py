"""Unit tests cho v2.35 — Bug fixes + UX improvements editor_page.

Bảo vệ các fix:
    * Bug 1: ``_on_text_editing`` đếm length không tính HTML tags / \\N.
    * Bug 2: ``_change_speed`` không có comparison redundant.
    * Bug 6: ``load_video`` show InfoBar.error nếu fail.
    * Bug 7: ``_on_video_clicked`` chỉ trigger LeftButton.
    * Bug 11: ``_on_selection_changed`` wrap try/finally đảm bảo unblock.
    * Bug 12+14: ``_handle_global_keys`` hỗ trợ Ctrl+S/Z/Y/F khi focus input.
    * Bug 15: ``_play_current_line`` set ``_loop_single_play``.
    * UX 3: Bỏ ``_edit_apply_text`` button + ``_on_apply_text_clicked``.
    * UX 12: ``dragEnterEvent`` + ``dropEvent`` cho drag-drop file.
    * UX 13: ``_on_auto_fix_timeline`` confirm dialog.
    * UX 14: ``_on_strip_tags_clicked`` không strip \\N.
    * Constants: ``_LOW_CONFIDENCE_THRESHOLD``, ``_MAX_LINE_CHARS``, etc.
    * Duplicate progress_bar bug fixed.
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


@pytest.fixture(scope="module")
def editor_source() -> str:
    return EDITOR_PAGE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def editor_tree(editor_source: str) -> ast.Module:
    return ast.parse(editor_source)


class TestBug1TextEditingLength:
    """Bug 1: ``_on_text_editing`` strip HTML/N trước khi đếm length."""

    @pytest.mark.skip(reason="Introspection lỗi thời: logic đã refactor sang hàm thuần strip_formatting_tags; hành vi kiểm bởi test_text_utils_v314.")
    def test_clean_text_used_for_length(self, editor_source: str) -> None:
        """Phải gán kết quả ``re.sub`` vào biến ``text_clean`` và dùng để đếm."""
        # Tìm method _on_text_editing.
        idx = editor_source.index("def _on_text_editing(self)")
        snippet = editor_source[idx:idx + 800]
        # Phải có biến text_clean = re.sub(...)
        assert "text_clean = re.sub" in snippet, (
            "Kết quả re.sub phải được gán cho biến (Bug 1 fix)"
        )
        # max_len phải dùng text_clean.split, không phải text gốc.
        assert "text_clean.split" in snippet


class TestBug2ChangeSpeedSimplified:
    """Bug 2: ``_change_speed`` không còn redundant comparison."""

    def test_no_redundant_comparison(self, editor_source: str) -> None:
        """Trước đây có pattern ``f'{x}x' not in [...] else f'{x}x'`` —
        2 branch trả về cùng giá trị (bug logic)."""
        idx = editor_source.index("def _change_speed(self")
        snippet = editor_source[idx:idx + 1500]
        # Không còn pattern redundant.
        assert "not in[" not in snippet


class TestBug6LoadVideoErrorShown:
    """Bug 6: ``load_video`` show InfoBar.error nếu exception."""

    def test_infobar_error_called_on_exception(self, editor_source: str) -> None:
        idx = editor_source.index("def load_video(self, path: Path)")
        snippet = editor_source[idx:idx + 2000]
        # Phải có InfoBar.error trong except block.
        assert "InfoBar.error(" in snippet


class TestBug7VideoClickedLeftOnly:
    """Bug 7: ``_on_video_clicked`` chỉ trigger trên LeftButton."""

    def test_only_left_button_toggles(self, editor_source: str) -> None:
        # Tìm method theo tên — không phụ thuộc vào type annotation cụ thể.
        idx = editor_source.index("def _on_video_clicked(self, button")
        snippet = editor_source[idx:idx + 500]
        # Phải check == LeftButton, không phải `in (...)`.
        assert "Qt.MouseButton.LeftButton" in snippet
        # Không còn check `in (...)` tuple với RightButton.
        assert "RightButton" not in snippet


class TestBug11SelectionChangedSafeUnblock:
    """Bug 11: ``_on_selection_changed`` wrap try/finally đảm bảo unblock signals."""

    def test_has_try_finally_block(self, editor_source: str) -> None:
        idx = editor_source.index("def _on_selection_changed(self)")
        snippet = editor_source[idx:idx + 2500]
        # Phải có try/finally đảm bảo block/unblock signals.
        assert "self._edit_start.blockSignals(True)" in snippet
        assert "try:" in snippet
        assert "finally:" in snippet
        assert "self._edit_start.blockSignals(False)" in snippet
        assert "self._edit_end.blockSignals(False)" in snippet
        assert "self._edit_text.blockSignals(False)" in snippet


class TestBug12And14GlobalKeysInInput:
    """Bug 12+14: ``_handle_global_keys`` hỗ trợ Ctrl+S/Z/Y/F khi đang focus input."""

    def test_ctrl_s_works_in_input(self, editor_source: str) -> None:
        idx = editor_source.index("def _handle_global_keys(self")
        snippet = editor_source[idx:idx + 4500]
        # Phải có check Ctrl+S trong input branch.
        assert "Key_S" in snippet
        assert "self._on_export_clicked()" in snippet

    def test_ctrl_o_open_subtitle_shortcut_exists(self, editor_source: str) -> None:
        """v2.35 NEW: Ctrl+O mở subtitle, Ctrl+Shift+O mở video."""
        # Tìm method body đầy đủ — không slice cố định, tìm boundary.
        start = editor_source.index("def _handle_global_keys(self")
        end = editor_source.index("def _play_current_line(self)", start)
        method_body = editor_source[start:end]
        assert "Key_O" in method_body
        assert "_on_open_clicked" in method_body
        assert "_on_open_video_clicked" in method_body


class TestBug15PlayCurrentLineSync:
    """Bug 15: ``_play_current_line`` set ``_loop_single_play=True`` rõ ràng."""

    def test_loop_single_play_set(self, editor_source: str) -> None:
        idx = editor_source.index("def _play_current_line(self)")
        snippet = editor_source[idx:idx + 1500]
        assert "self._loop_single_play = True" in snippet
        # Show InfoBar để user biết.
        assert "InfoBar.info(" in snippet

    def test_loop_single_play_initialized_in_init(self, editor_source: str) -> None:
        """``_loop_single_play`` phải khởi tạo ở __init__ (rõ ràng)."""
        idx = editor_source.index("class EditorPage(QWidget):")
        init_idx = editor_source.index("def __init__(self, container", idx)
        snippet = editor_source[init_idx:init_idx + 4000]
        assert "self._loop_single_play: bool = False" in snippet


class TestUx3DeadCodeButtonRemoved:
    """UX 3: Button "Lưu Chữ" ẩn vĩnh viễn đã xoá."""

    def test_no_edit_apply_text_attribute(self, editor_source: str) -> None:
        # Không còn ``self._edit_apply_text`` hoặc ``_on_apply_text_clicked`` (active).
        # Trừ comment/docstring giải thích đã xoá.
        for line in editor_source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            # Cho phép trong docstring comment.
            assert "self._edit_apply_text" not in stripped, (
                f"Còn reference self._edit_apply_text ở: {line!r}"
            )

    def test_no_on_apply_text_clicked_method(self, editor_tree: ast.Module) -> None:
        """Method ``_on_apply_text_clicked`` đã xoá."""
        for node in ast.walk(editor_tree):
            if isinstance(node, ast.FunctionDef):
                assert node.name != "_on_apply_text_clicked"


class TestUx12DragDropSupport:
    """UX 12: dragEnter/dropEvent cho phép kéo file vào page."""

    def test_drag_enter_event_method_exists(self, editor_tree: ast.Module) -> None:
        method_names = set()
        for node in ast.walk(editor_tree):
            if isinstance(node, ast.ClassDef) and node.name == "EditorPage":
                method_names = {
                    item.name for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                break
        assert "dragEnterEvent" in method_names
        assert "dropEvent" in method_names

    def test_video_extensions_defined(self, editor_source: str) -> None:
        assert "_VIDEO_EXTENSIONS" in editor_source
        # Common extensions phải có.
        for ext in (".mp4", ".mkv", ".avi", ".mov"):
            assert f'"{ext}"' in editor_source

    def test_subtitle_extensions_defined(self, editor_source: str) -> None:
        assert "_SUBTITLE_EXTENSIONS" in editor_source
        for ext in (".srt", ".ass", ".vtt"):
            assert f'"{ext}"' in editor_source

    def test_accept_drops_set_in_init(self, editor_source: str) -> None:
        assert "self.setAcceptDrops(True)" in editor_source


class TestUx13AutoFixConfirm:
    """UX 13: ``_on_auto_fix_timeline`` confirm dialog trước modify."""

    def test_question_messagebox_called(self, editor_source: str) -> None:
        idx = editor_source.index("def _on_auto_fix_timeline(self)")
        snippet = editor_source[idx:idx + 2500]
        # Phải gọi QMessageBox.question.
        assert "QMessageBox.question(" in snippet
        # Đếm error trước khi gọi service.
        assert "error_count" in snippet


class TestUx14StripTagsDoesNotRemoveLineBreak:
    """UX 14: ``_on_strip_tags_clicked`` không strip ``\\N`` line breaks."""

    @pytest.mark.skip(reason="Introspection lỗi thời: logic đã refactor sang hàm thuần strip_formatting_tags; hành vi kiểm bởi test_text_utils_v314.")
    def test_pattern_only_matches_braces_block(self, editor_source: str) -> None:
        idx = editor_source.index("def _on_strip_tags_clicked(self)")
        snippet = editor_source[idx:idx + 1500]
        # Pattern an toàn: chỉ match ``\{[^}]*\}`` (không match ``\N``).
        assert r"\{[^}]*\}" in snippet
        # InfoBar báo user biết đã strip bao nhiêu ký tự.
        assert "InfoBar.info(" in snippet


class TestConstantsExtracted:
    """v2.35: Magic numbers đã extract thành constants module-level."""

    def test_constants_declared(self, editor_source: str) -> None:
        for const_name in (
            "_LOW_CONFIDENCE_THRESHOLD",
            "_TOO_FAST_CPS_THRESHOLD",
            "_TOO_SHORT_DURATION_SEC",
            "_MAX_LINE_CHARS",
            "_DEBOUNCED_SEEK_MS",
            "_PREVIEW_DEBOUNCE_MS",
            "_SYNC_HIGHLIGHT_MS",
            "_FILTER_DEBOUNCE_MS",
            "_AUTOSAVE_INTERVAL_MS",
            "_MIN_PLAYBACK_SPEED",
            "_MAX_PLAYBACK_SPEED",
        ):
            assert f"{const_name}:" in editor_source or f"{const_name} =" in editor_source

    def test_constants_have_reasonable_values(self, editor_source: str) -> None:
        """Verify giá trị constants phù hợp standard subtitle production."""
        # Tìm dòng định nghĩa.
        for line in editor_source.splitlines():
            if line.startswith("_LOW_CONFIDENCE_THRESHOLD"):
                assert "0.6" in line  # OCR threshold chuẩn.
            elif line.startswith("_MAX_LINE_CHARS"):
                assert "40" in line   # BBC/Netflix standard.
            elif line.startswith("_MAX_PLAYBACK_SPEED"):
                assert "4.0" in line


class TestDuplicateProgressBarFixed:
    """v2.35 BUG: ``_build_statusbar`` trước đây tạo 2 lần ``_progress_bar``,
    lần 2 là dead code sau ``return``. Đã sửa."""

    def test_only_one_progress_bar_creation(self, editor_source: str) -> None:
        idx = editor_source.index("def _build_statusbar(self)")
        # Tìm boundary của method này (start của method tiếp theo).
        next_idx = editor_source.index("def _connect_signals(self)", idx)
        method_body = editor_source[idx:next_idx]
        # Chỉ có 1 lần ``self._progress_bar = ...`` assignment.
        progress_bar_assignments = method_body.count("self._progress_bar = ")
        assert progress_bar_assignments == 1, (
            f"Có {progress_bar_assignments} lần khởi tạo progress_bar (phải 1)"
        )

    def test_uses_fluent_progress_bar(self, editor_source: str) -> None:
        """Dùng Fluent ``ProgressBar`` thay vì stdlib ``QProgressBar``."""
        idx = editor_source.index("def _build_statusbar(self)")
        next_idx = editor_source.index("def _connect_signals(self)", idx)
        method_body = editor_source[idx:next_idx]
        assert "self._progress_bar = ProgressBar()" in method_body


class TestFilterDebounce:
    """v2.35: Filter ``_apply_filters`` qua debounce timer (perf optimize)."""

    def test_filter_debounce_timer_initialized(self, editor_source: str) -> None:
        assert "self._filter_debounce_timer = QTimer(self)" in editor_source
        assert "_FILTER_DEBOUNCE_MS" in editor_source

    def test_find_edit_text_changed_connects_to_timer(self, editor_source: str) -> None:
        # textChanged → start debounce timer thay vì trực tiếp _apply_filters.
        assert "self._find_edit.textChanged.connect(self._filter_debounce_timer.start)" in editor_source


class TestLoguruMigration:
    """v2.35: editor_page migrate sang Loguru."""

    def test_uses_loguru_logger(self, editor_source: str) -> None:
        """Import ``from loguru import logger`` thay vì stdlib logging."""
        assert "from loguru import logger" in editor_source

    def test_no_stdlib_logger_getLogger(self, editor_source: str) -> None:
        """Bỏ ``logger = logging.getLogger(__name__)``."""
        assert "logger = logging.getLogger(__name__)" not in editor_source


class TestSyntaxValid:
    """v2.35: editor_page sau khi sửa vẫn parse OK."""

    def test_ast_parses(self) -> None:
        source = EDITOR_PAGE_PATH.read_text(encoding="utf-8")
        ast.parse(source)
