"""Tests bảo vệ v3.6 editor_page bugfixes session 3.

TL-1: _toggle_loop no-selection không clear loop region
RO-1: _replace_one dùng mixed source (ViewModel + table cache)
SF-1: _on_seek_fallback_requested timer start vô ích
PP-1: _play_current_line / _nudge_time / keyboard tạo list copy không cần thiết
"""
from pathlib import Path

_PAGE = Path(__file__).resolve().parents[3] / "src/subtitles_extractor/presentation/pages/editor_page.py"


def _snip(source: str, method: str) -> str:
    start = source.find(f"def {method}")
    end = source.find("\n    def ", start + 1)
    return source[start:end if end != -1 else len(source)]


def _real_code_lines(snippet: str) -> list[str]:
    """Lọc ra dòng code thực, bỏ docstring và comment."""
    in_docstring = False
    lines = []
    for line in snippet.splitlines():
        s = line.strip()
        if s.startswith('"""') or s.startswith("'''"):
            in_docstring = not in_docstring
            continue
        if in_docstring or s.startswith("#") or not s:
            continue
        lines.append(line)
    return lines


# ── TL-1: _toggle_loop no-selection cleanup ──────────────────────────────────

class TestToggleLoopNoSelection:
    """TL-1: _toggle_loop phải clear loop region khi không có row được chọn."""

    def test_clear_loop_region_called_when_no_selection(self) -> None:
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_toggle_loop")
        # Tìm nhánh else (khi idx < 0) — nằm sau "if idx >= 0:"
        idx_check_pos = snippet.find("if idx >= 0:")
        assert idx_check_pos != -1
        # Lấy phần sau if block (else nhánh khi idx < 0)
        after_if = snippet[idx_check_pos:]
        else_pos = after_if.find("else:")
        assert else_pos != -1, "Phải có else: block"
        else_block = after_if[else_pos:]
        real_lines = [l for l in else_block.splitlines() if "clear_loop_region" in l and not l.strip().startswith("#")]
        assert len(real_lines) >= 1, (
            "TL-1: Nhánh else (không có selection) phải gọi clear_loop_region()."
        )

    def test_loop_times_reset_when_no_selection(self) -> None:
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_toggle_loop")
        idx_check_pos = snippet.find("if idx >= 0:")
        after_if = snippet[idx_check_pos:]
        else_pos = after_if.find("else:")
        else_block = after_if[else_pos:]
        has_reset = (
            "_loop_start_time = None" in else_block
            and "_loop_end_time = None" in else_block
        )
        assert has_reset, (
            "TL-1: Nhánh else phải reset _loop_start_time và _loop_end_time về None."
        )


# ── RO-1: _replace_one consistent source ─────────────────────────────────────

class TestReplaceOneConsistentSource:
    """RO-1: _replace_one phải dùng _table_model.events nhất quán."""

    def test_no_current_events_call(self) -> None:
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_replace_one")
        real_lines = _real_code_lines(snippet)
        bad_calls = [l for l in real_lines if "current_events" in l]
        assert len(bad_calls) == 0, (
            f"RO-1: _replace_one không được dùng current_events:\n"
            + "\n".join(bad_calls)
        )

    def test_uses_table_model_events(self) -> None:
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_replace_one")
        assert "_table_model.events" in snippet, (
            "RO-1: _replace_one phải dùng self._table_model.events."
        )

    def test_has_bounds_check(self) -> None:
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_replace_one")
        assert "len(model_events)" in snippet or "len(cache)" in snippet or ">= len(" in snippet, (
            "RO-1: Phải có bounds check tránh IndexError."
        )


# ── SF-1: _on_seek_fallback_requested ────────────────────────────────────────

class TestSeekFallbackNoop:
    """SF-1: _on_seek_fallback_requested không được start timer vô ích."""

    def _get_editor_page_snippet(self, src: str, method: str) -> str:
        """Lấy method từ class EditorPage, không phải từ AdvancedReOcrDialog."""
        editor_class_start = src.find("class EditorPage(")
        start = src.find(f"def {method}", editor_class_start)
        end = src.find("\n    def ", start + 1)
        return src[start:end if end != -1 else len(src)]

    def test_no_timer_start_in_seek_fallback(self) -> None:
        src = _PAGE.read_text(encoding="utf-8")
        snippet = self._get_editor_page_snippet(src, "_on_seek_fallback_requested")
        real_lines = _real_code_lines(snippet)
        bad = [l for l in real_lines if "_debounced_seek_timer.start()" in l]
        assert len(bad) == 0, (
            "SF-1: EditorPage._on_seek_fallback_requested không được gọi "
            "_debounced_seek_timer.start()."
        )

    def test_pending_seek_row_reset(self) -> None:
        src = _PAGE.read_text(encoding="utf-8")
        snippet = self._get_editor_page_snippet(src, "_on_seek_fallback_requested")
        assert "_pending_seek_row = -1" in snippet, (
            "Vẫn phải reset _pending_seek_row để hủy pending seek."
        )


# ── PP-1: _play_current_line / _nudge_time không dùng current_events ─────────

class TestNoCurrentEventsInHotPaths:
    """PP-1: Hot paths không tạo list copy thông qua current_events."""

    def test_play_current_line_no_current_events(self) -> None:
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_play_current_line")
        bad = [l for l in _real_code_lines(snippet) if "current_events" in l]
        assert len(bad) == 0, (
            "PP-1: _play_current_line không được dùng current_events — "
            "dùng _table_model.events[row] thay thế."
        )

    def test_nudge_time_no_current_events(self) -> None:
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_nudge_time")
        bad = [l for l in _real_code_lines(snippet) if "current_events" in l]
        assert len(bad) == 0, (
            "PP-1: _nudge_time không được dùng current_events."
        )

    def test_keyboard_seek_no_current_events(self) -> None:
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_handle_global_keys")
        # Tìm vùng Ctrl+Left và Ctrl+Right
        left_pos = snippet.find("Key_Left and ctrl and shift")
        right_pos = snippet.find("Key_Right and ctrl and shift")
        if left_pos != -1 and right_pos != -1:
            region = snippet[left_pos:right_pos + 200]
            bad = [l for l in _real_code_lines(region) if "current_events" in l]
            assert len(bad) == 0, (
                "PP-1: Keyboard shortcut Ctrl+Shift+Left/Right không được dùng current_events."
            )

    def test_play_current_line_uses_table_model(self) -> None:
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_play_current_line")
        assert "_table_model.events" in snippet, (
            "_play_current_line phải dùng _table_model.events[row]."
        )

    def test_nudge_time_uses_table_model(self) -> None:
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_nudge_time")
        assert "_table_model.events" in snippet, (
            "_nudge_time phải dùng _table_model.events[idx]."
        )
