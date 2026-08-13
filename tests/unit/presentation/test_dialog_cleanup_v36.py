"""Tests bảo vệ v3.6 dialog cleanup fixes.

DLG-1: AdvancedReOcrDialog không deleteLater → tích lũy MPV player → treo Export
DLG-2: EditorSettingsDialog leak
DLG-3: undo history dialog leak
DLG-4: merge similar dialogs leak

Nguyên nhân: dialog tạo với parent=self → Qt giữ sống đến khi parent hủy.
Mỗi Re-OCR tạo dialog mới kèm MPV player + native window handle.
Tích lũy → QFileDialog mở (cùng parent) tương tác GPU context → TREO.
"""
from pathlib import Path

_PAGE = Path(__file__).resolve().parents[3] / "src/subtitles_extractor/presentation/pages/editor_page.py"


def _snip(source: str, method: str) -> str:
    start = source.find(f"def {method}")
    end = source.find("\n    def ", start + 1)
    return source[start:end if end != -1 else len(source)]


# ── DLG-1: AdvancedReOcrDialog cleanup (CRITICAL) ────────────────────────────

class TestReOcrDialogCleanup:
    """DLG-1: _on_reocr_clicked PHẢI deleteLater dialog sau khi dùng."""

    def test_reocr_dialog_has_deletelater(self) -> None:
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_on_reocr_clicked")
        assert "deleteLater()" in snippet, (
            "DLG-1 CRITICAL: _on_reocr_clicked PHẢI gọi dlg.deleteLater(). "
            "Không hủy → tích lũy AdvancedReOcrDialog (kèm MPV player) → "
            "treo Export. Đây là nguyên nhân gốc của bug treo khi xuất tệp."
        )

    def test_reocr_dialog_uses_try_finally(self) -> None:
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_on_reocr_clicked")
        assert "try:" in snippet and "finally:" in snippet, (
            "DLG-1: deleteLater phải trong finally để chạy kể cả khi early-return."
        )

    def test_deletelater_in_finally_block(self) -> None:
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_on_reocr_clicked")
        finally_pos = snippet.find("finally:")
        assert finally_pos != -1
        finally_block = snippet[finally_pos:]
        assert "deleteLater()" in finally_block, (
            "DLG-1: dlg.deleteLater() phải nằm trong finally block."
        )

    def test_get_values_before_finally(self) -> None:
        """get_values() phải gọi TRƯỚC finally (khi dialog còn sống)."""
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_on_reocr_clicked")
        getvalues_pos = snippet.find("get_values()")
        finally_pos = snippet.find("finally:")
        assert getvalues_pos != -1 and finally_pos != -1
        assert getvalues_pos < finally_pos, (
            "get_values() phải gọi trong try (trước finally) khi dialog còn sống."
        )


# ── DLG-2/3/4: Các dialog khác cleanup ───────────────────────────────────────

class TestOtherDialogsCleanup:
    def test_settings_dialog_deletelater(self) -> None:
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_on_show_settings")
        assert "deleteLater()" in snippet, (
            "DLG-2: _on_show_settings phải deleteLater EditorSettingsDialog."
        )

    def test_undo_history_dialog_deletelater(self) -> None:
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_show_undo_history")
        assert "deleteLater()" in snippet, (
            "DLG-3: _show_undo_history phải deleteLater dialog."
        )

    def test_merge_similar_dialogs_deletelater(self) -> None:
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_on_merge_similar_dialog")
        # Phải có ít nhất 2 deleteLater (dlg + preview_dlg)
        count = snippet.count("deleteLater()")
        assert count >= 2, (
            f"DLG-4: _on_merge_similar_dialog phải deleteLater cả 2 dialog "
            f"(dlg + preview_dlg), tìm thấy {count}."
        )


# ── Verify AdvancedReOcrDialog có cleanup services ──────────────────────────

class TestReOcrDialogServiceCleanup:
    """Xác nhận _cleanup_services giải phóng MPV + seek service."""

    def test_cleanup_services_releases_player(self) -> None:
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_cleanup_services")
        assert "release_player" in snippet, (
            "_cleanup_services phải gọi release_player() để giải phóng MPV."
        )

    def test_cleanup_services_stops_seek_service(self) -> None:
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_cleanup_services")
        assert "_seek_service" in snippet and ".stop()" in snippet, (
            "_cleanup_services phải stop seek service (SeekWorker thread)."
        )

    def test_cleanup_called_on_all_exit_paths(self) -> None:
        """accept/reject/closeEvent đều phải gọi _cleanup_services."""
        src = _PAGE.read_text(encoding="utf-8")
        for method in ("accept", "reject", "closeEvent"):
            snippet = _snip(src, method)
            # Chỉ check method đầu tiên tìm thấy (của AdvancedReOcrDialog)
            assert "_cleanup_services()" in snippet, (
                f"AdvancedReOcrDialog.{method} phải gọi _cleanup_services()."
            )


# ── DLG-1+: Củng cố cleanup robustness ───────────────────────────────────────

class TestCleanupRobustness:
    """DLG-1+: _cleanup_services phải robust và guard queued handlers."""

    def test_seek_service_stop_wrapped_in_suppress(self) -> None:
        """_seek_service.stop() phải bọc suppress để release_player luôn chạy."""
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_cleanup_services")
        # Tìm dòng CODE thực (không phải comment) chứa _seek_service.stop()
        code_lines = [
            l for l in snippet.splitlines()
            if "_seek_service.stop()" in l and not l.strip().startswith("#")
        ]
        assert len(code_lines) >= 1, "_seek_service.stop() phải tồn tại trong code"
        # Dòng ngay trước đó (hoặc cùng block) phải có suppress
        stop_line_idx = None
        all_lines = snippet.splitlines()
        for i, l in enumerate(all_lines):
            if "_seek_service.stop()" in l and not l.strip().startswith("#"):
                stop_line_idx = i
                break
        assert stop_line_idx is not None
        # Kiểm tra dòng trước có suppress
        prev_line = all_lines[stop_line_idx - 1]
        assert "suppress" in prev_line, (
            "DLG-1+: _seek_service.stop() phải nằm trong contextlib.suppress "
            f"(dòng trước: {prev_line!r})."
        )

    def test_is_closing_flag_set(self) -> None:
        """_cleanup_services phải set _is_closing = True."""
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_cleanup_services")
        assert "_is_closing = True" in snippet, (
            "DLG-1+: _cleanup_services phải set _is_closing=True để guard handlers."
        )

    def test_position_changed_guards_closing(self) -> None:
        """_on_canvas_position_changed phải guard _is_closing."""
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_on_canvas_position_changed")
        assert "_is_closing" in snippet, (
            "_on_canvas_position_changed phải check _is_closing (queued signal safety)."
        )

    def test_seek_frame_ready_guards_closing(self) -> None:
        """_on_seek_frame_ready phải guard _is_closing (async seek service)."""
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_on_seek_frame_ready")
        assert "_is_closing" in snippet, (
            "_on_seek_frame_ready phải check _is_closing — đây là handler nguy hiểm "
            "nhất vì frame_ready đến từ seek service async, có thể queued sau cleanup."
        )

    def test_video_canvas_not_nulled_in_cleanup(self) -> None:
        """video_canvas KHÔNG được set None trong cleanup (tránh None-access)."""
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_cleanup_services")
        # Không được có 'self.video_canvas = None'
        bad = [l for l in snippet.splitlines() if "video_canvas = None" in l and not l.strip().startswith("#")]
        assert not bad, (
            "DLG-1+: KHÔNG set video_canvas=None trong cleanup — queued handler "
            "vẫn có thể tham chiếu. Dialog.deleteLater() sẽ hủy video_canvas (cascade)."
        )
