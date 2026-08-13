"""Tests bảo vệ v3.6 export busy-race fixes.

Bộ 3 lỗi gây treo intermittent:
 EXPORT-FIX-1: _set_busy(False) phải gọi TRƯỚC export_finished.emit()
 EXPORT-FIX-2: export_to_file() phải trả False khi _is_busy=True
 EXPORT-FIX-3: _on_export_clicked phải kiểm tra _is_busy sau dialog đóng
"""
from pathlib import Path

_VM_SRC = (
    Path(__file__).resolve().parents[3]
    / "src/subtitles_extractor/presentation/view_models/editor_page_view_model.py"
)
_PAGE_SRC = (
    Path(__file__).resolve().parents[3]
    / "src/subtitles_extractor/presentation/pages/editor_page.py"
)


def _snip(source: str, method: str) -> str:
    start = source.find(f"def {method}")
    end = source.find("\n    def ", start + 1)
    return source[start:end if end != -1 else len(source)]


# ── EXPORT-FIX-1: _set_busy trước export_finished ──────────────────────────

class TestSetBusyBeforeExportFinished:
    """_set_busy(False) phải gọi TRƯỚC export_finished.emit()."""

    def test_set_busy_before_export_finished_in_success_handler(self) -> None:
        src = _VM_SRC.read_text(encoding="utf-8")
        snippet = _snip(src, "_on_export_worker_success")

        busy_pos   = snippet.find("_set_busy(False)")
        export_pos = snippet.find("export_finished.emit(")

        assert busy_pos != -1,   "_set_busy(False) phải tồn tại trong _on_export_worker_success"
        assert export_pos != -1, "export_finished.emit() phải tồn tại"
        assert busy_pos < export_pos, (
            "EXPORT-FIX-1: _set_busy(False) phải gọi TRƯỚC export_finished.emit().\n"
            "Nếu export_finished chạy trước: _on_export_finished gọi setEnabled(True) "
            "khi _is_busy còn True → tạo race window → user click lúc này → "
            "export_to_file() return sớm → nút kẹt disabled → TREO."
        )

    def test_state_changed_before_set_busy(self) -> None:
        """Thứ tự đúng: state_changed → _set_busy(False) → export_finished."""
        src = _VM_SRC.read_text(encoding="utf-8")
        snippet = _snip(src, "_on_export_worker_success")

        # Bỏ qua docstring và comments — chỉ lấy code thực
        in_docstring = False
        code_lines: list[tuple[int, str]] = []
        for i, line in enumerate(snippet.splitlines()):
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_docstring = not in_docstring
                continue
            if in_docstring:
                continue
            if stripped.startswith("#") or not stripped:
                continue
            code_lines.append((i, line))

        def find_code_line(keyword: str) -> int:
            for i, line in code_lines:
                if keyword in line:
                    return i
            return -1

        state_line  = find_code_line("state_changed.emit(")
        busy_line   = find_code_line("_set_busy(False)")
        export_line = find_code_line("export_finished.emit(")

        assert state_line  != -1, "state_changed.emit() phải tồn tại (code, không phải comment)"
        assert busy_line   != -1, "_set_busy(False) phải tồn tại (code)"
        assert export_line != -1, "export_finished.emit() phải tồn tại (code)"
        assert state_line < busy_line < export_line, (
            f"Thứ tự code phải là state_changed(L{state_line}) → "
            f"_set_busy(L{busy_line}) → export_finished(L{export_line}).\n"
            "EXPORT-FIX-1: Nếu _set_busy(False) sau export_finished, _on_export_finished "
            "gọi setEnabled(True) khi _is_busy vẫn True → race window."
        )


# ── EXPORT-FIX-2: export_to_file trả False khi busy ────────────────────────

class TestExportToFileReturnsFalseWhenBusy:
    """export_to_file() phải trả False khi _is_busy=True."""

    def test_export_to_file_has_explicit_return_false(self) -> None:
        src = _VM_SRC.read_text(encoding="utf-8")
        snippet = _snip(src, "export_to_file")

        # Phải có return False khi is_busy
        real_returns = [
            line for line in snippet.splitlines()
            if "return False" in line and not line.strip().startswith("#")
        ]
        assert len(real_returns) >= 1, (
            "EXPORT-FIX-2: export_to_file() phải return False khi _is_busy=True, "
            "không phải return None (implicit). Caller cần biết export bị reject "
            "để khôi phục UI state."
        )

    def test_export_to_file_returns_true_when_started(self) -> None:
        src = _VM_SRC.read_text(encoding="utf-8")
        snippet = _snip(src, "export_to_file")

        real_returns = [
            line for line in snippet.splitlines()
            if "return True" in line and not line.strip().startswith("#")
        ]
        assert len(real_returns) >= 1, (
            "export_to_file() phải return True khi export đã được bắt đầu thành công."
        )


# ── EXPORT-FIX-3: _on_export_clicked guard sau dialog ──────────────────────

class TestExportClickedBusyGuardAfterDialog:
    """_on_export_clicked phải kiểm tra _is_busy SAU KHI dialog đóng."""

    def test_is_busy_check_after_dialog_call(self) -> None:
        src = _PAGE_SRC.read_text(encoding="utf-8")
        snippet = _snip(src, "_on_export_clicked")

        dialog_pos   = snippet.find("getSaveFileName(")
        is_busy_pos  = snippet.find("_is_busy")

        assert is_busy_pos != -1, (
            "EXPORT-FIX-3: _on_export_clicked phải kiểm tra _is_busy "
            "SAU KHI dialog đóng để phát hiện trường hợp hệ thống busy "
            "trong lúc dialog mở."
        )
        assert dialog_pos < is_busy_pos, (
            "_is_busy check phải xuất hiện SAU getSaveFileName() call."
        )

    def test_export_started_check_handles_false(self) -> None:
        """_on_export_clicked phải xử lý khi export_to_file trả False."""
        src = _PAGE_SRC.read_text(encoding="utf-8")
        snippet = _snip(src, "_on_export_clicked")

        # Phải có logic kiểm tra return value của export_to_file
        assert "started" in snippet or "not started" in snippet or "= self._view_model.export_to_file" in snippet, (
            "EXPORT-FIX-3: _on_export_clicked phải bắt return value của "
            "export_to_file() và phục hồi UI khi nó trả False."
        )

    def test_button_restored_when_export_rejected(self) -> None:
        """Khi export bị reject, nút phải được restore."""
        src = _PAGE_SRC.read_text(encoding="utf-8")
        snippet = _snip(src, "_on_export_clicked")

        # Phải có logic restore nút
        assert "setEnabled(bool(" in snippet or "setEnabled(True" in snippet, (
            "Khi export bị reject (is_busy), phải restore nút về trạng thái có thể click."
        )


# ── Integration: _on_export_finished không prematurely enable ───────────────

class TestExportFinishedButtonState:
    """_on_export_finished được gọi khi _is_busy đã False → setEnabled(True) đúng."""

    def test_set_busy_false_precedes_export_finished_signal(self) -> None:
        """Đảm bảo khi _on_export_finished chạy, _is_busy đã là False."""
        src = _VM_SRC.read_text(encoding="utf-8")
        snippet = _snip(src, "_on_export_worker_success")

        # _set_busy(False) BEFORE export_finished.emit() — đã test ở trên
        busy_pos   = snippet.find("_set_busy(False)")
        export_pos = snippet.find("export_finished.emit(")
        assert busy_pos < export_pos, "Phải có thứ tự đúng"

    def test_export_finished_unconditional_enable_is_safe(self) -> None:
        """_on_export_finished gọi setEnabled(True) — sau fix, _is_busy đã False."""
        src = _PAGE_SRC.read_text(encoding="utf-8")
        snippet = _snip(src, "_on_export_finished")

        # setEnabled(True) vẫn có trong _on_export_finished (giữ nguyên)
        assert "setEnabled(True)" in snippet, (
            "_on_export_finished phải có setEnabled(True) — "
            "với EXPORT-FIX-1, _is_busy đã False khi hàm này chạy → an toàn."
        )
