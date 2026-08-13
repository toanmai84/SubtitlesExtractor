"""Tests bảo vệ v3.6 ViewModel bugfixes.

POP-1: _populate_form thiếu try/finally → _is_populating kẹt True
ARD-1: _on_analysis_ready_show_dialog thiếu try/finally → _is_busy kẹt True
LV-1:  load_video() chỉ bắt SubtitlesExtractorError → OSError crash silent
RAR-1: review_auto_roi_again không check _is_busy
"""
from pathlib import Path

_EVM_SRC = (
    Path(__file__).resolve().parents[3]
    / "src/subtitles_extractor/presentation/view_models/extract_page_view_model.py"
)
_SETTINGS_SRC = (
    Path(__file__).resolve().parents[3]
    / "src/subtitles_extractor/presentation/pages/settings_page.py"
)


def _snip_method(source: str, method: str) -> str:
    start = source.find(f"def {method}")
    end = source.find("\n    def ", start + 1)
    return source[start:end if end != -1 else len(source)]


# ── POP-1: _populate_form try/finally ────────────────────────────────────────

class TestPopulateFormTryFinally:
    """POP-1: _populate_form phải có try/finally để _is_populating luôn reset."""

    def test_populate_form_has_try_finally(self) -> None:
        src = _SETTINGS_SRC.read_text(encoding="utf-8")
        snippet = _snip_method(src, "_populate_form")
        assert "try:" in snippet, "Phải có try: block"
        assert "finally:" in snippet, "POP-1: Phải có finally: để đảm bảo _is_populating reset"

    def test_is_populating_false_in_finally(self) -> None:
        src = _SETTINGS_SRC.read_text(encoding="utf-8")
        snippet = _snip_method(src, "_populate_form")
        finally_pos = snippet.find("finally:")
        assert finally_pos != -1
        finally_block = snippet[finally_pos:]
        assert "_is_populating = False" in finally_block, (
            "POP-1: _is_populating = False phải nằm trong finally block."
        )

    def test_is_populating_not_reset_before_finally(self) -> None:
        """_is_populating = False không được đặt trước finally (nên nằm trong finally)."""
        src = _SETTINGS_SRC.read_text(encoding="utf-8")
        snippet = _snip_method(src, "_populate_form")
        try_start = snippet.find("try:")
        finally_pos = snippet.find("finally:")
        # Tìm _is_populating = False ở vùng try...finally
        try_block = snippet[try_start:finally_pos]
        # Không được có _is_populating = False trong try block (chỉ nên có trong finally)
        real_assignments = [
            l for l in try_block.splitlines()
            if "_is_populating = False" in l and not l.strip().startswith("#")
        ]
        assert len(real_assignments) == 0, (
            "Không được có '_is_populating = False' trong try block — "
            "chỉ trong finally để đảm bảo luôn chạy."
        )


# ── ARD-1: _on_analysis_ready_show_dialog try/finally ───────────────────────

class TestAnalysisReadyDialogTryFinally:
    """ARD-1: _on_analysis_ready_show_dialog phải có try/finally quanh _set_busy(True)."""

    def test_has_try_finally(self) -> None:
        src = _EVM_SRC.read_text(encoding="utf-8")
        snippet = _snip_method(src, "_on_analysis_ready_show_dialog")
        assert "try:" in snippet, "Phải có try: block"
        assert "finally:" in snippet, "ARD-1: Phải có finally: để đảm bảo _set_busy(False)"

    def test_set_busy_false_in_finally(self) -> None:
        src = _EVM_SRC.read_text(encoding="utf-8")
        snippet = _snip_method(src, "_on_analysis_ready_show_dialog")
        finally_pos = snippet.find("finally:")
        assert finally_pos != -1
        finally_block = snippet[finally_pos:]
        code_lines = [l for l in finally_block.splitlines() if not l.strip().startswith("#")]
        assert any("_set_busy(False)" in l for l in code_lines), (
            "ARD-1: _set_busy(False) phải nằm trong finally block."
        )

    def test_no_set_busy_false_outside_finally(self) -> None:
        """_set_busy(False) không nên gọi ở nhiều nơi ngoài finally."""
        src = _EVM_SRC.read_text(encoding="utf-8")
        snippet = _snip_method(src, "_on_analysis_ready_show_dialog")
        finally_pos = snippet.find("finally:")
        before_finally = snippet[:finally_pos] if finally_pos != -1 else snippet
        code_calls = [
            l for l in before_finally.splitlines()
            if "_set_busy(False)" in l and not l.strip().startswith("#")
        ]
        assert len(code_calls) == 0, (
            "ARD-1: Không nên có _set_busy(False) TRƯỚC finally — "
            "nên chỉ có trong finally để đảm bảo 1 nơi duy nhất reset."
        )


# ── LV-1: load_video exception handling ─────────────────────────────────────

class TestLoadVideoExceptionHandling:
    """LV-1: load_video() phải bắt OSError, FileNotFoundError."""

    def test_file_not_found_caught(self) -> None:
        src = _EVM_SRC.read_text(encoding="utf-8")
        snippet = _snip_method(src, "load_video")
        assert "FileNotFoundError" in snippet, (
            "LV-1: load_video phải bắt FileNotFoundError cho video không tồn tại."
        )

    def test_oserror_caught(self) -> None:
        src = _EVM_SRC.read_text(encoding="utf-8")
        snippet = _snip_method(src, "load_video")
        assert "OSError" in snippet, (
            "LV-1: load_video phải bắt OSError cho video corrupt/locked."
        )

    def test_errors_emit_extraction_failed(self) -> None:
        src = _EVM_SRC.read_text(encoding="utf-8")
        snippet = _snip_method(src, "load_video")
        # Mỗi exception path phải emit extraction_failed
        assert snippet.count("extraction_failed.emit(") >= 2, (
            "Phải có ít nhất 2 lần emit extraction_failed (SubtitlesExtractorError + OSError)."
        )


# ── RAR-1: review_auto_roi_again _is_busy check ──────────────────────────────

class TestReviewAutoRoiAgainBusyCheck:
    """RAR-1: review_auto_roi_again phải check _is_busy."""

    def test_busy_check_present(self) -> None:
        src = _EVM_SRC.read_text(encoding="utf-8")
        snippet = _snip_method(src, "review_auto_roi_again")
        assert "_is_busy" in snippet, (
            "RAR-1: review_auto_roi_again phải kiểm tra _is_busy trước khi mở dialog. "
            "Gọi giữa extraction → _set_busy(False) can thiệp sai flow."
        )

    def test_not_called_when_busy(self) -> None:
        src = _EVM_SRC.read_text(encoding="utf-8")
        snippet = _snip_method(src, "review_auto_roi_again")
        assert "not self._is_busy" in snippet or "is_busy" in snippet, (
            "Phải có logic: không gọi dialog khi _is_busy=True."
        )
