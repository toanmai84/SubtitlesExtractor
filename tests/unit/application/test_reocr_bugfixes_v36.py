"""Unit tests bảo vệ v3.6 Re-OCR bugfixes.

Bug R1: success message bị xóa ngay bởi (0,0,"") → user không thấy
Bug R2: single-row expand cả start+end: lần 2 update_timing dùng stale ref → revert lần 1
Bug R3: _save_roi_to_video_state trước metadata validation → persist ROI sai khi video lỗi
Bug R4: khi Re-OCR no-results, timing changes từ update_timing vẫn còn → phải undo thủ công
Bug R5: waveform reocr region không clear sau khi Re-OCR hoàn tất
"""
from __future__ import annotations

from pathlib import Path

import pytest


_VM_PATH = (
    Path(__file__).resolve().parents[3]
    / "src" / "subtitles_extractor"
    / "presentation" / "view_models" / "editor_page_view_model.py"
)
_PAGE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src" / "subtitles_extractor"
    / "presentation" / "pages" / "editor_page.py"
)


@pytest.fixture(scope="module")
def vm_source() -> str:
    return _VM_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def page_source() -> str:
    return _PAGE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Bug R1: progress_changed(0,0,"") → xóa success message
# ---------------------------------------------------------------------------


class TestReOcrProgressClearFix:
    """Bug R1: _on_reocr_progress không xóa status khi msg rỗng."""

    def test_on_reocr_progress_keeps_status_when_msg_empty(
        self, page_source: str
    ) -> None:
        """Khi msg='', status_label phải được giữ nguyên (không setText(''))."""
        method_start = page_source.find("def _on_reocr_progress(self,")
        next_method = page_source.find("\n    def ", method_start + 1)
        snippet = page_source[method_start:next_method]

        # Phải có điều kiện if msg trước khi setText
        assert "if msg" in snippet or "if msg:" in snippet, (
            "Bug R1: _on_reocr_progress phải kiểm tra `if msg:` trước khi "
            "gọi setText() — tránh xóa thông báo 'Re-OCR hoàn tất' ngay lập tức."
        )

    def test_thread_finished_emits_zero_progress_for_bar_only(
        self, vm_source: str
    ) -> None:
        """_on_reocr_thread_finished vẫn emit progress(0,0,'') để ẩn progress bar."""
        method_start = vm_source.find("def _on_reocr_thread_finished(self)")
        next_method = vm_source.find("\n    def ", method_start + 1)
        snippet = vm_source[method_start:next_method]

        assert "progress_changed.emit(0, 0," in snippet, (
            "_on_reocr_thread_finished cần emit progress_changed(0,0,...) để "
            "ẩn progress bar, nhưng View phải ignore msg='' khi cập nhật status."
        )


# ---------------------------------------------------------------------------
# Bug R2+R4: explicit_time_ranges thay thế update_timing pre-expand
# ---------------------------------------------------------------------------


class TestExplicitTimeRangesReplacePreExpand:
    """Bug R2+R4: start_reocr phải hỗ trợ explicit_time_ranges."""

    def test_start_reocr_has_explicit_time_ranges_param(
        self, vm_source: str
    ) -> None:
        """start_reocr phải có parameter explicit_time_ranges."""
        method_start = vm_source.find("def start_reocr(")
        next_method = vm_source.find("\n    def ", method_start + 1)
        signature = vm_source[method_start:next_method]

        assert "explicit_time_ranges" in signature, (
            "Bug R2/R4: start_reocr phải có parameter explicit_time_ranges. "
            "Dùng explicit_time_ranges để tránh phải gọi update_timing() trước "
            "Re-OCR — ngăn stale-ref revert (R2) và orphaned undo entries (R4)."
        )

    def test_start_reocr_uses_explicit_ranges_when_provided(
        self, vm_source: str
    ) -> None:
        """start_reocr phải dùng explicit_time_ranges khi được cung cấp."""
        method_start = vm_source.find("def start_reocr(")
        next_method = vm_source.find("\n    def ", method_start + 1)
        snippet = vm_source[method_start:next_method]

        assert (
            "explicit_time_ranges is not None" in snippet
            or "if explicit_time_ranges" in snippet
        ), (
            "start_reocr phải có logic: 'if explicit_time_ranges is not None: "
            "time_ranges = explicit_time_ranges'"
        )

    def test_fast_reocr_no_update_timing_calls(self, page_source: str) -> None:
        """_on_fast_reocr_clicked không được gọi update_timing() nữa."""
        method_start = page_source.find("def _on_fast_reocr_clicked(self)")
        next_method = page_source.find("\n    def ", method_start + 1)
        snippet = page_source[method_start:next_method]

        # Chỉ đếm ACTUAL calls, không đếm mentions trong comment (#)
        real_calls = [
            line for line in snippet.splitlines()
            if "update_timing(" in line and not line.strip().startswith("#")
        ]
        assert len(real_calls) == 0, (
            f"Bug R2/R4: _on_fast_reocr_clicked vẫn còn {len(real_calls)} "
            "lần gọi thực sự update_timing(). Phải dùng explicit_time_ranges thay thế."
        )

    def test_fast_reocr_passes_explicit_time_ranges(self, page_source: str) -> None:
        """_on_fast_reocr_clicked phải truyền explicit_time_ranges cho start_reocr."""
        method_start = page_source.find("def _on_fast_reocr_clicked(self)")
        next_method = page_source.find("\n    def ", method_start + 1)
        snippet = page_source[method_start:next_method]

        assert "explicit_time_ranges" in snippet, (
            "_on_fast_reocr_clicked phải truyền explicit_time_ranges=... "
            "khi gọi start_reocr()."
        )

    def test_advanced_reocr_no_update_timing_calls(self, page_source: str) -> None:
        """_on_reocr_clicked (nâng cao) không được gọi update_timing() nữa."""
        method_start = page_source.find("def _on_reocr_clicked(self)")
        next_method = page_source.find("\n    def ", method_start + 1)
        snippet = page_source[method_start:next_method]

        real_calls = [
            line for line in snippet.splitlines()
            if "update_timing(" in line and not line.strip().startswith("#")
        ]
        assert len(real_calls) == 0, (
            f"Bug R2/R4: _on_reocr_clicked vẫn còn {len(real_calls)} "
            "lần gọi thực sự update_timing(). Phải dùng explicit_time_ranges."
        )

    def test_advanced_reocr_passes_explicit_time_ranges(
        self, page_source: str
    ) -> None:
        """_on_reocr_clicked phải truyền explicit_time_ranges."""
        method_start = page_source.find("def _on_reocr_clicked(self)")
        next_method = page_source.find("\n    def ", method_start + 1)
        snippet = page_source[method_start:next_method]

        assert "explicit_time_ranges" in snippet


# ---------------------------------------------------------------------------
# Bug R3: _save_roi_to_video_state sau metadata validation — source analysis
# ---------------------------------------------------------------------------


class TestSaveRoiAfterMetadataValidation:
    """Bug R3: ROI không được lưu trước khi metadata đã được validate."""

    def test_metadata_read_before_save_roi(self, vm_source: str) -> None:
        method_start = vm_source.find("def start_reocr(")
        next_method = vm_source.find("\n    def ", method_start + 1)
        snippet = vm_source[method_start:next_method]
        metadata_pos = snippet.find("metadata_reader.read(")
        save_roi_pos = snippet.find("_save_roi_to_video_state(")
        assert metadata_pos != -1
        assert save_roi_pos != -1
        assert metadata_pos < save_roi_pos, (
            "Bug R3: _save_roi_to_video_state() phải xuất hiện SAU metadata_reader.read()."
        )

    def test_save_roi_after_request_build(self, vm_source: str) -> None:
        method_start = vm_source.find("def start_reocr(")
        next_method = vm_source.find("\n    def ", method_start + 1)
        snippet = vm_source[method_start:next_method]
        build_request_pos = snippet.find("_build_reocr_request(")
        save_roi_pos = snippet.find("_save_roi_to_video_state(")
        assert build_request_pos < save_roi_pos, (
            "Bug R3: _save_roi_to_video_state() nên xuất hiện sau _build_reocr_request()."
        )


# ---------------------------------------------------------------------------
# Integration: _build_time_ranges qua source analysis
# ---------------------------------------------------------------------------


class TestBuildTimeRanges:
    """Kiểm tra _build_time_ranges logic."""

    def test_build_time_ranges_method_exists(self, vm_source: str) -> None:
        assert "_build_time_ranges" in vm_source

    def test_build_time_ranges_has_duration_check(self, vm_source: str) -> None:
        method_start = vm_source.find("def _build_time_ranges(")
        end = vm_source.find("\n    def ", method_start + 1)
        snippet = vm_source[method_start:end if end != -1 else len(vm_source)]
        assert "duration_sec" in snippet or "duration" in snippet

    def test_explicit_time_ranges_fallback_to_build(self, vm_source: str) -> None:
        method_start = vm_source.find("def start_reocr(")
        next_method = vm_source.find("\n    def ", method_start + 1)
        snippet = vm_source[method_start:next_method]
        assert "explicit_time_ranges" in snippet
        assert "_build_time_ranges" in snippet
