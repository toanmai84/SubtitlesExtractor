"""Unit tests bảo vệ miscellaneous v3.6 bugfixes.

MW-1: closeEvent thiếu QThreadPool.waitForDone
MW-2: extract_page resources không được dọn khi close
LF-1: load_from_file thiếu UnicodeDecodeError / OSError
KE-1: KeyError.__str__() để lộ dấu nháy đơn trong message
DB-1: _deserialize_roi dùng "UNKNOWN" làm default → KeyError → ROI lost
AF-1: _on_auto_fix_timeline đếm sai (bỏ sót small-gap cases)
"""
from __future__ import annotations

from pathlib import Path

import pytest

_VM_SRC = (
    Path(__file__).resolve().parents[3]
    / "src/subtitles_extractor/presentation/view_models/editor_page_view_model.py"
)
_MW_SRC = (
    Path(__file__).resolve().parents[3]
    / "src/subtitles_extractor/presentation/main_window.py"
)
_DB_SRC = (
    Path(__file__).resolve().parents[3]
    / "src/subtitles_extractor/infrastructure/database/sqlite_video_state_repository.py"
)
_PAGE_SRC = (
    Path(__file__).resolve().parents[3]
    / "src/subtitles_extractor/presentation/pages/editor_page.py"
)


def _snip(source: str, method: str) -> str:
    start = source.find(f"def {method}")
    end = source.find("\n    def ", start + 1)
    return source[start:end if end != -1 else len(source)]


# ── MW-1: QThreadPool.waitForDone ────────────────────────────────────────────

class TestCloseEventThreadPoolWait:
    def test_waitForDone_called_in_close_event(self) -> None:
        src = _MW_SRC.read_text(encoding="utf-8")
        snippet = _snip(src, "closeEvent")
        assert "waitForDone(" in snippet, (
            "MW-1: closeEvent PHẢI gọi QThreadPool.globalInstance().waitForDone() "
            "trước container.shutdown() để AutoSave/Export runnables hoàn tất."
        )

    def test_waitForDone_before_shutdown(self) -> None:
        src = _MW_SRC.read_text(encoding="utf-8")
        snippet = _snip(src, "closeEvent")
        wait_pos = snippet.find("waitForDone(")
        # Tìm vị trí CALL thực sự (không phải comment) — dùng "self._container.shutdown()"
        shutdown_pos = snippet.find("self._container.shutdown()")
        assert wait_pos != -1, "waitForDone() phải tồn tại"
        assert shutdown_pos != -1, "self._container.shutdown() phải tồn tại"
        assert wait_pos < shutdown_pos, (
            "waitForDone() phải được gọi TRƯỚC self._container.shutdown()."
        )


# ── MW-2: extract_page resources cleanup ────────────────────────────────────

class TestCloseEventExtractPageCleanup:
    def test_seek_thread_cleanup_in_main_window(self) -> None:
        src = _MW_SRC.read_text(encoding="utf-8")
        snippet = _snip(src, "closeEvent")
        assert "_seek_thread" in snippet, (
            "MW-2: closeEvent phải dọn extract_page._seek_thread. "
            "closeEvent() của child widget không tự gọi khi parent đóng."
        )

    def test_video_reader_cleanup_in_main_window(self) -> None:
        src = _MW_SRC.read_text(encoding="utf-8")
        snippet = _snip(src, "closeEvent")
        assert "_video_reader" in snippet, (
            "MW-2: closeEvent phải dọn extract_page._video_reader (đóng file handle)."
        )


# ── LF-1: load_from_file exception coverage ─────────────────────────────────

class TestLoadFromFileExceptions:
    def test_unicode_error_caught(self) -> None:
        src = _VM_SRC.read_text(encoding="utf-8")
        snippet = _snip(src, "load_from_file")
        assert "UnicodeDecodeError" in snippet, (
            "LF-1: load_from_file PHẢI bắt UnicodeDecodeError. "
            "File không phải UTF-8 sẽ crash nếu không handle."
        )

    def test_oserror_caught(self) -> None:
        src = _VM_SRC.read_text(encoding="utf-8")
        snippet = _snip(src, "load_from_file")
        assert "OSError" in snippet, (
            "LF-1: load_from_file PHẢI bắt OSError "
            "(file bị xoá, permission denied, disk error)."
        )


# ── KE-1: KeyError clean message ─────────────────────────────────────────────

class TestKeyErrorCleanMessage:
    def test_keyerror_args0_used_not_str(self) -> None:
        src = _VM_SRC.read_text(encoding="utf-8")
        snippet = _snip(src, "load_from_file")
        # Phải có args[0] để tránh dấu nháy đơn trong message
        assert "args[0]" in snippet, (
            "KE-1: Phải dùng exc.args[0] thay vì str(exc) cho KeyError. "
            "str(KeyError('msg')) → \"'msg'\" (có dấu nháy) trông rất xấu."
        )

    def test_keyerror_str_has_quotes_problem(self) -> None:
        """Minh hoạ vấn đề: str(KeyError) bọc chuỗi trong quotes (repr format)."""
        exc = KeyError("Định dạng 'pdf' không hỗ trợ")
        str_version = str(exc)
        # Python dùng single quotes nếu key không có single quote,
        # double quotes nếu key có single quote → luôn có dấu quote bao quanh.
        is_wrapped = (
            (str_version.startswith("'") and str_version.endswith("'"))
            or (str_version.startswith('"') and str_version.endswith('"'))
        )
        assert is_wrapped, (
            f"str(KeyError) phải có quote bao quanh nhưng nhận: {str_version!r}"
        )
        # exc.args[0] phải trả về chuỗi gốc không có quote
        clean = exc.args[0]
        assert not (clean.startswith("'") or clean.startswith('"')), (
            "exc.args[0] phải KHÔNG có dấu quote bao quanh"
        )

    def test_keyerror_args0_gives_clean_string(self) -> None:
        """exc.args[0] phải trả chuỗi không có dấu nháy."""
        test_msg = "Định dạng 'xyz' không được hỗ trợ."
        exc = KeyError(test_msg)
        assert exc.args[0] == test_msg
        assert str(exc) == repr(test_msg)  # Có quotes trong str()
        assert exc.args[0] != str(exc)     # Chứng minh khác nhau


# ── DB-1: _deserialize_roi fallback alignment ────────────────────────────────

class TestDeserializeRoiFallback:
    def test_unknown_alignment_returns_roi_not_none(self) -> None:
        """'UNKNOWN' alignment không được làm discard toàn bộ ROI nữa."""
        from subtitles_extractor.infrastructure.database.sqlite_video_state_repository import (
            SqliteVideoStateRepository,
        )
        import json, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            repo = SqliteVideoStateRepository(Path(tmp) / "test.db")
            bad_json = json.dumps({
                "x": 0, "y": 720, "width": 1920, "height": 200,
                "alignment": "UNKNOWN",  # Invalid!
                "orientation": "HORIZONTAL",
            })
            roi = repo._deserialize_roi(bad_json)
            # Sau bugfix: ROI phải được trả về với fallback alignment
            assert roi is not None, (
                "DB-1: 'UNKNOWN' alignment phải fallback về CENTER, "
                "không bỏ cả ROI."
            )

    def test_missing_alignment_uses_fallback(self) -> None:
        """Thiếu trường alignment → dùng default CENTER."""
        from subtitles_extractor.infrastructure.database.sqlite_video_state_repository import (
            SqliteVideoStateRepository,
        )
        import json, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            repo = SqliteVideoStateRepository(Path(tmp) / "test.db")
            json_no_align = json.dumps({
                "x": 10, "y": 50, "width": 100, "height": 40,
                # alignment absent
                "orientation": "HORIZONTAL",
            })
            roi = repo._deserialize_roi(json_no_align)
            assert roi is not None, "Thiếu alignment key → phải fallback, không discard"

    def test_valid_roi_deserializes_correctly(self) -> None:
        """ROI hợp lệ vẫn deserialize đúng sau bugfix."""
        from subtitles_extractor.infrastructure.database.sqlite_video_state_repository import (
            SqliteVideoStateRepository,
        )
        import json, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            repo = SqliteVideoStateRepository(Path(tmp) / "test.db")
            good_json = json.dumps({
                "x": 0, "y": 800, "width": 1920, "height": 280,
                "alignment": "CENTER",
                "orientation": "HORIZONTAL",
            })
            roi = repo._deserialize_roi(good_json)
            assert roi is not None
            assert roi.x == 0
            assert roi.y == 800
            assert roi.width == 1920


# ── AF-1: auto_fix count includes small gaps ────────────────────────────────

class TestAutoFixTimelineCount:
    def test_auto_fix_source_counts_small_gaps(self) -> None:
        """_on_auto_fix_timeline phải đếm cả small-gap (< 0.150s), không chỉ overlap."""
        src = _PAGE_SRC.read_text(encoding="utf-8")
        snippet = _snip(src, "_on_auto_fix_timeline")
        assert "0.150" in snippet or "0.15" in snippet, (
            "AF-1: _on_auto_fix_timeline phải đếm cả small positive gap < 0.150s, "
            "không chỉ overlap (gap < 0)."
        )

    def test_auto_fix_service_counts_match_behaviour(self) -> None:
        """auto_fix_timeline service phải sửa cả overlap lẫn small gap."""
        from subtitles_extractor.application.services.subtitle_editor_service import (
            SubtitleEditorService,
        )
        from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
        from subtitles_extractor.domain.value_objects.confidence import Confidence
        from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

        svc = SubtitleEditorService()
        svc.load([
            SubtitleEvent(1, "A", TimeInterval(0.0, 1.0), Confidence(0.9), 5),
            SubtitleEvent(2, "B", TimeInterval(1.05, 2.0), Confidence(0.9), 5),  # gap=0.05 < 0.15
        ])
        fixes = svc.auto_fix_timeline()
        assert fixes == 1, f"Phải fix 1 small-gap case, nhưng nhận {fixes}"
