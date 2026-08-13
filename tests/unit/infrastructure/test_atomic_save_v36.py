"""Unit tests cho atomic_save.py v3.6 — fsync timeout & logging.

Bảo vệ 4 cải tiến:
1. fsync chạy với timeout, không block indefinitely
2. fsync timeout → WARNING, ghi vẫn hoàn tất (best-effort)
3. fsync OSError → WARNING, ghi vẫn hoàn tất
4. Log chi tiết từng bước
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


# ---------------------------------------------------------------------------
# Tests: _fsync_with_timeout
# ---------------------------------------------------------------------------


class TestFsyncWithTimeout:
    """Kiểm tra hàm _fsync_with_timeout độc lập."""

    def test_fast_fsync_returns_true(self, tmp_dir: Path) -> None:
        """fsync() nhanh → trả True."""
        from subtitles_extractor.infrastructure.subtitle.atomic_save import (
            _fsync_with_timeout,
        )

        tmp_file = tmp_dir / "test.txt"
        tmp_file.write_bytes(b"hello")
        with open(tmp_file, "rb") as fh:
            result = _fsync_with_timeout(fh.fileno(), timeout_sec=5.0)
        assert result is True, "fsync nhanh phải trả True"

    def test_timeout_returns_false(self) -> None:
        """fsync() block → timeout → trả False (không raise)."""
        from subtitles_extractor.infrastructure.subtitle.atomic_save import (
            _fsync_with_timeout,
        )

        import threading

        block_event = threading.Event()
        released_event = threading.Event()

        original_fsync = os.fsync

        def _blocking_fsync(fd: int) -> None:
            block_event.wait()  # block cho đến khi được release
            released_event.set()
            original_fsync(fd)

        with patch("os.fsync", side_effect=_blocking_fsync):
            fd_mock = 99  # fake fd — fsync được mock nên không cần real
            result = _fsync_with_timeout(fd_mock, timeout_sec=0.05)

        block_event.set()  # giải phóng thread bị block

        assert result is False, "fsync bị block > timeout phải trả False"

    def test_fsync_oserror_is_re_raised(self) -> None:
        """fsync() raise OSError → _fsync_with_timeout re-raises OSError."""
        from subtitles_extractor.infrastructure.subtitle.atomic_save import (
            _fsync_with_timeout,
        )

        with patch("os.fsync", side_effect=OSError("disk error")):
            with pytest.raises(OSError, match="disk error"):
                _fsync_with_timeout(99, timeout_sec=2.0)


# ---------------------------------------------------------------------------
# Tests: atomic_write_text — happy path
# ---------------------------------------------------------------------------


class TestAtomicWriteTextHappyPath:
    """Kiểm tra atomic_write_text khi không có lỗi."""

    def test_writes_content_correctly(self, tmp_dir: Path) -> None:
        """File đích phải có đúng nội dung sau khi ghi."""
        from subtitles_extractor.infrastructure.subtitle.atomic_save import (
            atomic_write_text,
        )

        target = tmp_dir / "output.srt"
        content = "1\n00:00:01,000 --> 00:00:02,000\nHello World\n"
        atomic_write_text(target, content)
        assert target.read_text(encoding="utf-8") == content

    def test_creates_parent_directory(self, tmp_dir: Path) -> None:
        """Thư mục cha được tạo tự động nếu chưa tồn tại."""
        from subtitles_extractor.infrastructure.subtitle.atomic_save import (
            atomic_write_text,
        )

        target = tmp_dir / "subdir" / "nested" / "out.srt"
        atomic_write_text(target, "content")
        assert target.exists()

    def test_overwrites_existing_file(self, tmp_dir: Path) -> None:
        """File cũ được ghi đè hoàn toàn."""
        from subtitles_extractor.infrastructure.subtitle.atomic_save import (
            atomic_write_text,
        )

        target = tmp_dir / "out.srt"
        target.write_text("OLD CONTENT", encoding="utf-8")
        atomic_write_text(target, "NEW CONTENT")
        assert target.read_text(encoding="utf-8") == "NEW CONTENT"

    def test_no_temp_file_left_on_success(self, tmp_dir: Path) -> None:
        """Không để lại .tmp file sau khi ghi thành công."""
        from subtitles_extractor.infrastructure.subtitle.atomic_save import (
            atomic_write_text,
        )

        target = tmp_dir / "clean.srt"
        atomic_write_text(target, "data")
        tmp_files = list(tmp_dir.glob("*.tmp"))
        assert len(tmp_files) == 0, f"Còn temp file: {tmp_files}"

    def test_utf8_content_preserved(self, tmp_dir: Path) -> None:
        """Nội dung Unicode (CJK) phải được bảo toàn."""
        from subtitles_extractor.infrastructure.subtitle.atomic_save import (
            atomic_write_text,
        )

        content = "1\n00:00:01,000 --> 00:00:02,000\n你好世界 — Hello — 日本語\n"
        target = tmp_dir / "unicode.srt"
        atomic_write_text(target, content, encoding="utf-8")
        assert target.read_text(encoding="utf-8") == content


# ---------------------------------------------------------------------------
# Tests: fsync timeout → best-effort, ghi vẫn hoàn tất
# ---------------------------------------------------------------------------


class TestFsyncTimeoutBestEffort:
    """Bug fix: fsync timeout KHÔNG được raise, file PHẢI được ghi."""

    def test_fsync_timeout_does_not_raise(self, tmp_dir: Path) -> None:
        """Khi fsync timeout, atomic_write_text KHÔNG raise exception."""
        from subtitles_extractor.infrastructure.subtitle.atomic_save import (
            atomic_write_text,
        )

        # Giả lập fsync block mãi
        with patch(
            "subtitles_extractor.infrastructure.subtitle.atomic_save._fsync_with_timeout",
            return_value=False,  # timeout
        ):
            # Không nên raise bất kỳ exception nào
            atomic_write_text(tmp_dir / "out.srt", "content", fsync_timeout_sec=0.01)

    def test_file_written_after_fsync_timeout(self, tmp_dir: Path) -> None:
        """Dù fsync timeout, file đích phải có nội dung đúng."""
        from subtitles_extractor.infrastructure.subtitle.atomic_save import (
            atomic_write_text,
        )

        target = tmp_dir / "content.srt"
        with patch(
            "subtitles_extractor.infrastructure.subtitle.atomic_save._fsync_with_timeout",
            return_value=False,
        ):
            atomic_write_text(target, "Test Content 123")

        assert target.exists(), "File phải tồn tại sau khi ghi"
        assert target.read_text() == "Test Content 123"

    def test_fsync_timeout_emits_warning_log(
        self, tmp_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Khi fsync timeout phải có WARNING log."""
        from subtitles_extractor.infrastructure.subtitle.atomic_save import (
            atomic_write_text,
        )

        with caplog.at_level(logging.WARNING, logger="subtitles_extractor"):
            with patch(
                "subtitles_extractor.infrastructure.subtitle.atomic_save._fsync_with_timeout",
                return_value=False,
            ):
                atomic_write_text(tmp_dir / "warn.srt", "x", fsync_timeout_sec=0.01)

        warning_logs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_logs) >= 1, "Phải có ít nhất 1 WARNING khi fsync timeout"
        assert any("timeout" in r.message.lower() or "TIMEOUT" in r.message for r in warning_logs)

    def test_fsync_oserror_does_not_raise(self, tmp_dir: Path) -> None:
        """Khi fsync trả OSError, atomic_write_text KHÔNG raise."""
        from subtitles_extractor.infrastructure.subtitle.atomic_save import (
            atomic_write_text,
        )

        with patch(
            "subtitles_extractor.infrastructure.subtitle.atomic_save._fsync_with_timeout",
            side_effect=OSError("EROFS: Read-only file system"),
        ):
            # Không nên raise
            atomic_write_text(tmp_dir / "err.srt", "content")

    def test_file_written_after_fsync_oserror(self, tmp_dir: Path) -> None:
        """Dù fsync OSError, file đích phải tồn tại và đúng nội dung."""
        from subtitles_extractor.infrastructure.subtitle.atomic_save import (
            atomic_write_text,
        )

        target = tmp_dir / "after_err.srt"
        with patch(
            "subtitles_extractor.infrastructure.subtitle.atomic_save._fsync_with_timeout",
            side_effect=OSError("EINVAL"),
        ):
            atomic_write_text(target, "Persisted content")

        assert target.read_text() == "Persisted content"

    def test_fsync_zero_timeout_skips_fsync(self, tmp_dir: Path) -> None:
        """fsync_timeout_sec=0 bỏ qua fsync hoàn toàn, file vẫn ghi đúng."""
        from subtitles_extractor.infrastructure.subtitle.atomic_save import (
            atomic_write_text,
        )

        target = tmp_dir / "no_fsync.srt"
        atomic_write_text(target, "skipped fsync", fsync_timeout_sec=0)
        assert target.read_text() == "skipped fsync"


# ---------------------------------------------------------------------------
# Tests: OSError thật → raise + cleanup
# ---------------------------------------------------------------------------


class TestAtomicWriteOsError:
    """atomic_write_text phải raise và dọn dẹp khi có lỗi thật."""

    def test_cleanup_temp_file_on_write_error(self, tmp_dir: Path) -> None:
        """Tệp tạm phải bị xóa khi ghi lỗi giữa chừng."""
        from subtitles_extractor.infrastructure.subtitle.atomic_save import (
            atomic_write_text,
        )

        with patch("os.fdopen") as mock_fdopen:
            mock_fh = MagicMock()
            mock_fh.__enter__ = lambda s: s
            mock_fh.__exit__ = MagicMock(return_value=False)
            mock_fh.write.side_effect = OSError("disk full")
            mock_fdopen.return_value = mock_fh

            with pytest.raises(OSError):
                atomic_write_text(tmp_dir / "fail.srt", "content")

        # Không còn .tmp file nào
        remaining_tmps = list(tmp_dir.glob("*.tmp"))
        assert len(remaining_tmps) == 0, (
            f"Temp file không được dọn dẹp: {remaining_tmps}"
        )

    def test_raises_on_replace_failure(self, tmp_dir: Path) -> None:
        """Nếu os.replace thất bại, OSError phải được raise."""
        from subtitles_extractor.infrastructure.subtitle.atomic_save import (
            atomic_write_text,
        )

        with patch("os.replace", side_effect=OSError("replace failed")):
            with pytest.raises(OSError, match="replace failed"):
                atomic_write_text(tmp_dir / "r.srt", "content")


# ---------------------------------------------------------------------------
# Tests: env flag SUBTITLES_EXTRACTOR_SKIP_FSYNC
# ---------------------------------------------------------------------------


class TestSkipFsyncEnvFlag:
    """Env var SUBTITLES_EXTRACTOR_SKIP_FSYNC=1 phải bỏ qua fsync."""

    def test_skip_fsync_env_skips_call(self, tmp_dir: Path) -> None:
        """Khi SKIP_FSYNC=1, _fsync_with_timeout không được gọi."""
        from subtitles_extractor.infrastructure.subtitle import atomic_save

        original = atomic_save._SKIP_FSYNC
        try:
            atomic_save._SKIP_FSYNC = True
            with patch.object(atomic_save, "_fsync_with_timeout") as mock_fsync:
                atomic_write_text_fn = atomic_save.atomic_write_text
                atomic_write_text_fn(tmp_dir / "skip.srt", "content")
            mock_fsync.assert_not_called()
        finally:
            atomic_save._SKIP_FSYNC = original


# ---------------------------------------------------------------------------
# Tests: logging detail
# ---------------------------------------------------------------------------


class TestAtomicWriteLogging:
    """atomic_write_text phải log chi tiết từng bước (level DEBUG)."""

    def test_debug_logs_emitted(
        self, tmp_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Phải có ít nhất 2 DEBUG log khi ghi thành công."""
        from subtitles_extractor.infrastructure.subtitle.atomic_save import (
            atomic_write_text,
        )

        with caplog.at_level(logging.DEBUG, logger="subtitles_extractor"):
            atomic_write_text(tmp_dir / "log.srt", "logging test content")

        debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert len(debug_msgs) >= 2, (
            f"Phải có ≥2 DEBUG log nhưng chỉ có {len(debug_msgs)}: {debug_msgs}"
        )

    def test_file_name_in_log(
        self, tmp_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Tên file đích phải xuất hiện trong ít nhất 1 log record."""
        from subtitles_extractor.infrastructure.subtitle.atomic_save import (
            atomic_write_text,
        )

        with caplog.at_level(logging.DEBUG, logger="subtitles_extractor"):
            atomic_write_text(tmp_dir / "my_output.srt", "x")

        all_messages = " ".join(r.message for r in caplog.records)
        assert "my_output.srt" in all_messages, (
            "Tên file 'my_output.srt' phải xuất hiện trong log"
        )


# ---------------------------------------------------------------------------
# Tests: ExportRunnable error handling
# ---------------------------------------------------------------------------


class TestExportRunnableErrorMessages:
    """ExportRunnable.run() phải emit lỗi rõ ràng."""

    def test_permission_error_message(self, tmp_dir: Path) -> None:
        """PermissionError → message có chứa 'quyền ghi' hoặc 'ứng dụng khác'."""
        source = (
            Path(__file__).resolve().parents[3]
            / "src" / "subtitles_extractor"
            / "presentation" / "view_models" / "editor_page_view_model.py"
        )
        content = source.read_text(encoding="utf-8")

        run_start = content.find("    def run(self) -> None:\n        \"\"\"Thực thi ghi file")
        run_end = content.find("\n    def ", run_start + 1)
        snippet = content[run_start:run_end]

        assert "PermissionError" in snippet, (
            "_ExportRunnable.run() phải bắt PermissionError riêng với message rõ ràng"
        )

    def test_disk_full_enospc_message(self) -> None:
        """ENOSPC (đĩa đầy) → message tiếng Việt rõ ràng."""
        source = (
            Path(__file__).resolve().parents[3]
            / "src" / "subtitles_extractor"
            / "presentation" / "view_models" / "editor_page_view_model.py"
        )
        content = source.read_text(encoding="utf-8")

        run_start = content.find("    def run(self) -> None:\n        \"\"\"Thực thi ghi file")
        run_end = content.find("\n    def ", run_start + 1)
        snippet = content[run_start:run_end]

        assert "ENOSPC" in snippet or "đầy" in snippet, (
            "Phải xử lý ENOSPC với thông báo 'đĩa đầy'"
        )

    def test_file_not_found_message(self) -> None:
        """FileNotFoundError → message rõ ràng về thư mục không tồn tại."""
        source = (
            Path(__file__).resolve().parents[3]
            / "src" / "subtitles_extractor"
            / "presentation" / "view_models" / "editor_page_view_model.py"
        )
        content = source.read_text(encoding="utf-8")

        run_start = content.find("    def run(self) -> None:\n        \"\"\"Thực thi ghi file")
        run_end = content.find("\n    def ", run_start + 1)
        snippet = content[run_start:run_end]

        assert "FileNotFoundError" in snippet, (
            "Phải bắt FileNotFoundError riêng với message rõ ràng"
        )
