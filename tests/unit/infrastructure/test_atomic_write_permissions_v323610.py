"""[v3.23.610] Ghi tệp nguyên tử không được làm tệp phụ đề thành "chỉ chủ sở hữu đọc".

``tempfile.mkstemp`` tạo tệp tạm quyền 0600; sau ``os.replace`` tệp phụ đề xuất ra mang
luôn quyền đó (Linux/macOS): máy chủ phim (Plex/Jellyfin chạy bằng tài khoản khác) không
đọc được, và tệp đã có quyền rộng hơn bị thu hẹp mỗi lần ghi lại. gemini-srt-translator
gặp và sửa đúng lỗi này ("Preserve output permissions for atomic writes").
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.subtitle.atomic_save import atomic_write_text

pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="Quyền POSIX không áp dụng trên Windows."
)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_new_file_follows_the_umask(tmp_path: Path) -> None:
    previous = os.umask(0o022)
    try:
        target = tmp_path / "phim.srt"
        atomic_write_text(target, "1\n", fsync_timeout_sec=0)
    finally:
        os.umask(previous)
    assert _mode(target) == 0o644


def test_rewriting_keeps_the_existing_permissions(tmp_path: Path) -> None:
    target = tmp_path / "phim.srt"
    target.write_text("cũ", encoding="utf-8")
    target.chmod(0o664)
    atomic_write_text(target, "mới", fsync_timeout_sec=0)
    assert target.read_text(encoding="utf-8") == "mới"
    assert _mode(target) == 0o664
