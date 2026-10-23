"""[v3.23.615] Tái dùng file video trên cloud không được "làm trẻ lại" mốc tải lên.

Gemini xoá file 48 giờ sau khi TẢI LÊN, không phải sau lần dùng cuối. Chẩn đoán dịch thật
14/10: ``files/6iip7kcgg24f`` tải lúc 08:27 nhưng phiên dịch ghi ``uploaded_at`` 09:23 —
giờ của lần TÁI DÙNG khi dịch. Bộ nhớ đệm tải video cũng ghi đè mốc mỗi lần chuẩn bị lại,
nên sau 48 giờ ứng dụng vẫn đem tên file đã bị xoá ra dò.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from subtitles_extractor.infrastructure.database.sqlite_translation_session_store import (
    CloudVideoFile,
    merge_cloud_files,
)
from subtitles_extractor.infrastructure.translation.gemini_video_context import (
    GeminiVideoContextProvider,
)

UPLOADED = datetime(2026, 10, 14, 1, 27, 3, tzinfo=UTC)
REUSED = UPLOADED + timedelta(minutes=56)


def _cloud(name: str, at: datetime) -> CloudVideoFile:
    return CloudVideoFile(name, 0.0, 176.85, at.isoformat())


class TestSessionCloudFiles:
    def test_reuse_keeps_the_original_upload_time(self) -> None:
        merged = merge_cloud_files(
            [_cloud("files/6iip7kcgg24f", UPLOADED)],
            [_cloud("files/6iip7kcgg24f", REUSED)],
            now=REUSED,
        )
        assert [cf.uploaded_at for cf in merged] == [UPLOADED.isoformat()]

    def test_new_upload_is_recorded_as_is(self) -> None:
        merged = merge_cloud_files(
            [_cloud("files/old", UPLOADED)], [_cloud("files/new", REUSED)], now=REUSED
        )
        assert [(cf.remote_name, cf.uploaded_at) for cf in merged] == [
            ("files/old", UPLOADED.isoformat()),
            ("files/new", REUSED.isoformat()),
        ]

    def test_expired_record_does_not_leak_its_time(self) -> None:
        later = UPLOADED + timedelta(hours=50)
        merged = merge_cloud_files(
            [_cloud("files/a", UPLOADED)], [_cloud("files/a", later)], now=later
        )
        assert [cf.uploaded_at for cf in merged] == [later.isoformat()]


class TestProviderUploadCache:
    def test_restoring_the_same_file_keeps_its_age(self, tmp_path: Path) -> None:
        db = tmp_path / "cache.db"
        provider = GeminiVideoContextProvider(api_key="k", cache_db_path=db)
        provider._cache_store("src-key", "files/a")
        old = (datetime.now(UTC) - timedelta(hours=48)).isoformat()
        with sqlite3.connect(db) as conn:
            conn.execute("UPDATE video_context_uploads SET created_at=?", (old,))
        provider._cache_store("src-key", "files/a")  # tái dùng: ghi lại cùng file
        assert provider._cache_lookup("src-key") is None

    def test_new_file_gets_a_fresh_age(self, tmp_path: Path) -> None:
        db = tmp_path / "cache.db"
        provider = GeminiVideoContextProvider(api_key="k", cache_db_path=db)
        provider._cache_store("src-key", "files/a")
        old = (datetime.now(UTC) - timedelta(hours=48)).isoformat()
        with sqlite3.connect(db) as conn:
            conn.execute("UPDATE video_context_uploads SET created_at=?", (old,))
        provider._cache_store("src-key", "files/b")  # tải lại: file mới
        assert provider._cache_lookup("src-key") == "files/b"
